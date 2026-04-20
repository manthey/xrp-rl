# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn",
#   "pydantic",
# ]
# ///

import argparse
import asyncio
import json
import math
import os
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

FIELD_LENGTH_MM = 2175
FIELD_WIDTH_MM = 1150
GOAL_WIDTH_MM = 500
GOAL_DEPTH_MM = 120
CORNER_RADIUS_MM = 250
ROBOT_LENGTH_MM = 160
ROBOT_WIDTH_MM = 180
ROBOT_CORNER_RADIUS_MM = 20
BALL_DIAMETER_MM = 60
BALL_RADIUS_MM = BALL_DIAMETER_MM / 2
TAPE_WIDTH_MM = 25

WHEEL_DIAMETER_MM = 60.0
WHEEL_BASE_MM = 155.0
TICKS_PER_REV = 585
MM_PER_TICK = math.pi * WHEEL_DIAMETER_MM / TICKS_PER_REV
MAX_WHEEL_SPEED_MMPS = 250.0

FRICTION_PER_SEC = 40.0
RESTITUTION = 0.9
SIM_HZ = 60

TAPE_LINES = [
    {'x_mm': -FIELD_LENGTH_MM / 4, 'color': 'blue'},
    {'x_mm': FIELD_LENGTH_MM / 4, 'color': 'red'},
    {'x_mm': 0, 'color': 'white'},
    {'x_mm': -FIELD_LENGTH_MM / 2 + 1, 'color': 'blue'},
    {'x_mm': FIELD_LENGTH_MM / 2 - 1, 'color': 'red'},
]
FIELD_CONFIG = {
    'field_length_mm': FIELD_LENGTH_MM,
    'field_width_mm': FIELD_WIDTH_MM,
    'goal_width_mm': GOAL_WIDTH_MM,
    'goal_depth_mm': GOAL_DEPTH_MM,
    'corner_radius_mm': CORNER_RADIUS_MM,
    'robot_length_mm': ROBOT_LENGTH_MM,
    'robot_width_mm': ROBOT_WIDTH_MM,
    'robot_corner_radius_mm': ROBOT_CORNER_RADIUS_MM,
    'ball_diameter_mm': BALL_DIAMETER_MM,
    'tape_width_mm': TAPE_WIDTH_MM,
    'tape_lines': TAPE_LINES,
    'friction_per_sec': FRICTION_PER_SEC,
    'restitution': RESTITUTION,
    'wheel_base_mm': WHEEL_BASE_MM,
    'max_wheel_speed_mmps': MAX_WHEEL_SPEED_MMPS,
}
ball_state: dict = {
    'world_x_mm': 0.0,
    'world_y_mm': 0.0,
    'vel_x_mmps': 0.0,
    'vel_y_mmps': 0.0,
}
robots: dict[str, dict] = {}
websocket_clients: list[WebSocket] = []


class RobotPose:
    """
    Dead-reckoning pose tracker using differential drive kinematics.
    Suitable for use on the XRP (MicroPython) or in simulation.
    Call update_from_encoders each control cycle with current raw tick counts.
    Call correct_pose when a ground-truth position is known (tape line, wall).
    """

    def __init__(self, x_mm: float = 0.0, y_mm: float = 0.0, heading_deg: float = 0.0):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.heading_deg = heading_deg
        self.prev_left_ticks = 0
        self.prev_right_ticks = 0
        self.accuracy = 1.0

    def update_from_encoders(self, left_ticks: int, right_ticks: int):
        dl = (left_ticks - self.prev_left_ticks) * MM_PER_TICK
        dr = (right_ticks - self.prev_right_ticks) * MM_PER_TICK
        self.prev_left_ticks = left_ticks
        self.prev_right_ticks = right_ticks
        dist = (dl + dr) / 2.0
        d_heading_rad = (dr - dl) / WHEEL_BASE_MM
        heading_rad = math.radians(self.heading_deg)
        mid_heading = heading_rad + d_heading_rad / 2.0
        self.x_mm += dist * math.cos(mid_heading)
        self.y_mm += dist * math.sin(mid_heading)
        self.heading_deg = math.degrees(heading_rad + d_heading_rad) % 360
        self.accuracy = max(0.0, self.accuracy - abs(dist) * 0.0001)

    def correct_pose(self, x_mm: float, y_mm: float, heading_deg: float, accuracy: float = 1.0):
        self.x_mm = x_mm
        self.y_mm = y_mm
        self.heading_deg = heading_deg
        self.accuracy = accuracy

    def partial_correct_x(self, x_mm: float):
        self.x_mm = x_mm
        self.accuracy = min(1.0, self.accuracy + 0.3)

    def partial_correct_y(self, y_mm: float):
        self.y_mm = y_mm
        self.accuracy = min(1.0, self.accuracy + 0.3)


def closest_point_on_rounded_rect(px, py, cx, cy, half_len, half_wid, corner_r, heading_rad):
    cos_h = math.cos(-heading_rad)
    sin_h = math.sin(-heading_rad)
    lx = (px - cx) * cos_h - (py - cy) * sin_h
    ly = (px - cx) * sin_h + (py - cy) * cos_h
    inner_half_len = half_len - corner_r
    inner_half_wid = half_wid - corner_r
    clamped_x = max(-inner_half_len, min(inner_half_len, lx))
    clamped_y = max(-inner_half_wid, min(inner_half_wid, ly))
    ox, oy = lx - clamped_x, ly - clamped_y
    olen = math.sqrt(ox * ox + oy * oy)
    if olen > 0:
        scale = corner_r / olen
        surface_lx = clamped_x + ox * scale
        surface_ly = clamped_y + oy * scale
        inside = olen < corner_r
    else:
        surface_lx = clamped_x + corner_r
        surface_ly = clamped_y
        inside = True
    cos_fwd = math.cos(heading_rad)
    sin_fwd = math.sin(heading_rad)
    wx = cx + surface_lx * cos_fwd - surface_ly * sin_fwd
    wy = cy + surface_lx * sin_fwd + surface_ly * cos_fwd
    return wx, wy, inside


def point_in_field(px, py):
    half_len = FIELD_LENGTH_MM / 2
    half_wid = FIELD_WIDTH_MM / 2
    goal_half = GOAL_WIDTH_MM / 2
    if abs(px) > half_len and abs(py) < goal_half:
        if abs(px) <= half_len + GOAL_DEPTH_MM:
            return True
        return False
    if abs(py) > half_wid:
        return False
    if abs(py) >= goal_half and abs(px) > half_len:
        return False
    inner_x = half_len - CORNER_RADIUS_MM
    inner_y = half_wid - CORNER_RADIUS_MM
    for ccx in (-inner_x, inner_x):
        for ccy in (-inner_y, inner_y):
            if (px - ccx) * ccx > 0 and (py - ccy) * ccy > 0:
                dx, dy = px - ccx, py - ccy
                if dx * dx + dy * dy > CORNER_RADIUS_MM * CORNER_RADIUS_MM:
                    return False
    return True


def robot_corners(rx, ry, heading_rad, half_len, half_wid):
    cos_h = math.cos(heading_rad)
    sin_h = math.sin(heading_rad)
    corners = []
    for lx in (-half_len, half_len):
        for ly in (-half_wid, half_wid):
            wx = rx + lx * cos_h - ly * sin_h
            wy = ry + lx * sin_h + ly * cos_h
            corners.append((wx, wy))
    return corners


def robots_overlap(r1, r2):
    rx1 = r1.get('world_x_mm')
    ry1 = r1.get('world_y_mm')
    rx2 = r2.get('world_x_mm')
    ry2 = r2.get('world_y_mm')
    if rx1 is None or ry1 is None or rx2 is None or ry2 is None:
        return False
    heading1 = math.radians(r1.get('world_heading_deg', 0))
    heading2 = math.radians(r2.get('world_heading_deg', 0))
    half_len = ROBOT_LENGTH_MM / 2
    half_wid = ROBOT_WIDTH_MM / 2
    corners1 = robot_corners(rx1, ry1, heading1, half_len, half_wid)
    corners2 = robot_corners(rx2, ry2, heading2, half_len, half_wid)
    for cx, cy in corners1:
        _, _, inside = closest_point_on_rounded_rect(
            cx, cy, rx2, ry2, half_len, half_wid, ROBOT_CORNER_RADIUS_MM, heading2)
        if inside:
            return True
    for cx, cy in corners2:
        _, _, inside = closest_point_on_rounded_rect(
            cx, cy, rx1, ry1, half_len, half_wid, ROBOT_CORNER_RADIUS_MM, heading1)
        if inside:
            return True
    return False


def constrain_robot_to_field(robot):
    rx = robot.get('world_x_mm', 0.0)
    ry = robot.get('world_y_mm', 0.0)
    rh = math.radians(robot.get('world_heading_deg', 0.0))
    half_len = ROBOT_LENGTH_MM / 2
    half_wid = ROBOT_WIDTH_MM / 2
    corners = robot_corners(rx, ry, rh, half_len, half_wid)
    all_in = all(point_in_field(cx, cy) for cx, cy in corners)
    if all_in:
        return
    max_shift = max(ROBOT_LENGTH_MM, ROBOT_WIDTH_MM)
    max_h = math.radians(5)
    best_rx, best_ry, best_rh = rx, ry, rh
    best_distance = float('inf')
    while max_shift >= 0.5:
        last_rx, last_ry, last_rh = best_rx, best_ry, best_rh
        for oy in range(3):
            for ox in range(3):
                for oh in range(3):
                    test_x = last_rx + (ox if ox != 2 else -1) * max_shift
                    test_y = last_ry + (oy if oy != 2 else -1) * max_shift
                    test_h = last_rh + (oh if oh != 2 else -1) * max_h
                    test_corners = robot_corners(test_x, test_y, test_h, half_len, half_wid)
                    if all(point_in_field(cx, cy) for cx, cy in test_corners):
                        dist = (test_x - rx) ** 2 + (test_y - ry) ** 2
                        if dist < best_distance:
                            best_distance = dist
                            best_rx = test_x
                            best_ry = test_y
                            best_rh = test_h
        max_shift /= 2
        max_h /= 2
    robot['world_x_mm'] = best_rx
    robot['world_y_mm'] = best_ry
    robot['world_heading_deg'] = math.degrees(best_rh)


def resolve_robot_overlaps():
    robot_list = list(robots.values())
    for i in range(len(robot_list)):
        for j in range(i + 1, len(robot_list)):
            r1 = robot_list[i]
            r2 = robot_list[j]
            if robots_overlap(r1, r2):
                rx1 = r1.get('world_x_mm', 0.0)
                ry1 = r1.get('world_y_mm', 0.0)
                rx2 = r2.get('world_x_mm', 0.0)
                ry2 = r2.get('world_y_mm', 0.0)
                dx = rx2 - rx1
                dy = ry2 - ry1
                dist = math.sqrt(dx * dx + dy * dy)
                if dist > 0:
                    nx, ny = dx / dist, dy / dist
                else:
                    nx, ny = 1.0, 0.0
                separation = (ROBOT_LENGTH_MM + ROBOT_WIDTH_MM) / 2
                push = separation / 2
                r1['world_x_mm'] = rx1 - nx * push
                r1['world_y_mm'] = ry1 - ny * push
                r2['world_x_mm'] = rx2 + nx * push
                r2['world_y_mm'] = ry2 + ny * push
                constrain_robot_to_field(r1)
                constrain_robot_to_field(r2)


def field_boundary_response(bx, by, vx, vy, radius):  # noqa
    half_len = FIELD_LENGTH_MM / 2
    half_wid = FIELD_WIDTH_MM / 2
    goal_half = GOAL_WIDTH_MM / 2
    if abs(bx) > half_len and abs(by) < goal_half:
        sign = 1 if bx > 0 else -1
        back = sign * (half_len + GOAL_DEPTH_MM)
        if sign * bx + radius > sign * back:
            bx = back - sign * radius
            vx = -sign * abs(vx) * RESTITUTION
        for ws in (-1, 1):
            if ws * by + radius > goal_half:
                by = ws * (goal_half - radius)
                vy = -ws * abs(vy) * RESTITUTION
        return bx, by, vx, vy
    for ws in (-1, 1):
        if ws * by + radius > half_wid:
            by = ws * (half_wid - radius)
            vy = -ws * abs(vy) * RESTITUTION
    if abs(by) >= goal_half:
        for ws in (-1, 1):
            if ws * bx + radius > half_len:
                bx = ws * (half_len - radius)
                vx = -ws * abs(vx) * RESTITUTION
    inner_x = half_len - CORNER_RADIUS_MM
    inner_y = half_wid - CORNER_RADIUS_MM
    boundary = CORNER_RADIUS_MM - radius
    for ccx in (-inner_x, inner_x):
        for ccy in (-inner_y, inner_y):
            if (bx - ccx) * ccx <= 0 or (by - ccy) * ccy <= 0:
                continue
            dx, dy = bx - ccx, by - ccy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > boundary and dist > 0:
                nx, ny = -dx / dist, -dy / dist
                bx = ccx - nx * boundary
                by = ccy - ny * boundary
                dot = vx * nx + vy * ny
                if dot < 0:
                    vx -= (1 + RESTITUTION) * dot * nx
                    vy -= (1 + RESTITUTION) * dot * ny
    return bx, by, vx, vy


def collide_ball_with_robot(bx, by, vx, vy, robot):
    rx = robot.get('world_x_mm')
    ry = robot.get('world_y_mm')
    if rx is None or ry is None:
        return bx, by, vx, vy
    heading_rad = math.radians(robot.get('world_heading_deg', 0))
    cpx, cpy, inside = closest_point_on_rounded_rect(
        bx, by, rx, ry, ROBOT_LENGTH_MM / 2, ROBOT_WIDTH_MM / 2,
        ROBOT_CORNER_RADIUS_MM, heading_rad)
    dx, dy = bx - cpx, by - cpy
    dist = math.sqrt(dx * dx + dy * dy)
    if not inside and dist >= BALL_RADIUS_MM:
        return bx, by, vx, vy
    if dist > 0:
        sign = -1.0 if inside else 1.0
        nx, ny = sign * dx / dist, sign * dy / dist
    else:
        nx, ny = 1.0, 0.0
    penetration = BALL_RADIUS_MM + dist if inside else BALL_RADIUS_MM - dist
    bx += nx * penetration
    by += ny * penetration
    vl = robot.get('cmd_vel_left', 0.0)
    vr = robot.get('cmd_vel_right', 0.0)
    forward_speed = (vl + vr) / 2.0
    angular_speed = (vr - vl) / WHEEL_BASE_MM
    fwd_x, fwd_y = math.cos(heading_rad), math.sin(heading_rad)
    contact_rx, contact_ry = cpx - rx, cpy - ry
    robot_vx = forward_speed * fwd_x - angular_speed * contact_ry
    robot_vy = forward_speed * fwd_y + angular_speed * contact_rx
    rel_dot = (vx - robot_vx) * nx + (vy - robot_vy) * ny
    if rel_dot < 0:
        impulse = -(1 + RESTITUTION) * rel_dot
        vx += impulse * nx
        vy += impulse * ny
    return bx, by, vx, vy


def apply_friction(vx, vy, dt):
    speed = math.sqrt(vx * vx + vy * vy)
    if speed == 0:
        return 0.0, 0.0
    new_speed = max(0.0, speed - FRICTION_PER_SEC * dt)
    factor = new_speed / speed
    return vx * factor, vy * factor


def step_virtual_robot(robot: dict, dt: float):
    vl = robot.get('cmd_vel_left', 0.0)
    vr = robot.get('cmd_vel_right', 0.0)
    if vl == 0.0 and vr == 0.0:
        return
    heading_rad = math.radians(robot.get('world_heading_deg', 0.0))
    dist = (vl + vr) / 2.0 * dt
    d_heading = (vr - vl) / WHEEL_BASE_MM * dt
    mid_heading = heading_rad + d_heading / 2.0
    robot['world_x_mm'] = robot.get('world_x_mm', 0.0) + dist * math.cos(mid_heading)
    robot['world_y_mm'] = robot.get('world_y_mm', 0.0) + dist * math.sin(mid_heading)
    robot['world_heading_deg'] = math.degrees(heading_rad + d_heading) % 360
    robot['left_encoder'] = robot.get('left_encoder', 0.0) + vl * dt / MM_PER_TICK
    robot['right_encoder'] = robot.get('right_encoder', 0.0) + vr * dt / MM_PER_TICK


def clamp_speed(v: float) -> float:
    return max(-MAX_WHEEL_SPEED_MMPS, min(MAX_WHEEL_SPEED_MMPS, v))


async def simulation_loop():
    dt = 1.0 / SIM_HZ
    while True:
        await asyncio.sleep(dt)
        for robot in robots.values():
            if robot.get('virtual'):
                step_virtual_robot(robot, dt)
        for robot in robots.values():
            constrain_robot_to_field(robot)
        resolve_robot_overlaps()
        vx = ball_state['vel_x_mmps']
        vy = ball_state['vel_y_mmps']
        bx = ball_state['world_x_mm']
        by = ball_state['world_y_mm']
        bx += vx * dt
        by += vy * dt
        for robot in robots.values():
            bx, by, vx, vy = collide_ball_with_robot(bx, by, vx, vy, robot)
        bx, by, vx, vy = field_boundary_response(bx, by, vx, vy, BALL_RADIUS_MM)
        vx, vy = apply_friction(vx, vy, dt)
        ball_state['world_x_mm'] = bx
        ball_state['world_y_mm'] = by
        ball_state['vel_x_mmps'] = vx
        ball_state['vel_y_mmps'] = vy
        if abs(vx) < 0.5 and abs(vy) < 0.5:
            ball_state['vel_x_mmps'] = 0.0
            ball_state['vel_y_mmps'] = 0.0
        virtual_updates = {
            rid: dict(rob) for rid, rob in robots.items() if rob.get('virtual')
        }
        if virtual_updates:
            await broadcast({'type': 'virtual_robots', 'data': virtual_updates})
        await broadcast({'type': 'ball', 'data': dict(ball_state)})


@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(simulation_loop())
    yield
    task.cancel()
    try:
        await task
    except asyncio.CancelledError:
        pass


app = FastAPI(lifespan=lifespan)


class Telemetry(BaseModel):
    robot_id: str
    left_encoder: float
    right_encoder: float
    distance_cm: float
    reflectance_left: float
    reflectance_right: float
    cmd_vel_left: float
    cmd_vel_right: float
    world_x_mm: float | None = None
    world_y_mm: float | None = None
    world_heading_deg: float | None = None
    pose_accuracy: float | None = None


class BallState(BaseModel):
    world_x_mm: float
    world_y_mm: float
    vel_x_mmps: float = 0.0
    vel_y_mmps: float = 0.0


class PoseOverride(BaseModel):
    robot_id: str
    world_x_mm: float
    world_y_mm: float
    world_heading_deg: float


class RobotCommand(BaseModel):
    robot_id: str
    cmd_vel_left: float
    cmd_vel_right: float


@app.post('/telemetry')
async def receive_telemetry(telemetry: Telemetry):
    data = telemetry.model_dump()
    if telemetry.robot_id not in robots:
        robots[telemetry.robot_id] = {}
    robots[telemetry.robot_id].update(data)
    await broadcast({'type': 'telemetry', 'data': data})
    return {'status': 'ok'}


@app.post('/ball')
async def update_ball(state: BallState):
    ball_state.update(state.model_dump())
    await broadcast({'type': 'ball', 'data': dict(ball_state)})
    return {'status': 'ok'}


@app.post('/pose_override')
async def override_pose(override: PoseOverride):
    robot_id = override.robot_id
    if robot_id not in robots:
        robots[robot_id] = {
            'robot_id': robot_id,
            'virtual': True,
            'cmd_vel_left': 0.0,
            'cmd_vel_right': 0.0,
            'left_encoder': 0.0,
            'right_encoder': 0.0,
            'distance_cm': 0.0,
            'reflectance_left': 1.0,
            'reflectance_right': 1.0,
        }
    robots[robot_id]['world_x_mm'] = override.world_x_mm
    robots[robot_id]['world_y_mm'] = override.world_y_mm
    robots[robot_id]['world_heading_deg'] = override.world_heading_deg
    constrain_robot_to_field(robots[robot_id])
    resolve_robot_overlaps()
    await broadcast({'type': 'pose_override', 'data': override.model_dump()})
    return {'status': 'ok'}


@app.post('/cmd_vel')
async def command_velocity(cmd: RobotCommand):
    robot_id = cmd.robot_id
    if robot_id in robots and robots[robot_id].get('virtual'):
        robots[robot_id]['cmd_vel_left'] = clamp_speed(cmd.cmd_vel_left)
        robots[robot_id]['cmd_vel_right'] = clamp_speed(cmd.cmd_vel_right)
    return {'status': 'ok'}


@app.get('/robots')
async def get_robots():
    return robots


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        await ws.send_json({'type': 'init', 'robots': robots, 'ball': ball_state})
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get('type') == 'cmd_vel':
                robot_id = msg.get('robot_id')
                if robot_id in robots and robots[robot_id].get('virtual'):
                    robots[robot_id]['cmd_vel_left'] = clamp_speed(
                        float(msg.get('cmd_vel_left', 0)))
                    robots[robot_id]['cmd_vel_right'] = clamp_speed(
                        float(msg.get('cmd_vel_right', 0)))
    except WebSocketDisconnect:
        websocket_clients.remove(ws)


async def broadcast(message: dict):
    disconnected = []
    for ws in websocket_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        websocket_clients.remove(ws)


def build_html(config: dict) -> str:
    html = open(os.path.join(os.path.dirname(__file__), 'render.html')).read()
    return html.replace('CONFIG_JSON', json.dumps(config))


@app.get('/', response_class=HTMLResponse)
async def index():
    return build_html(FIELD_CONFIG)


def main():
    parser = argparse.ArgumentParser(description='XRP Soccer Field Telemetry Visualizer')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', '-p', type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == '__main__':
    main()
