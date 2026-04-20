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
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import uvicorn

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
WHEEL_RADIUS_MM = WHEEL_DIAMETER_MM / 2
WHEEL_BASE_MM = 155.0
TICKS_PER_REV = 585
MM_PER_TICK = math.pi * WHEEL_DIAMETER_MM / TICKS_PER_REV
MAX_WHEEL_SPEED_MMPS = 250.0

FRICTION_PER_SEC = 40.0
RESTITUTION = 0.9
SIM_HZ = 60

TAPE_LINES = [
    {"x_mm": -FIELD_LENGTH_MM / 4,     "color": "blue"},
    {"x_mm":  FIELD_LENGTH_MM / 4,     "color": "red"},
    {"x_mm":  0,                        "color": "white"},
    {"x_mm": -FIELD_LENGTH_MM / 2 + 1, "color": "blue"},
    {"x_mm":  FIELD_LENGTH_MM / 2 - 1, "color": "red"},
]

FIELD_CONFIG = {
    "field_length_mm":        FIELD_LENGTH_MM,
    "field_width_mm":         FIELD_WIDTH_MM,
    "goal_width_mm":          GOAL_WIDTH_MM,
    "goal_depth_mm":          GOAL_DEPTH_MM,
    "corner_radius_mm":       CORNER_RADIUS_MM,
    "robot_length_mm":        ROBOT_LENGTH_MM,
    "robot_width_mm":         ROBOT_WIDTH_MM,
    "robot_corner_radius_mm": ROBOT_CORNER_RADIUS_MM,
    "ball_diameter_mm":       BALL_DIAMETER_MM,
    "tape_width_mm":          TAPE_WIDTH_MM,
    "tape_lines":             TAPE_LINES,
    "friction_per_sec":       FRICTION_PER_SEC,
    "restitution":            RESTITUTION,
    "wheel_base_mm":          WHEEL_BASE_MM,
    "max_wheel_speed_mmps":   MAX_WHEEL_SPEED_MMPS,
}


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
        dl = (left_ticks  - self.prev_left_ticks)  * MM_PER_TICK
        dr = (right_ticks - self.prev_right_ticks) * MM_PER_TICK
        self.prev_left_ticks  = left_ticks
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


def closest_point_on_segment(px, py, ax, ay, bx, by):
    dx, dy = bx - ax, by - ay
    seg_len_sq = dx * dx + dy * dy
    if seg_len_sq == 0:
        return ax, ay
    t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / seg_len_sq))
    return ax + t * dx, ay + t * dy


def closest_point_on_rounded_rect(px, py, cx, cy, half_len, half_wid, corner_r, heading_rad):
    cos_h = math.cos(-heading_rad)
    sin_h = math.sin(-heading_rad)
    lx = (px - cx) * cos_h - (py - cy) * sin_h
    ly = (px - cx) * sin_h + (py - cy) * cos_h

    inner_half_len = half_len - corner_r
    inner_half_wid = half_wid - corner_r

    clamped_x = max(-inner_half_len, min(inner_half_len, lx))
    clamped_y = max(-inner_half_wid, min(inner_half_wid, ly))

    cos_fwd = math.cos(heading_rad)
    sin_fwd = math.sin(heading_rad)
    wx = cx + clamped_x * cos_fwd - clamped_y * sin_fwd
    wy = cy + clamped_x * sin_fwd + clamped_y * cos_fwd
    return wx, wy


def ball_inside_rounded_rect(px, py, cx, cy, half_len, half_wid, corner_r, heading_rad):
    cos_h = math.cos(-heading_rad)
    sin_h = math.sin(-heading_rad)
    lx = (px - cx) * cos_h - (py - cy) * sin_h
    ly = (px - cx) * sin_h + (py - cy) * cos_h
    inner_half_len = half_len - corner_r
    inner_half_wid = half_wid - corner_r
    clamped_x = max(-inner_half_len, min(inner_half_len, lx))
    clamped_y = max(-inner_half_wid, min(inner_half_wid, ly))
    dx = lx - clamped_x
    dy = ly - clamped_y
    return math.sqrt(dx * dx + dy * dy) <= corner_r


def field_boundary_response(bx, by, vx, vy, radius):
    half_len = FIELD_LENGTH_MM / 2
    half_wid = FIELD_WIDTH_MM / 2
    goal_half = GOAL_WIDTH_MM / 2
    goal_depth = GOAL_DEPTH_MM

    in_left_goal  = bx < -half_len and abs(by) < goal_half
    in_right_goal = bx >  half_len and abs(by) < goal_half

    new_bx, new_by, new_vx, new_vy = bx, by, vx, vy

    if in_left_goal:
        left_wall = -half_len - goal_depth
        top_wall  = -goal_half
        bot_wall  =  goal_half
        if new_bx - radius < left_wall:
            new_bx = left_wall + radius
            new_vx = abs(new_vx) * RESTITUTION
        if new_by - radius < top_wall:
            new_by = top_wall + radius
            new_vy = abs(new_vy) * RESTITUTION
        if new_by + radius > bot_wall:
            new_by = bot_wall - radius
            new_vy = -abs(new_vy) * RESTITUTION
        return new_bx, new_by, new_vx, new_vy

    if in_right_goal:
        right_wall =  half_len + goal_depth
        top_wall   = -goal_half
        bot_wall   =  goal_half
        if new_bx + radius > right_wall:
            new_bx = right_wall - radius
            new_vx = -abs(new_vx) * RESTITUTION
        if new_by - radius < top_wall:
            new_by = top_wall + radius
            new_vy = abs(new_vy) * RESTITUTION
        if new_by + radius > bot_wall:
            new_by = bot_wall - radius
            new_vy = -abs(new_vy) * RESTITUTION
        return new_bx, new_by, new_vx, new_vy

    corner_r = CORNER_RADIUS_MM
    inner_half_len = half_len - corner_r
    inner_half_wid = half_wid - corner_r

    if new_by - radius < -half_wid:
        new_by = -half_wid + radius
        new_vy = abs(new_vy) * RESTITUTION
    if new_by + radius > half_wid:
        new_by = half_wid - radius
        new_vy = -abs(new_vy) * RESTITUTION

    entering_left_goal  = new_bx - radius < -half_len and abs(new_by) < goal_half
    entering_right_goal = new_bx + radius >  half_len and abs(new_by) < goal_half

    if not entering_left_goal and new_bx - radius < -half_len:
        new_bx = -half_len + radius
        new_vx = abs(new_vx) * RESTITUTION
    if not entering_right_goal and new_bx + radius > half_len:
        new_bx = half_len - radius
        new_vx = -abs(new_vx) * RESTITUTION

    corner_centers = [
        (-inner_half_len, -inner_half_wid),
        ( inner_half_len, -inner_half_wid),
        ( inner_half_len,  inner_half_wid),
        (-inner_half_len,  inner_half_wid),
    ]
    for (ccx, ccy) in corner_centers:
        in_corner_quadrant = (
            (new_bx < ccx if ccx < 0 else new_bx > ccx) and
            (new_by < ccy if ccy < 0 else new_by > ccy)
        )
        if not in_corner_quadrant:
            continue
        dist = math.sqrt((new_bx - ccx) ** 2 + (new_by - ccy) ** 2)
        boundary = corner_r - radius
        if dist > boundary and dist > 0:
            nx = (ccx - new_bx) / dist
            ny = (ccy - new_by) / dist
            new_bx = ccx - nx * boundary
            new_by = ccy - ny * boundary
            dot = new_vx * nx + new_vy * ny
            if dot < 0:
                new_vx -= (1 + RESTITUTION) * dot * nx
                new_vy -= (1 + RESTITUTION) * dot * ny

    return new_bx, new_by, new_vx, new_vy


def collide_ball_with_robot(bx, by, vx, vy, robot):
    rx = robot.get("world_x_mm")
    ry = robot.get("world_y_mm")
    if rx is None or ry is None:
        return bx, by, vx, vy

    heading_rad = math.radians(robot.get("world_heading_deg", 0))
    half_len = ROBOT_LENGTH_MM / 2
    half_wid = ROBOT_WIDTH_MM / 2
    corner_r = ROBOT_CORNER_RADIUS_MM

    cpx, cpy = closest_point_on_rounded_rect(bx, by, rx, ry, half_len, half_wid, corner_r, heading_rad)
    dx = bx - cpx
    dy = by - cpy
    dist = math.sqrt(dx * dx + dy * dy)

    if dist == 0 or dist >= BALL_RADIUS_MM:
        return bx, by, vx, vy

    nx = dx / dist
    ny = dy / dist
    penetration = BALL_RADIUS_MM - dist
    new_bx = bx + nx * penetration
    new_by = by + ny * penetration

    dot = vx * nx + vy * ny
    if dot < 0:
        new_vx = vx - (1 + RESTITUTION) * dot * nx
        new_vy = vy - (1 + RESTITUTION) * dot * ny
    else:
        new_vx, new_vy = vx, vy

    return new_bx, new_by, new_vx, new_vy


def apply_friction(vx, vy, dt):
    speed = math.sqrt(vx * vx + vy * vy)
    if speed == 0:
        return 0.0, 0.0
    reduction = FRICTION_PER_SEC * dt
    new_speed = max(0.0, speed - reduction)
    factor = new_speed / speed
    return vx * factor, vy * factor


def step_virtual_robot(robot: dict, dt: float):
    vl = robot.get("cmd_vel_left", 0.0)
    vr = robot.get("cmd_vel_right", 0.0)
    if vl == 0.0 and vr == 0.0:
        return

    heading_rad = math.radians(robot.get("world_heading_deg", 0.0))
    dist = (vl + vr) / 2.0 * dt
    d_heading = (vr - vl) / WHEEL_BASE_MM * dt

    mid_heading = heading_rad + d_heading / 2.0
    robot["world_x_mm"] = robot.get("world_x_mm", 0.0) + dist * math.cos(mid_heading)
    robot["world_y_mm"] = robot.get("world_y_mm", 0.0) + dist * math.sin(mid_heading)
    robot["world_heading_deg"] = math.degrees(heading_rad + d_heading) % 360

    ticks_per_sec = MAX_WHEEL_SPEED_MMPS / MM_PER_TICK
    robot["left_encoder"]  = robot.get("left_encoder",  0.0) + vl * dt / MM_PER_TICK
    robot["right_encoder"] = robot.get("right_encoder", 0.0) + vr * dt / MM_PER_TICK


ball_state: dict = {
    "world_x_mm": 0.0,
    "world_y_mm": 0.0,
    "vel_x_mmps": 0.0,
    "vel_y_mmps": 0.0,
}
robots: dict[str, dict] = {}
websocket_clients: list[WebSocket] = []


async def simulation_loop():
    dt = 1.0 / SIM_HZ
    while True:
        await asyncio.sleep(dt)

        for robot in robots.values():
            if robot.get("virtual"):
                step_virtual_robot(robot, dt)

        vx = ball_state["vel_x_mmps"]
        vy = ball_state["vel_y_mmps"]
        bx = ball_state["world_x_mm"]
        by = ball_state["world_y_mm"]

        if abs(vx) < 0.5 and abs(vy) < 0.5:
            ball_state["vel_x_mmps"] = 0.0
            ball_state["vel_y_mmps"] = 0.0
        else:
            bx += vx * dt
            by += vy * dt

            for robot in robots.values():
                bx, by, vx, vy = collide_ball_with_robot(bx, by, vx, vy, robot)

            bx, by, vx, vy = field_boundary_response(bx, by, vx, vy, BALL_RADIUS_MM)
            vx, vy = apply_friction(vx, vy, dt)

            ball_state["world_x_mm"] = bx
            ball_state["world_y_mm"] = by
            ball_state["vel_x_mmps"] = vx
            ball_state["vel_y_mmps"] = vy

        virtual_updates = {
            rid: {k: v for k, v in rob.items()}
            for rid, rob in robots.items()
            if rob.get("virtual")
        }
        if virtual_updates:
            await broadcast({"type": "virtual_robots", "data": virtual_updates})
        await broadcast({"type": "ball", "data": dict(ball_state)})


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


@app.post("/telemetry")
async def receive_telemetry(telemetry: Telemetry):
    data = telemetry.model_dump()
    if telemetry.robot_id not in robots:
        robots[telemetry.robot_id] = {}
    robots[telemetry.robot_id].update(data)
    await broadcast({"type": "telemetry", "data": data})
    return {"status": "ok"}


@app.post("/ball")
async def update_ball(state: BallState):
    ball_state.update(state.model_dump())
    await broadcast({"type": "ball", "data": dict(ball_state)})
    return {"status": "ok"}


@app.post("/pose_override")
async def override_pose(override: PoseOverride):
    robot_id = override.robot_id
    if robot_id not in robots:
        robots[robot_id] = {
            "robot_id":        robot_id,
            "virtual":         True,
            "cmd_vel_left":    0.0,
            "cmd_vel_right":   0.0,
            "left_encoder":    0.0,
            "right_encoder":   0.0,
            "distance_cm":     0.0,
            "reflectance_left":  1.0,
            "reflectance_right": 1.0,
        }
    robots[robot_id]["world_x_mm"]        = override.world_x_mm
    robots[robot_id]["world_y_mm"]        = override.world_y_mm
    robots[robot_id]["world_heading_deg"] = override.world_heading_deg
    await broadcast({"type": "pose_override", "data": override.model_dump()})
    return {"status": "ok"}


@app.post("/cmd_vel")
async def command_velocity(cmd: RobotCommand):
    robot_id = cmd.robot_id
    if robot_id in robots and robots[robot_id].get("virtual"):
        robots[robot_id]["cmd_vel_left"]  = max(-MAX_WHEEL_SPEED_MMPS, min(MAX_WHEEL_SPEED_MMPS, cmd.cmd_vel_left))
        robots[robot_id]["cmd_vel_right"] = max(-MAX_WHEEL_SPEED_MMPS, min(MAX_WHEEL_SPEED_MMPS, cmd.cmd_vel_right))
    return {"status": "ok"}


@app.get("/robots")
async def get_robots():
    return robots


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        await ws.send_json({"type": "init", "robots": robots, "ball": ball_state})
        while True:
            raw = await ws.receive_text()
            msg = json.loads(raw)
            if msg.get("type") == "cmd_vel":
                robot_id = msg.get("robot_id")
                if robot_id in robots and robots[robot_id].get("virtual"):
                    vl = max(-MAX_WHEEL_SPEED_MMPS, min(MAX_WHEEL_SPEED_MMPS, float(msg.get("cmd_vel_left", 0))))
                    vr = max(-MAX_WHEEL_SPEED_MMPS, min(MAX_WHEEL_SPEED_MMPS, float(msg.get("cmd_vel_right", 0))))
                    robots[robot_id]["cmd_vel_left"]  = vl
                    robots[robot_id]["cmd_vel_right"] = vr
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
    config_json = json.dumps(config)
    return r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>XRP Soccer Field</title>
<style>
  * { box-sizing: border-box; }
  body {
    background: #1a1a1a;
    color: #e0e0e0;
    font-family: monospace;
    margin: 0;
    padding: 0;
  }
  #canvas-container { width: 100vw; overflow: hidden; }
  canvas { display: block; background: #2d6a2d; width: 100%; height: auto; }
  #panel { padding: 10px; }
  table { border-collapse: collapse; width: 100%; font-size: 11px; }
  td, th { border: 1px solid #444; padding: 3px 6px; }
  th { background: #333; }
  .section-title { margin: 10px 0 4px; font-weight: bold; font-size: 13px; }
  input[type=number] { width: 70px; background: #333; color: #e0e0e0; border: 1px solid #666; }
  input[type=text]   { background: #333; color: #e0e0e0; border: 1px solid #666; }
  button { background: #555; color: #e0e0e0; border: 1px solid #888; cursor: pointer; padding: 2px 8px; }
  button:hover { background: #777; }
  #ball-info { font-size: 12px; margin: 4px 0; }
  #joysticks { display: flex; flex-wrap: wrap; gap: 16px; margin-top: 8px; }
  .joystick-wrap { display: flex; flex-direction: column; align-items: center; gap: 4px; }
  .joystick-label { font-size: 11px; }
  .joystick-canvas { touch-action: none; cursor: grab; border: 1px solid #555; border-radius: 4px; background: #222; display: block; }
</style>
</head>
<body>
<div id="canvas-container">
  <canvas id="field"></canvas>
</div>
<div id="panel">
  <div class="section-title">Ball State</div>
  <div id="ball-info">x: - &nbsp; y: - &nbsp; vx: - &nbsp; vy: -</div>
  <div class="section-title">Robots</div>
  <table id="robot-table">
    <thead><tr>
      <th>ID</th><th>X mm</th><th>Y mm</th><th>Hdg</th>
      <th>Dist cm</th><th>Refl L</th><th>Refl R</th>
      <th>Enc L</th><th>Enc R</th><th>Virt</th>
    </tr></thead>
    <tbody id="robot-tbody"></tbody>
  </table>
  <div class="section-title">Pose Override / Create Virtual Robot</div>
  <div>
    Robot ID: <input id="ov-id" type="text" style="width:80px">
    X mm: <input id="ov-x" type="number" value="0">
    Y mm: <input id="ov-y" type="number" value="0">
    Hdg: <input id="ov-hdg" type="number" value="0">
    <button onclick="sendOverride()">Set</button>
  </div>
  <div class="section-title">Ball Control</div>
  <div>
    X mm: <input id="ball-x" type="number" value="0">
    Y mm: <input id="ball-y" type="number" value="0">
    Vel X mm/s: <input id="ball-vx" type="number" value="0">
    Vel Y mm/s: <input id="ball-vy" type="number" value="0">
    <button onclick="sendBallState()">Set</button>
  </div>
  <div class="section-title">Virtual Robot Controls</div>
  <div id="joysticks"></div>
</div>
<script>
const C = """ + config_json + r""";

const canvas = document.getElementById('field');
const ctx = canvas.getContext('2d');

let scale = 1;
let gd = 0;

function computeLayout() {
  scale = window.innerWidth / (C.field_length_mm + 2 * C.goal_depth_mm + 2);
  gd = Math.round(scale * C.goal_depth_mm);
  canvas.width  = window.innerWidth;
  canvas.height = Math.round(scale * C.field_width_mm) + 2;
}

const s = mm => mm * scale;

function fieldToCanvas(x, y) {
  return [
    (x + C.field_length_mm / 2) * scale + 1 + gd,
    (C.field_width_mm / 2 - y) * scale + 1,
  ];
}

let robots = {};
let ball = {world_x_mm: 0, world_y_mm: 0, vel_x_mmps: 0, vel_y_mmps: 0};

const ws = new WebSocket(`ws://${location.host}/ws`);

const joystickState = {};

function makeJoystick(robotId) {
  const size = 120;
  const knobR = 18;
  const wrap = document.createElement('div');
  wrap.className = 'joystick-wrap';
  wrap.dataset.robotId = robotId;

  const label = document.createElement('div');
  label.className = 'joystick-label';
  label.textContent = robotId;
  wrap.appendChild(label);

  const jc = document.createElement('canvas');
  jc.className = 'joystick-canvas';
  jc.width = size;
  jc.height = size;
  wrap.appendChild(jc);

  const velLabel = document.createElement('div');
  velLabel.className = 'joystick-label';
  velLabel.textContent = 'L: 0  R: 0';
  wrap.appendChild(velLabel);

  const state = { dx: 0, dy: 0, active: false };
  joystickState[robotId] = { state, velLabel, jc, size, knobR };

  function drawJoystick() {
    const jctx = jc.getContext('2d');
    const cx = size / 2, cy = size / 2;
    const maxR = size / 2 - knobR - 4;
    jctx.clearRect(0, 0, size, size);
    jctx.beginPath();
    jctx.arc(cx, cy, size / 2 - 2, 0, 2 * Math.PI);
    jctx.fillStyle = '#2a2a2a';
    jctx.fill();
    jctx.strokeStyle = '#555';
    jctx.lineWidth = 1;
    jctx.stroke();
    jctx.beginPath();
    jctx.moveTo(cx, cy - maxR);
    jctx.lineTo(cx, cy + maxR);
    jctx.moveTo(cx - maxR, cy);
    jctx.lineTo(cx + maxR, cy);
    jctx.strokeStyle = '#444';
    jctx.lineWidth = 1;
    jctx.stroke();
    const kx = cx - state.dx * maxR;
    const ky = cy - state.dy * maxR;
    jctx.beginPath();
    jctx.arc(kx, ky, knobR, 0, 2 * Math.PI);
    jctx.fillStyle = '#888';
    jctx.fill();
    jctx.strokeStyle = '#bbb';
    jctx.lineWidth = 1.5;
    jctx.stroke();
  }

  function getOffsetInCanvas(e) {
    const rect = jc.getBoundingClientRect();
    const cx = e.clientX ?? e.touches[0].clientX;
    const cy = e.clientY ?? e.touches[0].clientY;
    return [cx - rect.left, cy - rect.top];
  }

  function updateFromOffset(ox, oy) {
    const cx = size / 2, cy = size / 2;
    let dx = -(ox - cx) / (size / 2 - knobR - 4);
    let dy = -(oy - cy) / (size / 2 - knobR - 4);
    const mag = Math.sqrt(dx * dx + dy * dy);
    if (mag > 1) { dx /= mag; dy /= mag; }
    state.dx = dx;
    state.dy = dy;
    sendVelocity(robotId, dx, dy, velLabel);
    drawJoystick();
  }

  jc.addEventListener('mousedown', e => {
    state.active = true;
    const [ox, oy] = getOffsetInCanvas(e);
    updateFromOffset(ox, oy);
  });
  jc.addEventListener('touchstart', e => {
    e.preventDefault();
    state.active = true;
    const [ox, oy] = getOffsetInCanvas(e);
    updateFromOffset(ox, oy);
  }, { passive: false });
  window.addEventListener('mousemove', e => {
    if (!state.active) return;
    const [ox, oy] = getOffsetInCanvas(e);
    updateFromOffset(ox, oy);
  });
  window.addEventListener('touchmove', e => {
    if (!state.active) return;
    e.preventDefault();
    const [ox, oy] = getOffsetInCanvas(e);
    updateFromOffset(ox, oy);
  }, { passive: false });
  function release() {
    if (!state.active) return;
    state.active = false;
    state.dx = 0;
    state.dy = 0;
    sendVelocity(robotId, 0, 0, velLabel);
    drawJoystick();
  }
  window.addEventListener('mouseup', release);
  window.addEventListener('touchend', release);

  drawJoystick();
  return wrap;
}

function sendVelocity(robotId, dx, dy, velLabel) {
  const maxV = C.max_wheel_speed_mmps;
  const wb   = C.wheel_base_mm;
  const vl = (dy - dx) * maxV;
  const vr = (dy + dx) * maxV;
  velLabel.textContent = `L: ${vl.toFixed(1)}  R: ${vr.toFixed(1)}`;
  if (ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({
      type: 'cmd_vel',
      robot_id: robotId,
      cmd_vel_left:  vl,
      cmd_vel_right: vr,
    }));
  }
}

function rebuildJoysticks() {
  const container = document.getElementById('joysticks');
  const virtualIds = Object.keys(robots).filter(id => robots[id].virtual);
  const existing = new Set([...container.querySelectorAll('.joystick-wrap')].map(el => el.dataset.robotId));
  for (const id of virtualIds) {
    if (!existing.has(id)) {
      container.appendChild(makeJoystick(id));
    }
  }
  for (const el of container.querySelectorAll('.joystick-wrap')) {
    if (!virtualIds.includes(el.dataset.robotId)) {
      el.remove();
      delete joystickState[el.dataset.robotId];
    }
  }
}

function drawField() {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.save();

  const fw   = s(C.field_length_mm);
  const fh   = s(C.field_width_mm);
  const r    = s(C.corner_radius_mm);
  const gw   = s(C.goal_width_mm);
  const ox   = 1 + gd;
  const oy   = 1;
  const gTop = oy + (fh - gw) / 2;

  for (const line of C.tape_lines) {
    const [lx] = fieldToCanvas(line.x_mm, 0);
    ctx.strokeStyle = line.color;
    ctx.lineWidth = Math.max(1, Math.round(s(C.tape_width_mm)));
    ctx.beginPath();
    ctx.moveTo(lx, oy);
    ctx.lineTo(lx, oy + fh);
    ctx.stroke();
  }

  ctx.strokeStyle = '#ffffff';
  ctx.lineWidth = 2;
  ctx.beginPath();
  ctx.moveTo(ox + r, oy);
  ctx.lineTo(ox + fw - r, oy);
  ctx.arcTo(ox + fw, oy, ox + fw, oy + r, r);
  ctx.lineTo(ox + fw, gTop);
  ctx.moveTo(ox + fw, gTop + gw);
  ctx.lineTo(ox + fw, oy + fh - r);
  ctx.arcTo(ox + fw, oy + fh, ox + fw - r, oy + fh, r);
  ctx.lineTo(ox + r, oy + fh);
  ctx.arcTo(ox, oy + fh, ox, oy + fh - r, r);
  ctx.lineTo(ox, gTop + gw);
  ctx.moveTo(ox, gTop);
  ctx.lineTo(ox, oy + r);
  ctx.arcTo(ox, oy, ox + r, oy, r);
  ctx.stroke();

  for (const side of [-1, 1]) {
    const wallX   = side === 1 ? ox + fw : ox;
    const wallDir = side === 1 ? 1 : -1;
    ctx.fillStyle = '#2d6a2d';
    ctx.fillRect(wallDir === 1 ? wallX : wallX - gd, gTop, gd, gw);
    ctx.strokeStyle = '#aaaaaa';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(wallX, gTop);
    ctx.lineTo(wallX + wallDir * gd, gTop);
    ctx.lineTo(wallX + wallDir * gd, gTop + gw);
    ctx.lineTo(wallX, gTop + gw);
    ctx.stroke();
  }

  ctx.restore();
}

function drawBall() {
  const [cx, cy] = fieldToCanvas(ball.world_x_mm, ball.world_y_mm);
  const r = s(C.ball_diameter_mm / 2);
  ctx.save();
  ctx.beginPath();
  ctx.arc(cx, cy, r, 0, 2 * Math.PI);
  ctx.fillStyle = '#f0f0f0';
  ctx.fill();
  ctx.strokeStyle = '#333333';
  ctx.lineWidth = 1;
  ctx.stroke();
  ctx.restore();
}

function drawRobots() {
  const len2 = s(C.robot_length_mm / 2);
  const wid2 = s(C.robot_width_mm / 2);
  const r = s(C.robot_corner_radius_mm);
  for (const id in robots) {
    const rob = robots[id];
    if (rob.world_x_mm == null || rob.world_y_mm == null) continue;
    const hdg = rob.world_heading_deg ?? 0;
    const [cx, cy] = fieldToCanvas(rob.world_x_mm, rob.world_y_mm);
    const rad = hdg * Math.PI / 180;

    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(-rad);
    ctx.beginPath();
    ctx.moveTo(-len2 + r, -wid2);
    ctx.lineTo(len2 - r, -wid2);
    ctx.arcTo(len2, -wid2, len2, -wid2 + r, r);
    ctx.lineTo(len2, wid2 - r);
    ctx.arcTo(len2, wid2, len2 - r, wid2, r);
    ctx.lineTo(-len2 + r, wid2);
    ctx.arcTo(-len2, wid2, -len2, wid2 - r, r);
    ctx.lineTo(-len2, -wid2 + r);
    ctx.arcTo(-len2, -wid2, -len2 + r, -wid2, r);
    ctx.closePath();
    ctx.fillStyle = id.toLowerCase().includes('red')  ? '#cc3333' :
                    id.toLowerCase().includes('blue') ? '#3366cc' : '#888888';
    ctx.fill();
    ctx.strokeStyle = rob.virtual ? '#ffff00' : '#ffffff';
    ctx.lineWidth = rob.virtual ? 2 : 1;
    ctx.stroke();
    ctx.beginPath();
    ctx.moveTo(len2, 0);
    ctx.lineTo(len2 * 0.6, -wid2 * 0.5);
    ctx.lineTo(len2 * 0.6,  wid2 * 0.5);
    ctx.closePath();
    ctx.fillStyle = '#ffffff';
    ctx.fill();
    ctx.restore();

    ctx.fillStyle = '#ffffff';
    ctx.font = `${Math.max(9, Math.round(s(40)))}px monospace`;
    ctx.fillText(id, cx, cy);
  }
}

function render() {
  computeLayout();
  drawField();
  drawBall();
  drawRobots();
}

function updateBallInfo() {
  document.getElementById('ball-info').textContent =
    `x: ${ball.world_x_mm.toFixed(1)} mm   y: ${ball.world_y_mm.toFixed(1)} mm` +
    `   vx: ${(ball.vel_x_mmps ?? 0).toFixed(1)} mm/s   vy: ${(ball.vel_y_mmps ?? 0).toFixed(1)} mm/s`;
}

function updateTable() {
  const tbody = document.getElementById('robot-tbody');
  tbody.innerHTML = '';
  for (const id in robots) {
    const rob = robots[id];
    const tr = document.createElement('tr');
    for (const f of [
      id,
      rob.world_x_mm?.toFixed(0)        ?? '-',
      rob.world_y_mm?.toFixed(0)        ?? '-',
      rob.world_heading_deg?.toFixed(1) ?? '-',
      rob.distance_cm?.toFixed(1)       ?? '-',
      rob.reflectance_left?.toFixed(3)  ?? '-',
      rob.reflectance_right?.toFixed(3) ?? '-',
      rob.left_encoder?.toFixed(0)      ?? '-',
      rob.right_encoder?.toFixed(0)     ?? '-',
      rob.virtual ? 'yes' : 'no',
    ]) {
      const td = document.createElement('td');
      td.textContent = f;
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
}

function sendOverride() {
  fetch('/pose_override', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      robot_id:          document.getElementById('ov-id').value,
      world_x_mm:        parseFloat(document.getElementById('ov-x').value),
      world_y_mm:        parseFloat(document.getElementById('ov-y').value),
      world_heading_deg: parseFloat(document.getElementById('ov-hdg').value),
    }),
  });
}

function sendBallState() {
  fetch('/ball', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      world_x_mm: parseFloat(document.getElementById('ball-x').value),
      world_y_mm: parseFloat(document.getElementById('ball-y').value),
      vel_x_mmps: parseFloat(document.getElementById('ball-vx').value),
      vel_y_mmps: parseFloat(document.getElementById('ball-vy').value),
    }),
  });
}

window.addEventListener('resize', render);

ws.onmessage = (event) => {
  const msg = JSON.parse(event.data);
  if (msg.type === 'init') {
    robots = msg.robots;
    ball   = msg.ball ?? ball;
    rebuildJoysticks();
  } else if (msg.type === 'telemetry') {
    const d = msg.data;
    robots[d.robot_id] = {...(robots[d.robot_id] ?? {}), ...d};
  } else if (msg.type === 'ball') {
    ball = msg.data;
  } else if (msg.type === 'pose_override') {
    const d = msg.data;
    if (!robots[d.robot_id]) robots[d.robot_id] = {robot_id: d.robot_id, virtual: true};
    robots[d.robot_id].world_x_mm        = d.world_x_mm;
    robots[d.robot_id].world_y_mm        = d.world_y_mm;
    robots[d.robot_id].world_heading_deg = d.world_heading_deg;
    rebuildJoysticks();
  } else if (msg.type === 'virtual_robots') {
    for (const [id, data] of Object.entries(msg.data)) {
      robots[id] = {...(robots[id] ?? {}), ...data};
    }
  }
  render();
  updateBallInfo();
  updateTable();
};

render();
</script>
</body>
</html>"""


@app.get("/", response_class=HTMLResponse)
async def index():
    return build_html(FIELD_CONFIG)


def main():
    parser = argparse.ArgumentParser(description="XRP Soccer Field Telemetry Visualizer")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", "-p", type=int, default=8080)
    args = parser.parse_args()
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
