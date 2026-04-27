#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "uvicorn[standard]",
#   "pydantic",
# ]
# ///

import argparse
import asyncio
import json
import math
import os
import random
import time
from contextlib import asynccontextmanager

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from util import (BALL_RADIUS_MM, CORNER_RADIUS_MM, FIELD_CONFIG,
                  FIELD_LENGTH_MM, FIELD_WIDTH_MM, GOAL_DEPTH_MM,
                  GOAL_WIDTH_MM, MAX_WHEEL_SPEED_MMPS, MM_PER_TICK,
                  ROBOT_CORNER_RADIUS_MM, ROBOT_DISTANCE_SENSOR_OFFSET,
                  ROBOT_LENGTH_MM, ROBOT_REFLECTANCE_SENSOR_OFFSET,
                  ROBOT_REFLECTANCE_SENSOR_SIDE, ROBOT_WIDTH_MM, TAPE_LINES,
                  TAPE_WIDTH_MM, WHEEL_BASE_MM, clamp_speed,
                  closest_point_on_rounded_rect, point_in_field, ray_to_ball,
                  ray_to_field_boundary, ray_to_robot, robot_corners)

FRICTION_PER_SEC = 60.0
RESTITUTION = 0.7
SIM_HZ = 60
EPISODE_RESTART_DELAY_SEC = 1.0
EPISODE_MAXIMUM_TIME = 300.0
BROADCAST_HZ = 60

ball_state: dict = {
    'world_x_mm': 0.0,
    'world_y_mm': 0.0,
    'vel_x_mmps': 0.0,
    'vel_y_mmps': 0.0,
}
robots: dict[str, dict] = {}
websocket_clients: list[WebSocket] = []
robot_rewards: dict[str, dict] = {}
reward_memory: dict[str, dict] = {}
sim_state = {'training': False, 'episode_finished': False, 'restart': None,
             'run_number': 0, 'run_start_time': None, 'sim_time': 0.0,
             'sim_start': 0.0, 'fast': False}
pending_broadcasts: dict[tuple, dict] = {}


def reset_episode():
    ball_state.update({'world_x_mm': 0.0, 'world_y_mm': 0.0, 'vel_x_mmps': 0.0, 'vel_y_mmps': 0.0})
    robot_rewards.clear()
    reward_memory.clear()
    sim_state['episode_finished'] = False
    sim_state['restart'] = None
    sim_state['run_number'] += 1
    sim_state['run_start_time'] = time.time()
    sim_state['sim_start'] = sim_state['sim_time']
    for team in ('red', 'blue'):
        robot_ids = sorted(rid for rid, robot in robots.items() if robot.get('team', 'red') == team)
        base_x = (-1 if team == 'red' else 1) * FIELD_LENGTH_MM * 3 / 8
        base_heading = 0 if team == 'red' else 180
        offset = 0 if random.random() >= 0.5 else 1
        for index, robot in enumerate([robots[id] for id in robot_ids]):
            heading = (base_heading + random.gauss(0, 5)) % 360
            robot.update({
                'world_x_mm': random.gauss(base_x, 100),
                'world_y_mm': random.gauss(
                    (1 if (index + offset) % 2 == 0 else -1) * FIELD_WIDTH_MM / 4, 100),
                'world_heading_deg': heading,
                'left_encoder': 0.0,
                'right_encoder': 0.0,
                'cmd_vel_left': 0.0,
                'cmd_vel_right': 0.0,
                'distance_cm': 65535.0,
                'training': sim_state['training'],
                'reset': True})
            robot.pop('imu_state', None)
            generate_imu_reading(robot)


class RobotPose:
    """
    Dead-reckoning pose tracker using differential drive kinematics.
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


def generate_distance_reading(robot, robots_dict, ball):
    rx = robot.get('world_x_mm')
    ry = robot.get('world_y_mm')
    if rx is None or ry is None:
        return 65535
    heading_rad = math.radians(robot.get('world_heading_deg', 0.0))
    dx = math.cos(heading_rad)
    dy = math.sin(heading_rad)
    rx += dx * ROBOT_DISTANCE_SENSOR_OFFSET
    ry += dy * ROBOT_DISTANCE_SENSOR_OFFSET
    min_distance = None
    t = ray_to_field_boundary(rx, ry, dx, dy)
    if t is not None and t > 0:
        min_distance = t if min_distance is None else min(min_distance, t)
    t = ray_to_ball(rx, ry, dx, dy, ball)
    if t is not None and t > 0:
        min_distance = t if min_distance is None else min(min_distance, t)
    robot_id = robot.get('robot_id')
    for other_id, other in robots_dict.items():
        if other_id == robot_id:
            continue
        t = ray_to_robot(rx, ry, dx, dy, other)
        if t is not None and t > 0:
            min_distance = t if min_distance is None else min(min_distance, t)
    if min_distance is None or min_distance < 0:
        return 65535
    distance_mm = min_distance
    if distance_mm < 20:
        distance_mm = 20
    noise_std = 0.5 + distance_mm * 0.01
    distance_mm += random.gauss(0, noise_std)
    far_noise = 800
    if random.random() < 0.15 * (distance_mm - far_noise) / (1000 - far_noise):
        return 65535
    if distance_mm > 1000:
        return 65535
    distance_cm = distance_mm / 10.0
    return max(0, min(65535, distance_cm))


def generate_reflectance_readings(robot):
    rx = robot.get('world_x_mm')
    ry = robot.get('world_y_mm')
    if rx is None or ry is None:
        return 65535
    heading_rad = math.radians(robot.get('world_heading_deg', 0.0))
    dx = math.cos(heading_rad)
    dy = math.sin(heading_rad)
    rx += dx * ROBOT_REFLECTANCE_SENSOR_OFFSET
    ry += dy * ROBOT_REFLECTANCE_SENSOR_OFFSET
    for side, mul in [('left', -1), ('right', 1)]:
        sx = rx + dy * ROBOT_REFLECTANCE_SENSOR_SIDE * mul
        # sy = ry + dx * ROBOT_REFLECTANCE_SENSOR_SIDE
        rel = 0.9
        for t in TAPE_LINES:
            if abs(sx - t['x_mm']) < TAPE_WIDTH_MM / 2:
                rel = t['rel']
        robot[f'reflectance_{side}'] = min(1, random.gauss(rel, 0.03))


def generate_imu_reading(robot):
    if 'imu_state' not in robot:
        robot['imu_state'] = {'bias': random.uniform(0, 360), 'drift': 0.0}
    robot['imu_state']['drift'] += random.gauss(0, 0.02)
    robot['imu_state']['drift'] = max(-0.3, min(0.3, robot['imu_state']['drift']))
    imu = (robot['world_heading_deg'] + robot['imu_state']
           ['bias'] + robot['imu_state']['drift']) % 360
    imu += random.gauss(0, 0.2)
    robot['imu_heading_deg'] = imu % 360


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
    check = True
    while check:
        check = False
        for i in range(len(robot_list)):
            for j in range(i + 1, len(robot_list)):
                r1 = robot_list[i]
                r2 = robot_list[j]
                if robots_overlap(r1, r2):
                    check = True
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
                    push = 0.5
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


def team_goal_direction(team):
    return 1.0 if team == 'red' else -1.0


def scored_goal_team():
    half_len = FIELD_LENGTH_MM / 2
    goal_half = GOAL_WIDTH_MM / 2 - BALL_RADIUS_MM
    bx = ball_state['world_x_mm']
    by = ball_state['world_y_mm']
    if abs(by) > goal_half:
        return None
    if bx > half_len:
        return 'red'
    if bx < -half_len:
        return 'blue'
    return None


def update_rewards(dt):
    scored_team = scored_goal_team()
    over_time = sim_state['run_start_time'] and (
        sim_state['sim_time'] - sim_state['sim_start'] > EPISODE_MAXIMUM_TIME)
    if not sim_state['episode_finished'] and (over_time or (
            scored_team is not None and scored_team != reward_memory.get('scored_team'))):
        sim_state['episode_finished'] = True
        sim_state['restart'] = sim_state['sim_time'] + EPISODE_RESTART_DELAY_SEC
        for robot in robots.values():
            robot['cmd_vel_left'] = robot['cmd_vel_right'] = 0.0
    already_scored = reward_memory.get('scored_team')
    new_goal = scored_team is not None and scored_team != already_scored
    if scored_team is None:
        reward_memory['scored_team'] = None
    elif new_goal:
        reward_memory['scored_team'] = scored_team

    bx = ball_state['world_x_mm']
    by = ball_state['world_y_mm']
    vx = ball_state['vel_x_mmps']

    for rid, robot in robots.items():
        rx = robot.get('world_x_mm')
        ry = robot.get('world_y_mm')
        if rx is None or ry is None:
            continue
        team = robot.get('team', 'red')
        direction = team_goal_direction(team)
        dist_to_ball = math.sqrt((bx - rx) ** 2 + (by - ry) ** 2)
        prev = reward_memory.get(rid)
        if prev is None:
            reward_memory[rid] = {
                'ball_x': bx,
                'ball_y': by,
                'dist_to_ball': dist_to_ball,
            }
            robot_rewards.setdefault(rid, {'reward': 0.0, 'terminal': False})
            continue

        ball_progress = direction * (bx - prev['ball_x'])
        approach = prev['dist_to_ball'] - dist_to_ball
        reward = -0.002 * dt
        reward += 0.0025 * ball_progress
        reward += 0.0010 * approach
        reward += 0.0005 * direction * vx * dt
        if dist_to_ball < 170:
            reward += 0.01 * dt

        terminal = False
        if new_goal:
            terminal = True
            reward += 100.0 if scored_team == team else -100.0

        entry = robot_rewards.setdefault(rid, {'reward': 0.0, 'terminal': False})
        entry['reward'] += reward
        entry['terminal'] = entry['terminal'] or terminal
        prev['ball_x'] = bx
        prev['ball_y'] = by
        prev['dist_to_ball'] = dist_to_ball


def broadcast_key(message: dict) -> tuple:
    msg_type = message.get('type', '')
    data = message.get('data')
    robot_id = data.get('robot_id') if isinstance(data, dict) else None
    return (msg_type, robot_id)


def enqueue_broadcast(message: dict):
    pending_broadcasts[broadcast_key(message)] = message


async def flush_broadcasts():
    if not pending_broadcasts or not websocket_clients:
        pending_broadcasts.clear()
        return
    messages = list(pending_broadcasts.values())
    pending_broadcasts.clear()
    disconnected = []
    for ws in websocket_clients:
        try:
            for msg in messages:
                await asyncio.wait_for(ws.send_json(msg), timeout=0.5)
        except Exception:
            disconnected.append(ws)
    for ws in set(disconnected):
        try:
            websocket_clients.remove(ws)
        except ValueError:
            pass


async def broadcast_loop():
    while True:
        await asyncio.sleep(1.0 / BROADCAST_HZ)
        await flush_broadcasts()


async def simulation_loop():  # noqa
    dt = 1.0 / SIM_HZ
    next_time = time.time()
    last_training_broadcast = 0.0
    while True:
        wait = 0 if sim_state['fast'] else max(0.001, next_time - time.time())
        next_time += dt
        sim_state['sim_time'] += dt
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            await asyncio.sleep(0)
        if sim_state['training'] and sim_state['restart'] is not None:
            if sim_state['sim_time'] >= sim_state['restart']:
                reset_episode()
                enqueue_broadcast({
                    'type': 'training_state',
                    'data': {
                        'training': sim_state['training'],
                        'run_number': sim_state['run_number'],
                        'run_start_time': sim_state['run_start_time'],
                        'sim_start': sim_state['sim_start'],
                        'sim_time': sim_state['sim_time'],
                    }
                })
                for robot in robots.values():
                    if robot.get('virtual'):
                        robot['training'] = True
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
        update_rewards(dt)
        for robot in robots.values():
            if robot.get('virtual'):
                robot['training'] = sim_state['training']
                robot['distance_cm'] = generate_distance_reading(robot, robots, ball_state)
                generate_reflectance_readings(robot)
                generate_imu_reading(robot)
        virtual_updates = {
            rid: dict(rob) for rid, rob in robots.items() if rob.get('virtual')
        }
        if virtual_updates:
            enqueue_broadcast({'type': 'virtual_robots', 'data': virtual_updates})
        enqueue_broadcast({'type': 'ball', 'data': dict(ball_state)})
        now = time.time()
        if sim_state['training'] and now - last_training_broadcast >= 10.0:
            last_training_broadcast = now
            enqueue_broadcast({
                'type': 'training_state',
                'data': {
                    'training': sim_state['training'],
                    'run_number': sim_state['run_number'],
                    'run_start_time': sim_state['run_start_time'],
                    'sim_start': sim_state['sim_start'],
                    'sim_time': sim_state['sim_time'],
                }
            })


@asynccontextmanager
async def lifespan(app: FastAPI):
    sim_task = asyncio.create_task(simulation_loop())
    bcast_task = asyncio.create_task(broadcast_loop())
    yield
    sim_task.cancel()
    bcast_task.cancel()
    for task in (sim_task, bcast_task):
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
    imu_heading_deg: float | None = None
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


class EstimatedPose(BaseModel):
    robot_id: str
    x_mm: float
    y_mm: float
    heading_deg: float
    std_x_mm: float
    std_y_mm: float
    std_heading_deg: float


class RobotCommand(BaseModel):
    robot_id: str
    straight: float
    turn: float


class TeamOverride(BaseModel):
    robot_id: str
    team: str


@app.get('/reward')
async def get_reward(robot_id: str):
    entry = robot_rewards.setdefault(robot_id, {'reward': 0.0, 'terminal': False})
    reward = entry['reward']
    terminal = entry['terminal']
    entry['reward'] = 0.0
    entry['terminal'] = False
    return {'robot_id': robot_id, 'reward': reward, 'terminal': terminal}


@app.post('/telemetry')
async def receive_telemetry(telemetry: Telemetry):
    data = telemetry.model_dump()
    if telemetry.robot_id not in robots:
        robots[telemetry.robot_id] = {}
    robots[telemetry.robot_id].update(data)
    enqueue_broadcast({'type': 'telemetry', 'data': data})
    return {'status': 'ok'}


@app.post('/ball')
async def update_ball(state: BallState):
    ball_state.update(state.model_dump())
    enqueue_broadcast({'type': 'ball', 'data': dict(ball_state)})
    return {'status': 'ok'}


@app.post('/pose_override')
async def override_pose(override: PoseOverride):
    robot_id = override.robot_id
    if robot_id not in robots:
        robots[robot_id] = {
            'robot_id': robot_id,
            'virtual': True,
            'team': 'red',
            'cmd_vel_left': 0.0,
            'cmd_vel_right': 0.0,
            'left_encoder': 0.0,
            'right_encoder': 0.0,
            'distance_cm': 0.0,
            'reflectance_left': 1.0,
            'reflectance_right': 1.0,
            'training': sim_state['training'],
            'reset': False,
        }
    robots[robot_id]['world_x_mm'] = override.world_x_mm
    robots[robot_id]['world_y_mm'] = override.world_y_mm
    robots[robot_id]['world_heading_deg'] = override.world_heading_deg
    robots[robot_id].pop('imu_state', None)
    generate_imu_reading(robots[robot_id])
    constrain_robot_to_field(robots[robot_id])
    resolve_robot_overlaps()
    enqueue_broadcast({'type': 'pose_override', 'data': override.model_dump()})
    return {'status': 'ok'}


@app.post('/estimated_pose')
async def receive_estimated_pose(est: EstimatedPose):
    rid = est.robot_id
    if rid not in robots:
        return {'status': 'error', 'message': 'unknown robot'}
    data = est.model_dump()
    robots[rid]['estimated_pose'] = data
    enqueue_broadcast({'type': 'estimated_pose', 'data': data})
    return {'status': 'ok'}


@app.post('/arcade')
async def arcade(cmd: RobotCommand):
    robot_id = cmd.robot_id
    if robot_id in robots and robots[robot_id].get('virtual') and not sim_state['episode_finished']:
        if time.time() - robots[robot_id].get('cmd_last', 0) > 1000:
            vl = (cmd.straight - cmd.turn) * MAX_WHEEL_SPEED_MMPS
            vr = (cmd.straight + cmd.turn) * MAX_WHEEL_SPEED_MMPS
            robots[robot_id]['cmd_vel_left'] = clamp_speed(vl)
            robots[robot_id]['cmd_vel_right'] = clamp_speed(vr)
    return {'status': 'ok'}


@app.post('/team')
async def set_team(team_override: TeamOverride):
    robot_id = team_override.robot_id
    if robot_id in robots:
        robots[robot_id]['team'] = team_override.team
    else:
        return {'status': 'error', 'message': 'robot not found'}
    return {'status': 'ok'}


@app.get('/robots')
async def get_robots():
    return robots


@app.get('/robot')
async def get_robot(robot_id: str):
    robot = robots[robot_id]
    data = dict(robot)
    data['reset'] = bool(robot.pop('reset', False))
    data['sim_start'] = sim_state['sim_start']
    data['sim_time'] = sim_state['sim_time']
    return data


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    websocket_clients.append(ws)
    try:
        await ws.send_json({'type': 'init', 'robots': robots, 'ball': ball_state})
        await ws.send_json({
            'type': 'training_state',
            'data': {
                'training': sim_state['training'],
                'run_number': sim_state['run_number'],
                'run_start_time': sim_state['run_start_time'],
                'sim_start': sim_state['sim_start'],
                'sim_time': sim_state['sim_time'],
            }
        })
        while True:
            raw = await ws.receive_text()
            try:
                msg = json.loads(raw)
            except Exception as e:
                print(e)
                continue
            if msg.get('type') == 'arcade':
                robot_id = msg.get('robot_id')
                if (robot_id in robots and robots[robot_id].get('virtual') and
                        not sim_state['episode_finished']):
                    vl = (msg.get('straight', 0) - msg.get('turn', 0)) * MAX_WHEEL_SPEED_MMPS
                    vr = (msg.get('straight', 0) + msg.get('turn', 0)) * MAX_WHEEL_SPEED_MMPS
                    robots[robot_id]['cmd_vel_left'] = clamp_speed(vl)
                    robots[robot_id]['cmd_vel_right'] = clamp_speed(vr)
                    if vl or vr:
                        robots[robot_id]['cmd_last'] = time.time()
                    else:
                        robots[robot_id]['cmd_last'] = 0
            if msg.get('type') == 'train':
                active = bool(msg.get('active'))
                if active and not sim_state['training']:
                    sim_state['training'] = True
                    reset_episode()
                elif not active:
                    sim_state['training'] = False
                    for robot in robots.values():
                        robot['training'] = False
                        robot['cmd_vel_left'] = robot['cmd_vel_right'] = 0.0
    except WebSocketDisconnect:
        pass
    finally:
        if ws in websocket_clients:
            websocket_clients.remove(ws)


def build_html(config: dict) -> str:
    html = open(os.path.join(os.path.dirname(__file__), 'render.html')).read()
    html = html.replace('MAINSCRIPT', open(
        os.path.join(os.path.dirname(__file__), 'render.js')).read())
    return html.replace('CONFIG_JSON', json.dumps(config))


@app.get('/', response_class=HTMLResponse)
async def index():
    return build_html(FIELD_CONFIG)


def main():
    parser = argparse.ArgumentParser(description='XRP Soccer Field Telemetry Visualizer')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', '-p', type=int, default=8080)
    parser.add_argument('--fast', action='store_true', default=False)
    args = parser.parse_args()
    sim_state['fast'] = args.fast
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
