#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11"
# dependencies = [
#   "fastapi",
#   "pydantic",
#   "uvicorn",
#   "websockets",
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
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

from util import (BALL_RADIUS_MM, CORNER_MEET, CORNER_RADIUS_MM,
                  FIELD_BOUNDARY_CORNERS, FIELD_BOUNDARY_SEGMENTS,
                  FIELD_CONFIG, FIELD_LENGTH_MM, FIELD_WIDTH_MM, GOAL_WIDTH_MM,
                  INERTIA_MMPS_PER_TICK, MAX_WHEEL_SPEED_MMPS, MM_PER_TICK,
                  ROBOT_CORNER_RADIUS_MM, ROBOT_DISTANCE_SENSOR_OFFSET,
                  ROBOT_DISTANCE_SENSOR_WIDTH_DEG, ROBOT_LENGTH_MM,
                  ROBOT_REFLECTANCE_SENSOR_OFFSET,
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
DISTANCE_SENSOR_CHECKS = 7

ball_state: dict = {
    'world_x_mm': 0.0,
    'world_y_mm': 0.0,
    'vel_x_mmps': 0.0,
    'vel_y_mmps': 0.0,
}
robots: dict[str, dict] = {}
ui_websocket_clients: list[WebSocket] = []
robot_websocket_clients: dict[str, WebSocket] = {}
robot_rewards: dict[str, dict] = {}
reward_memory: dict[str, dict] = {}
sim_state = {'training': False, 'episode_finished': False, 'restart': None,
             'run_number': 0, 'run_start_time': None, 'sim_time': 0.0,
             'run_record': [],
             'synced_robots': set(), 'sim_start': 0.0, 'fast': False}
pending_broadcasts: dict[tuple, dict] = {}


def spawn_pose(team: str, pos: str) -> dict:
    return {
        'world_x_mm': -900 if team == 'red' else 900,
        'world_y_mm': 300 if pos == 'high' else -300,
        'world_heading_deg': 0 if team == 'red' else 180,
    }


def ensure_virtual_robot(robot_id: str, team: str = 'red', pos: str = 'high') -> dict:
    robot = robots.get(robot_id)
    if robot is None:
        robot = {
            'robot_id': robot_id,
            'virtual': True,
            'team': team,
            'cmd_vel_left': 0.0,
            'cmd_vel_right': 0.0,
            'last_vel_left': 0.0,
            'last_vel_right': 0.0,
            'left_encoder': 0.0,
            'right_encoder': 0.0,
            'distance_cm': 65535.0,
            'reflectance_left': 1.0,
            'reflectance_right': 1.0,
            'imu_heading_deg': 0.0,
            'training': sim_state['training'],
            'reset': False,
        }
        robot.update(spawn_pose(team, pos))
        robots[robot_id] = robot
        generate_imu_reading(robot)
        constrain_robot_to_field(robot)
        resolve_robot_overlaps()
    else:
        robot['robot_id'] = robot_id
        robot['virtual'] = True
        robot['team'] = team
        robot['training'] = sim_state['training']
        if robot.get('world_x_mm') is None or robot.get('world_y_mm') is None:
            robot.update(spawn_pose(team, pos))
            robot.pop('imu_state', None)
            generate_imu_reading(robot)
            constrain_robot_to_field(robot)
            resolve_robot_overlaps()
    return robot


def connected_virtual_ids() -> set[str]:
    return {
        robot_id for robot_id, ws in robot_websocket_clients.items()
        if ws and robot_id in robots and robots[robot_id].get('virtual')
    }


def apply_arcade(robot_id: str, straight: float, turn: float):
    if (robot_id not in robots or not robots[robot_id].get('virtual') or
            sim_state['episode_finished']):
        return
    vl = (straight - turn) * MAX_WHEEL_SPEED_MMPS
    vr = (straight + turn) * MAX_WHEEL_SPEED_MMPS
    robots[robot_id]['cmd_vel_left'] = clamp_speed(vl)
    robots[robot_id]['cmd_vel_right'] = clamp_speed(vr)
    robots[robot_id]['cmd_last'] = time.time() if (vl or vr) else 0.0


def update_estimated_pose(data: dict):
    robot_id = data.get('robot_id')
    if robot_id not in robots:
        return
    robots[robot_id]['estimated_pose'] = data
    queue_broadcast({'type': 'estimated_pose', 'data': data})


async def send_robot_state(robot_id: str, ws: WebSocket):
    robot = robots.get(robot_id)
    if not robot or not robot.get('virtual'):
        return
    reward = robot_rewards.setdefault(robot_id, {'reward': 0.0, 'terminal_id': 0})
    reset = bool(robot.get('reset', False))
    await asyncio.wait_for(ws.send_json({
        'type': 'robot_state',
        'data': {
            'robot_id': robot_id,
            'left_encoder': robot.get('left_encoder', 0.0),
            'right_encoder': robot.get('right_encoder', 0.0),
            'distance_cm': robot.get('distance_cm', 65535.0),
            'reflectance_left': robot.get('reflectance_left', 1.0),
            'reflectance_right': robot.get('reflectance_right', 1.0),
            'imu_heading_deg': robot.get('imu_heading_deg', 0.0),
            'training': sim_state['training'],
            'reset': reset,
            'last_result': sim_state['run_record'][-1] if len(sim_state['run_record']) else '',
            'sim_start': sim_state['sim_start'],
            'sim_time': sim_state['sim_time'],
            'reward_total': reward['reward'],
            'terminal_id': reward['terminal_id'],
        }
    }), timeout=0.5)
    if reset:
        robot['reset'] = False


async def send_robot_states():
    disconnected = []
    for robot_id, ws in list(robot_websocket_clients.items()):
        try:
            await send_robot_state(robot_id, ws)
        except Exception:
            disconnected.append((robot_id, ws))
    for robot_id, ws in disconnected:
        if robot_websocket_clients.get(robot_id) is ws:
            robot_websocket_clients.pop(robot_id, None)
            sim_state['synced_robots'].discard(robot_id)


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
        for index, robot in enumerate([robots[robot_id] for robot_id in robot_ids]):
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
                'last_vel_left': 0.0,
                'last_vel_right': 0.0,
                'distance_cm': 65535.0,
                'training': sim_state['training'],
                'reset': True})
            robot.pop('imu_state', None)
            generate_imu_reading(robot)


class RobotPose:
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
    distance_mm = 655350
    for check in range(DISTANCE_SENSOR_CHECKS):
        theta = heading_rad + ((check / (DISTANCE_SENSOR_CHECKS - 1)) - 0.5) * \
            2 * math.radians(ROBOT_DISTANCE_SENSOR_WIDTH_DEG)
        dx = math.cos(theta)
        dy = math.sin(theta)
        t = ray_to_field_boundary(rx, ry, dx, dy)
        distance_mm = min(distance_mm, t) if t is not None and t > 0 else distance_mm
        t = ray_to_ball(rx, ry, dx, dy, ball)
        distance_mm = min(distance_mm, t) if t is not None and t > 0 else distance_mm
        robot_id = robot.get('robot_id')
        for other_id, other in robots_dict.items():
            if other_id == robot_id:
                continue
            t = ray_to_robot(rx, ry, dx, dy, other)
            distance_mm = min(distance_mm, t) if t is not None and t > 0 else distance_mm
    if distance_mm >= 655350 or distance_mm < 0:
        return 65535
    if distance_mm < 20:
        distance_mm = 20
    noise_std = 0.5 + distance_mm * 0.01
    distance_mm += random.gauss(0, noise_std)
    if distance_mm > 1000:
        return 65535
    far_noise = 800
    if random.random() < 0.15 * (distance_mm - far_noise) / (1000 - far_noise):
        return 65535
    return max(0, min(65535, distance_mm / 10.0))


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
        rel = 0.9
        for tape in TAPE_LINES:
            if abs(sx - tape['x_mm']) < TAPE_WIDTH_MM / 2:
                rel = tape['rel']
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
    if all(point_in_field(cx, cy) for cx, cy in corners):
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
                if not robots_overlap(r1, r2):
                    continue
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


def field_boundary_response(bx, by, vx, vy, radius):
    for x1, y1, lx, ly in FIELD_BOUNDARY_SEGMENTS:
        length_sq = lx * lx + ly * ly
        length = math.sqrt(length_sq)
        nx = ly / length
        ny = -lx / length
        px = bx - x1
        py = by - y1
        t = (px * lx + py * ly) / length_sq
        if 0 <= t <= 1:
            d_line = px * nx + py * ny
            if d_line < radius:
                push = radius - d_line
                bx += nx * push
                by += ny * push
                dot = vx * nx + vy * ny
                if dot < 0:
                    vx -= (1 + RESTITUTION) * dot * nx
                    vy -= (1 + RESTITUTION) * dot * ny
        else:
            if t < 0:
                cx, cy = x1, y1
            else:
                cx, cy = x1 + lx, y1 + ly
            dx = bx - cx
            dy = by - cy
            dist = math.sqrt(dx * dx + dy * dy)
            if dist < radius and dist > 0:
                enx = dx / dist
                eny = dy / dist
                push = radius - dist
                bx += enx * push
                by += eny * push
                dot = vx * enx + vy * eny
                if dot < 0:
                    vx -= (1 + RESTITUTION) * dot * enx
                    vy -= (1 + RESTITUTION) * dot * eny
    boundary = CORNER_RADIUS_MM - radius
    for ccx, ccy in FIELD_BOUNDARY_CORNERS:
        dx = bx - ccx
        dy = by - ccy
        dist = math.sqrt(dx * dx + dy * dy)
        if dist == 0:
            continue
        px = ccx + (dx / dist) * CORNER_RADIUS_MM
        py = ccy + (dy / dist) * CORNER_RADIUS_MM
        if abs(px) < CORNER_MEET:
            continue
        if (px - ccx) * ccx >= -1e-5 and (py - ccy) * ccy >= -1e-5:
            if dist > boundary:
                nx = -dx / dist
                ny = -dy / dist
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
    vl = robot.get('last_vel_left', 0.0)
    vr = robot.get('last_vel_right', 0.0)
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
    vlt = robot.get('cmd_vel_left', 0.0)
    vrt = robot.get('cmd_vel_right', 0.0)
    vl = robot.get('last_vel_left', 0.0)
    vr = robot.get('last_vel_right', 0.0)
    if vl != vlt:
        vl += max(min(vlt - vl, INERTIA_MMPS_PER_TICK), -INERTIA_MMPS_PER_TICK)
        robot['last_vel_left'] = vl
    if vr != vrt:
        vr += max(min(vrt - vr, INERTIA_MMPS_PER_TICK), -INERTIA_MMPS_PER_TICK)
        robot['last_vel_right'] = vr
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
        sim_state['run_record'].append(' ' if scored_team is None else scored_team[:1])
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
    vy = ball_state['vel_x_mmps']
    for robot_id, robot in robots.items():
        rx = robot.get('world_x_mm')
        ry = robot.get('world_y_mm')
        if rx is None or ry is None:
            continue
        team = robot.get('team', 'red')
        direction = team_goal_direction(team)
        dist_to_ball = math.sqrt((bx - rx) ** 2 + (by - ry) ** 2)
        prev = reward_memory.get(robot_id)
        if prev is None:
            reward_memory[robot_id] = {
                'ball_x': bx,
                'ball_y': by,
                'dist_to_ball': dist_to_ball,
            }
            robot_rewards.setdefault(robot_id, {'reward': 0.0, 'terminal_id': 0})
            continue
        ball_progress = direction * (bx - prev['ball_x'])
        approach = prev['dist_to_ball'] - dist_to_ball
        reward = 0
        reward += -0.0005 * dt
        # reward += 0.0005 * ball_progress
        # reward += 0.00025 * approach
        # reward += 0.0001 * max(0, direction * vx) * dt
        # reward += 0.0001 * direction * vx * dt
        if (vx * direction > 0 and
                abs(by + vy / vx * (FIELD_LENGTH_MM / 2 - bx)) < GOAL_WIDTH_MM / 2):
            reward += 0.1 * max(0, direction * vx) * dt
        if 'rx' in prev:
            dx = rx - bx
            dy = ry - by
            rvx = (rx - prev['rx']) / dt
            rvy = (ry - prev['ry']) / dt
            ux = rvx - vx
            uy = rvy - vy
            ux, uy = rvx, rvy  # ##DWM::
            if (rvx * direction > 0 and (ux or uy) and
                    (bx - rx) * direction > 0 and abs(by + rvy / rvx * (FIELD_LENGTH_MM / 2 - bx)) < GOAL_WIDTH_MM / 2 and
                    # abs(ry + rvy / rvx * (FIELD_LENGTH_MM / 2 - rx)) < GOAL_WIDTH_MM / 2 and
                    abs(dx * uy - dy * ux) / (ux**2 + uy**2)**0.5 < ROBOT_WIDTH_MM / 2):
                reward += 0.1 * max(0, direction * rvx) * dt
        if abs(vx) > abs(prev.get('ball_vx', 0)) * 1.5 and vx * direction > 0:
            reward += 5
        # if dist_to_ball < 200:
        #     if (rx - bx) * direction < 0:
        #         reward += 0.0025 * dt
        #     else:
        #         reward -= 0.0005 * dt
        # else:
        #     reward += -0.0001 * dt
        terminal = False
        if new_goal:
            terminal = True
            elapsed = sim_state['sim_time'] - sim_state['sim_start']
            win_score = 100 + 50 * (1 - min(1, elapsed / EPISODE_MAXIMUM_TIME))
            reward += (1 if scored_team == team else -1) * win_score
        entry = robot_rewards.setdefault(robot_id, {'reward': 0.0, 'terminal_id': 0})
        entry['reward'] += reward
        if terminal:
            entry['terminal_id'] += 1
        prev['ball_x'] = bx
        prev['ball_y'] = by
        prev['dist_to_ball'] = dist_to_ball
        prev['ball_vx'] = vx
        prev['rx'] = rx
        prev['ry'] = ry


def broadcast_key(message: dict) -> tuple:
    msg_type = message.get('type', '')
    data = message.get('data')
    robot_id = data.get('robot_id') if isinstance(data, dict) else None
    return msg_type, robot_id


def queue_broadcast(message: dict):
    pending_broadcasts[broadcast_key(message)] = message


async def flush_broadcasts():
    if not pending_broadcasts or not ui_websocket_clients:
        pending_broadcasts.clear()
        return
    messages = list(pending_broadcasts.values())
    pending_broadcasts.clear()
    disconnected = []
    for ws in ui_websocket_clients:
        try:
            for msg in messages:
                await asyncio.wait_for(ws.send_json(msg), timeout=0.5)
        except Exception:
            disconnected.append(ws)
    for ws in set(disconnected):
        try:
            ui_websocket_clients.remove(ws)
        except ValueError:
            pass


async def broadcast_loop():
    while True:
        await asyncio.sleep(1.0 / BROADCAST_HZ)
        await flush_broadcasts()


def run_summary():
    recent = sim_state['run_record'][-100:]
    return {'red': len([r for r in recent if r == 'r']),
            'blue': len([b for b in recent if b == 'b'])}


async def simulation_loop():  # noqa
    dt = 1.0 / SIM_HZ
    next_time = time.time()
    last_training_broadcast = 0.0
    last_report = time.time()
    report_frames = 0
    while True:
        if sim_state['fast']:
            dur = time.time() - last_report
            if dur > 10:
                print(f'{report_frames / dur:3.1f} Hz ({report_frames / dur / SIM_HZ:4.2f}x)')
                report_frames = 0
                last_report = time.time()
            report_frames += 1
        synced = connected_virtual_ids()
        if synced and sim_state['fast']:
            while not synced.issubset(sim_state['synced_robots']):
                await asyncio.sleep(0.001)
        wait = 0 if sim_state['fast'] else max(0.001, next_time - time.time())
        next_time += dt
        sim_state['sim_time'] += dt
        sim_state['synced_robots'].clear()
        if wait > 0:
            await asyncio.sleep(wait)
        else:
            await asyncio.sleep(0)
        if (sim_state['training'] and sim_state['restart'] is not None and
                sim_state['sim_time'] >= sim_state['restart']):
            reset_episode()
            queue_broadcast({
                'type': 'training_state',
                'data': {
                    'training': sim_state['training'],
                    'run_number': sim_state['run_number'],
                    'run_record': run_summary(),
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
        if sim_state['training']:
            update_rewards(dt)
        for robot in robots.values():
            if robot.get('virtual'):
                robot['training'] = sim_state['training']
                robot['distance_cm'] = generate_distance_reading(robot, robots, ball_state)
                generate_reflectance_readings(robot)
                generate_imu_reading(robot)
        await send_robot_states()
        virtual_updates = {
            robot_id: dict(robot) for robot_id, robot in robots.items() if robot.get('virtual')
        }
        if virtual_updates:
            queue_broadcast({'type': 'virtual_robots', 'data': virtual_updates})
        queue_broadcast({'type': 'ball', 'data': dict(ball_state)})
        now = time.time()
        if sim_state['training'] and now - last_training_broadcast >= 10.0:
            last_training_broadcast = now
            queue_broadcast({
                'type': 'training_state',
                'data': {
                    'training': sim_state['training'],
                    'run_number': sim_state['run_number'],
                    'run_record': run_summary(),
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


class TeamOverride(BaseModel):
    robot_id: str
    team: str


@app.post('/telemetry')
async def receive_telemetry(telemetry: Telemetry):
    data = telemetry.model_dump()
    if telemetry.robot_id not in robots:
        robots[telemetry.robot_id] = {}
    robots[telemetry.robot_id].update(data)
    queue_broadcast({'type': 'telemetry', 'data': data})
    return {'status': 'ok'}


@app.post('/ball')
async def update_ball(state: BallState):
    ball_state.update(state.model_dump())
    queue_broadcast({'type': 'ball', 'data': dict(ball_state)})
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
    queue_broadcast({'type': 'pose_override', 'data': override.model_dump()})
    return {'status': 'ok'}


@app.post('/team')
async def set_team(team_override: TeamOverride):
    robot_id = team_override.robot_id
    if robot_id in robots:
        robots[robot_id]['team'] = team_override.team
    else:
        return {'status': 'error', 'message': 'robot not found'}
    return {'status': 'ok'}


@app.websocket('/robot_ws')
async def robot_websocket_endpoint(ws: WebSocket):
    await ws.accept()
    robot_id = None
    try:
        hello = json.loads(await ws.receive_text())
        if hello.get('type') != 'hello' or not hello.get('robot_id'):
            await ws.close()
            return
        robot_id = hello['robot_id']
        ensure_virtual_robot(robot_id, hello.get('team', 'red'), hello.get('pos', 'high'))
        previous = robot_websocket_clients.get(robot_id)
        if previous is not None and previous is not ws:
            try:
                await previous.close()
            except Exception:
                pass
        robot_websocket_clients[robot_id] = ws
        await asyncio.wait_for(ws.send_json({'type': 'hello', 'robot_id': robot_id}), timeout=0.5)
        await send_robot_state(robot_id, ws)
        while True:
            msg = json.loads(await ws.receive_text())
            msg_type = msg.get('type')
            if msg_type == 'arcade':
                apply_arcade(robot_id, msg.get('straight', 0.0), msg.get('turn', 0.0))
            elif msg_type == 'estimated_pose':
                update_estimated_pose(msg.get('data', {}))
            elif msg_type == 'sync':
                sim_state['synced_robots'].add(robot_id)
    except WebSocketDisconnect:
        pass
    except Exception:
        pass
    finally:
        if robot_id and robot_websocket_clients.get(robot_id) is ws:
            robot_websocket_clients.pop(robot_id, None)
            sim_state['synced_robots'].discard(robot_id)


@app.websocket('/ws')
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    ui_websocket_clients.append(ws)
    try:
        await ws.send_json({'type': 'init', 'robots': robots, 'ball': ball_state})
        await ws.send_json({
            'type': 'training_state',
            'data': {
                'training': sim_state['training'],
                'run_number': sim_state['run_number'],
                'run_record': run_summary(),
                'run_start_time': sim_state['run_start_time'],
                'sim_start': sim_state['sim_start'],
                'sim_time': sim_state['sim_time'],
            }
        })
        while True:
            msg = json.loads(await ws.receive_text())
            if msg.get('type') == 'arcade':
                apply_arcade(msg.get('robot_id'), msg.get('straight', 0.0), msg.get('turn', 0.0))
            elif msg.get('type') == 'train':
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
    except Exception:
        pass
    finally:
        if ws in ui_websocket_clients:
            ui_websocket_clients.remove(ws)


@app.get('/q_files/list')
async def q_files_list():
    return {'files': [os.path.splitext(os.path.basename(f))[0] for f in sim_state['q_files']]}


@app.get('/q_files/{index}')
async def q_files_index(index: int):
    try:
        with open(sim_state['q_files'][index]) as f:
            return json.load(f)
    except Exception:
        return {'error': 'load failed'}


@app.get('/train')
async def set_train(active: bool):
    if active and not sim_state['training']:
        sim_state['training'] = True
        reset_episode()
    elif not active:
        sim_state['training'] = False
        for robot in robots.values():
            robot['training'] = False
            robot['cmd_vel_left'] = robot['cmd_vel_right'] = 0.0


def build_html(config: dict) -> str:
    html = open(os.path.join(os.path.dirname(__file__), 'render.html')).read()
    html = html.replace('MAINSCRIPT', open(
        os.path.join(os.path.dirname(__file__), 'render.js')).read())
    return html.replace('CONFIG_JSON', json.dumps(config))


@app.get('/qworker.js')
async def get_javascript() -> Response:
    js = open('qworker.js').read()
    return Response(content=js, media_type='application/javascript')


@app.get('/', response_class=HTMLResponse)
async def index():
    return build_html(FIELD_CONFIG)


def main():
    parser = argparse.ArgumentParser(description='XRP Soccer Field Telemetry Visualizer')
    parser.add_argument('--host', default='0.0.0.0')
    parser.add_argument('--port', '-p', type=int, default=8080)
    parser.add_argument('--fast', action='store_true', default=False)
    parser.add_argument('--q-file', action='append', default=[])
    args = parser.parse_args()
    sim_state['fast'] = args.fast
    sim_state['q_files'] = args.q_file
    uvicorn.run(app, host=args.host, port=args.port, access_log=False)


if __name__ == '__main__':
    main()
