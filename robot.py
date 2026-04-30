#!/usr/bin/env python3
# /// script
# dependencies = [
#   "websocket-client",
# ]
# ///

import math
import os
import time

import particle_filter
import rl

try:
    from machine import ADC, Pin
    from pestolink import PestoLinkAgent
    from XRPLib.defaults import (board, drivetrain, left_motor, rangefinder,
                                 reflectance, right_motor)
    is_simulation = False
except ImportError:
    import argparse
    import json
    from urllib.parse import urlparse

    from websocket import WebSocketTimeoutException, create_connection

    is_simulation = True

if is_simulation:  # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='http://127.0.0.1:8080')
    parser.add_argument('--robot-id', '--id', '--name', default='RBV-XRP2')
    parser.add_argument('--team', default='red', choices=['red', 'blue'])
    parser.add_argument('--pos', default='high', choices=['high', 'low'])
    parser.add_argument('--mode', default='train', choices=['manual', 'rl', 'train'])
    parser.add_argument('--q-file', default='qtable.json')
    args = parser.parse_args()

    class MockPin:
        def __init__(self, pin_id):
            self.pin_id = pin_id

    class MockADC:
        def __init__(self, pin):
            self.pin = pin

        def read_u16(self):
            return 32000

    class VirtualRobot:
        def __init__(self, url, robot_id):
            self.robot_id = robot_id
            self.team = args.team
            self.pos = args.pos
            self.ws_url = self.build_ws_url(url)
            self.ws = None
            self.left_encoder = 0
            self.right_encoder = 0
            self.distance_cm = 65535.0
            self.reflectance_left = 1.0
            self.reflectance_right = 1.0
            self.imu_heading_deg = 0.0
            self.last_pf_report = 0.0
            self.sim_time = 0.0
            self.sim_start = 0.0
            self.training = False
            self.episodes = [0, 0, 0, []]
            self.terminal_reward = None
            self.reward_total = 0.0
            self.last_reward_total = 0.0
            self.terminal_id = 0
            self.pending_terminal = False
            self.reset = False
            self.last_command = None

        def build_ws_url(self, url):
            parsed = urlparse(url)
            scheme = parsed.scheme if parsed.scheme in ('ws', 'wss') else (
                'wss' if parsed.scheme == 'https' else 'ws')
            path = parsed.path.rstrip('/')
            return f'{scheme}://{parsed.netloc}{path}/robot_ws'

        def close(self):
            ws = self.ws
            self.ws = None
            if ws is not None:
                try:
                    ws.close()
                except Exception:
                    pass

        def connect(self):
            while self.ws is None:
                try:
                    self.ws = create_connection(self.ws_url, timeout=2)
                    self.ws.settimeout(2)
                    self.ws.send(json.dumps({
                        'type': 'hello',
                        'robot_id': self.robot_id,
                        'team': self.team,
                        'pos': self.pos,
                    }))
                    self.last_command = None
                except Exception:
                    self.close()
                    time.sleep(0.5)

        def send(self, message):
            if self.ws is None:
                return False
            try:
                self.ws.send(json.dumps(message))
                return True
            except Exception:
                self.close()
                return False

        def wait_state(self):
            while True:
                if self.ws is None:
                    self.connect()
                try:
                    raw = self.ws.recv()
                except WebSocketTimeoutException:
                    continue
                except Exception:
                    self.close()
                    continue
                try:
                    msg = json.loads(raw)
                except Exception:
                    continue
                if msg.get('type') != 'robot_state':
                    continue
                data = msg.get('data', {})
                self.left_encoder = int(math.floor(data.get('left_encoder', self.left_encoder)))
                self.right_encoder = int(math.floor(data.get('right_encoder', self.right_encoder)))
                self.distance_cm = data.get('distance_cm', self.distance_cm)
                self.reflectance_left = data.get('reflectance_left', self.reflectance_left)
                self.reflectance_right = data.get('reflectance_right', self.reflectance_right)
                self.imu_heading_deg = data.get('imu_heading_deg', self.imu_heading_deg)
                self.training = data.get('training', self.training)
                self.sim_start = data.get('sim_start', self.sim_start)
                self.sim_time = data.get('sim_time', self.sim_time or time.time())
                reward_total = float(data.get('reward_total', self.reward_total))
                terminal_id = int(data.get('terminal_id', self.terminal_id))
                if terminal_id != self.terminal_id:
                    self.terminal_id = terminal_id
                    self.pending_terminal = True
                if data.get('reset', False):
                    if self.terminal_reward is not None:
                        self.episodes[1 if self.terminal_reward > 0 else 2] += 1
                        self.episodes[3].append(1 if self.terminal_reward > 0 else -1)
                    else:
                        self.episodes[3].append(0)
                    print(
                        f'{robot_name}  episodes {self.episodes[0]}, '
                        f'W {self.episodes[1]}, L {self.episodes[2]}, '
                        f'RP100 {sum(self.episodes[3][-100:]) / len(self.episodes[3][-100:]):4.2f}')
                    self.episodes[0] += 1
                    self.terminal_reward = None
                    self.reward_total = 0.0
                    self.last_reward_total = 0.0
                    self.terminal_id = 0
                    self.pending_terminal = False
                    self.reset = True
                else:
                    if reward_total < self.reward_total:
                        self.last_reward_total = 0.0
                    self.reward_total = reward_total
                return

        def sync(self):
            self.send({'type': 'sync', 'robot_id': self.robot_id})

        def send_pose(self, pose):
            now = self.sim_time
            if now - self.last_pf_report < 0.25:
                return
            data = {
                'robot_id': self.robot_id,
                'x_mm': round(pose['x_mm'], 1),
                'y_mm': round(pose['y_mm'], 1),
                'heading_deg': round(pose['heading_deg'], 1),
                'std_x_mm': round(pose['std_x_mm'], 1),
                'std_y_mm': round(pose['std_y_mm'], 1),
                'std_heading_deg': round(pose['std_heading_deg'], 1),
            }
            if self.send({'type': 'estimated_pose', 'data': data}):
                self.last_pf_report = now

        def get_reward(self):
            reward = self.reward_total - self.last_reward_total
            self.last_reward_total = self.reward_total
            terminal = self.pending_terminal
            self.pending_terminal = False
            if terminal:
                self.terminal_reward = reward
            return reward, terminal

        def arcade(self, straight, turn):
            command = (round(straight, 4), round(turn, 4))
            if command == self.last_command:
                return
            if self.send({
                'type': 'arcade',
                'robot_id': self.robot_id,
                'straight': command[0],
                'turn': command[1],
            }):
                self.last_command = command

    virtual_robot = VirtualRobot(args.simulator, args.robot_id)

    class MockPestoLink:
        def __init__(self, name):
            self.name = name

        def is_connected(self):
            return True

        def get_axis(self, axis):
            return 0.0

        def telemetryPrint(self, val, color='#000000'):
            pass

        def telemetryPrintBatteryVoltage(self, voltage):
            pass

    class MockDrivetrain:
        def arcade(self, straight, turn):
            virtual_robot.arcade(straight, turn)

    class MockMotor:
        def __init__(self, is_left):
            self.is_left = is_left

        def get_position_counts(self):
            return virtual_robot.left_encoder if self.is_left else virtual_robot.right_encoder

    class MockRangefinder:
        def distance(self):
            return virtual_robot.distance_cm

    class MockReflectance:
        def get_left(self):
            return virtual_robot.reflectance_left

        def get_right(self):
            return virtual_robot.reflectance_right

    class MockIMU:
        def calibrate(self):
            pass

        def get_heading(self):
            return virtual_robot.imu_heading_deg

    class MockBoard:
        imu = MockIMU()

    Pin = MockPin  # noqa
    ADC = MockADC   # noqa
    PestoLinkAgent = MockPestoLink  # noqa
    drivetrain = MockDrivetrain()  # noqa
    left_motor = MockMotor(True)  # noqa
    right_motor = MockMotor(False)  # noqa
    rangefinder = MockRangefinder()  # noqa
    reflectance = MockReflectance()  # noqa
    board = MockBoard()  # noqa
    robot_name = args.robot_id
    robot_team = args.team
    robot_mode = args.mode
    q_file = args.q_file
else:
    robot_name = 'RBV-XRP2'
    robot_team = 'red'
    q_file = 'qtable.json'
    robot_mode = 'rl' if os.path.exists(q_file) else 'manual'
    board.imu.calibrate()

pestolink = PestoLinkAgent(robot_name)
pf = particle_filter.ParticleFilter(team=robot_team)
agent = rl.QAgent(team=robot_team, epsilon=0.5 if robot_mode == 'train' else 0)
agent.load(q_file)
last_save = time.time()
next_action_time = 0
last_print = 0

while True:  # noqa
    if pestolink.is_connected():
        if is_simulation:
            virtual_robot.wait_state()
            if virtual_robot.reset:
                virtual_robot.reset = False
                pf.reset()
                agent.reset_episode()
        if robot_mode == 'manual':
            rotation = -1 * pestolink.get_axis(2)
            throttle = -1 * pestolink.get_axis(1)
            drivetrain.arcade(throttle, rotation)
        now = virtual_robot.sim_time if is_simulation else time.time()
        left_ticks = left_motor.get_position_counts()
        right_ticks = right_motor.get_position_counts()
        distance_cm = rangefinder.distance()
        refl_l = reflectance.get_left()
        refl_r = reflectance.get_right()
        heading = board.imu.get_heading()
        pf.step(left_ticks, right_ticks, distance_cm, refl_l, refl_r, heading)
        pose = pf.get_pose_with_error()
        if is_simulation:
            virtual_robot.send_pose(pose)
        if not next_action_time:
            next_action_time = now
        if robot_mode != 'manual' and now >= next_action_time and (
                robot_mode != 'train' or virtual_robot.training):
            state = agent.discretize(pose, distance_cm, refl_l, refl_r, agent.last_action)
            reward = 0
            terminal = False
            if is_simulation and robot_mode == 'train':
                reward, terminal = virtual_robot.get_reward()
            agent.learn_from_transition(state, reward, terminal)
            if terminal:
                agent.reset_episode()
                if robot_mode == 'train' and time.time() - last_save > 10:
                    agent.save(q_file)
                    last_save = time.time()
            action = agent.choose_action(state)
            straight, turn = agent.command(action)
            drivetrain.arcade(straight, turn)
            agent.remember(state, action)
            next_action_time = max(now, next_action_time + 0.25)
        if robot_mode == 'train' and time.time() - last_save > 30:
            agent.save(q_file)
            last_save = time.time()
        if time.time() - last_print > 10:
            print(f'{robot_name} {left_ticks:8d} {right_ticks:8d} {distance_cm:7.1f} '
                  f'{refl_l:4.2f} {refl_r:4.2f} {heading:5.1f} '
                  f'{pose["x_mm"]:7.1f} {pose["y_mm"]:6.1f} {pose["heading_deg"]:5.1f}')
            last_print = time.time()
        if is_simulation:
            virtual_robot.sync()
        else:
            battery_voltage = ADC(Pin('BOARD_VIN_MEASURE')).read_u16() / (1024 * 64 / 14)
            pestolink.telemetryPrintBatteryVoltage(battery_voltage)
    else:
        drivetrain.arcade(0, 0)
