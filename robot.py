import math
import time

import particle_filter

try:
    from machine import ADC, Pin
    from pestolink import PestoLinkAgent
    from XRPLib.defaults import (board, drivetrain, left_motor, rangefinder,
                                 reflectance, right_motor)
    is_simulation = False
except ImportError:
    import argparse

    import requests
    is_simulation = True

if is_simulation:  # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='http://127.0.0.1:8080')
    parser.add_argument('--robot-id', '--id', '--name', default='RBV-XRP2')
    parser.add_argument('--team', default='red', choices=['red', 'blue'])
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
            self.url = url
            self.robot_id = robot_id
            self.left_encoder = 0
            self.right_encoder = 0
            self.distance_cm = 65535.0
            self.reflectance_left = 1.0
            self.reflectance_right = 1.0
            self.imu_heading_deg = 0.0
            self.team = args.team
            self.session = requests.Session()
            self.last_pf_report = 0

            self.session.post(
                f'{self.url}/pose_override',
                json={
                    'robot_id': self.robot_id,
                    'world_x_mm': -900 if self.team == 'red' else 900,
                    'world_y_mm': 300 if self.team == 'red' else -300,
                    'world_heading_deg': 0 if self.team == 'red' else 180,
                })
            self.session.post(
                f'{self.url}/team',
                json={
                    'robot_id': self.robot_id,
                    'team': self.team,
                })

        def update_state(self):
            try:
                data = self.session.get(f'{self.url}/robot?robot_id={self.robot_id}').json()
                self.left_encoder = int(math.floor(data.get('left_encoder', self.left_encoder)))
                self.right_encoder = int(math.floor(data.get('right_encoder', self.right_encoder)))
                self.distance_cm = data.get('distance_cm', self.distance_cm)
                self.reflectance_left = data.get('reflectance_left', self.reflectance_left)
                self.reflectance_right = data.get('reflectance_right', self.reflectance_right)
                self.imu_heading_deg = data.get('imu_heading_deg', self.imu_heading_deg)
            except Exception:
                pass

        def send_pose(self):
            now = time.time()
            if now - self.last_pf_report < 0.1:
                return
            self.last_pf_report = now
            virtual_robot.session.post(
                f'{virtual_robot.url}/estimated_pose',
                json={
                    'robot_id': virtual_robot.robot_id,
                    'x_mm': pose['x_mm'],
                    'y_mm': pose['y_mm'],
                    'heading_deg': pose['heading_deg'],
                    'std_x_mm': pose['std_x_mm'],
                    'std_y_mm': pose['std_y_mm'],
                    'std_heading_deg': pose['std_heading_deg'],
                })

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
            virtual_robot.session.post(
                f'{virtual_robot.url}/arcade',
                json={
                    'robot_id': virtual_robot.robot_id,
                    'straight': straight,
                    'turn': turn
                })
            virtual_robot.update_state()

    class MockMotor:
        def __init__(self, is_left):
            self.is_left = is_left

        def get_position_counts(self):
            if self.is_left:
                return virtual_robot.left_encoder
            return virtual_robot.right_encoder

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

    class MockBoard():
        imu = MockIMU()

    Pin = MockPin  # noqa
    ADC = MockADC  # noqa
    PestoLinkAgent = MockPestoLink  # noqa
    drivetrain = MockDrivetrain()  # noqa
    left_motor = MockMotor(True)  # noqa
    right_motor = MockMotor(False)  # noqa
    rangefinder = MockRangefinder()  # noqa
    reflectance = MockReflectance()  # noqa
    board = MockBoard()  # noqa
    robot_name = args.robot_id
    robot_team = args.team
else:
    robot_name = 'RBV-XRP2'
    robot_team = 'red'
    board.imu.calibrate()

pestolink = PestoLinkAgent(robot_name)

pf = particle_filter.ParticleFilter(team=robot_team)
last_print = 0

while True:
    if pestolink.is_connected():
        rotation = -1 * pestolink.get_axis(2)
        throttle = -1 * pestolink.get_axis(1)

        drivetrain.arcade(throttle, rotation)

        left_ticks = left_motor.get_position_counts()
        right_ticks = right_motor.get_position_counts()
        distance_cm = rangefinder.distance()
        refl_l = reflectance.get_left()
        refl_r = reflectance.get_right()
        heading = board.imu.get_heading()

        pf.step(left_ticks, right_ticks, distance_cm, refl_l, refl_r, heading)
        pose = pf.get_pose_with_error()
        if is_simulation:
            virtual_robot.send_pose()
        if time.time() - last_print > 10:
            print(f'{robot_name} {left_ticks:8d} {right_ticks:8d} {distance_cm:7.1f} '
                  f'{refl_l:4.2f} {refl_r:4.2f} {heading:5.1f} '
                  f'{pose["x_mm"]:7.1f} {pose["y_mm"]:6.1f} {pose["heading_deg"]:5.1f}')
            last_print = time.time()
        batteryVoltage = (ADC(Pin('BOARD_VIN_MEASURE')).read_u16()) / (1024 * 64 / 14)
        pestolink.telemetryPrintBatteryVoltage(batteryVoltage)
        # This won't really work -- pestolink only sends the first 8 characters
        # of any telemetry and only sends one piece of telemetry every 0.5 s,
        # so we will get different values at different times
        # pestolink.telemetryPrint(str(left_ticks), '000000')
        # pestolink.telemetryPrint(str(right_ticks), '000000')
        # pestolink.telemetryPrint(str(distance_cm), '000000')
        # pestolink.telemetryPrint(str(refl_l), '000000')
        # pestolink.telemetryPrint(str(refl_r), '000000')
        # pestolink.telemetryPrint(str(heading()), '000000')
    else:  # default behavior when no BLE connection is open
        drivetrain.arcade(0, 0)
