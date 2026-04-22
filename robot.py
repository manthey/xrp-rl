import math

try:
    from machine import ADC, Pin
    from pestolink import PestoLinkAgent
    from XRPLib.defaults import (drivetrain, left_motor, rangefinder,
                                 reflectance, right_motor)
    is_simulation = False
except ImportError:
    import argparse
    import json
    import urllib.request
    is_simulation = True

if is_simulation:  # noqa
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulator', default='http://127.0.0.1:8080')
    parser.add_argument('--robot-id', default='RBV-XRP2')
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

            initialization_request = urllib.request.Request(
                f"{self.url}/pose_override",
                data=json.dumps({
                    'robot_id': self.robot_id,
                    'world_x_mm': 0.0,
                    'world_y_mm': 0.0,
                    'world_heading_deg': 0.0
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            urllib.request.urlopen(initialization_request)

        def update_state(self):
            try:
                response = urllib.request.urlopen(f"{self.url}/robot?robot_id={self.robot_id}")
                data = json.loads(response.read().decode('utf-8'))
                self.left_encoder = int(math.floor(data.get('left_encoder', self.left_encoder)))
                self.right_encoder = int(math.floor(data.get('right_encoder', self.right_encoder)))
                self.distance_cm = data.get('distance_cm', self.distance_cm)
                self.reflectance_left = data.get('reflectance_left', self.reflectance_left)
                self.reflectance_right = data.get('reflectance_right', self.reflectance_right)
            except Exception:
                pass

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
            request = urllib.request.Request(
                f"{virtual_robot.url}/arcade",
                data=json.dumps({
                    'robot_id': virtual_robot.robot_id,
                    'straight': straight,
                    'turn': turn
                }).encode('utf-8'),
                headers={'Content-Type': 'application/json'}
            )
            try:
                urllib.request.urlopen(request)
            except Exception:
                pass

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

    Pin = MockPin  # noqa
    ADC = MockADC  # noqa
    PestoLinkAgent = MockPestoLink  # noqa
    drivetrain = MockDrivetrain()  # noqa
    left_motor = MockMotor(True)  # noqa
    right_motor = MockMotor(False)  # noqa
    rangefinder = MockRangefinder()  # noqa
    reflectance = MockReflectance()  # noqa
    robot_name = args.robot_id
else:
    robot_name = 'RBV-XRP2'

pestolink = PestoLinkAgent(robot_name)

while True:
    if pestolink.is_connected():
        rotation = -1 * pestolink.get_axis(2)
        throttle = -1 * pestolink.get_axis(1)

        drivetrain.arcade(throttle, rotation)

        print(left_motor.get_position_counts(), right_motor.get_position_counts(),
              rangefinder.distance(), reflectance.get_left(), reflectance.get_right())

        batteryVoltage = (ADC(Pin('BOARD_VIN_MEASURE')).read_u16()) / (1024 * 64 / 14)
        pestolink.telemetryPrintBatteryVoltage(batteryVoltage)
        pestolink.telemetryPrint(left_motor.get_position_counts())
        pestolink.telemetryPrint(right_motor.get_position_counts())
        pestolink.telemetryPrint(rangefinder.distance())
        pestolink.telemetryPrint(reflectance.get_left())
        pestolink.telemetryPrint(reflectance.get_right())
    else:  # default behavior when no BLE connection is open
        drivetrain.arcade(0, 0)
