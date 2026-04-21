# Import necessary modules
# import math
# import time

# import bluetooth
from machine import ADC, Pin

try:
    from pestolink import PestoLinkAgent
    from XRPLib.defaults import (drivetrain, left_motor, rangefinder,
                                 reflectance, right_motor)
except Exception:
    class VirtualRobot:
        pass
    # TODO: call the simulate server to get encoder positions, distance sensor,
    # and reflectance. Expose enough functions to support drivetrain.arcade
    vr = VirtualRobot()


# Choose the name your robot shows up as in the Bluetooth paring menu
# Name should be 8 characters max!
robot_name = 'RBV-XRP2'

# Create an instance of the PestoLinkAgent class
pestolink = PestoLinkAgent(robot_name)

# Start an infinite loop
while True:
    if pestolink.is_connected():  # Check if a BLE connection is established
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
