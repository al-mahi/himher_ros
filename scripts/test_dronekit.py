#!/usr/bin/python

from __future__ import print_function
import time
from multiprocessing import Process
import copy
import numpy as np
from scipy.stats import multivariate_normal
import rospy
from dronekit import Vehicle, connect, LocationGlobalRelative, LocationGlobal, LocationLocal, VehicleMode
from pymavlink import mavutil


def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    """
    Move vehicle in direction based on specified velocity vectors.
    file:///home/alien/Downloads/dronekit/guide/vehicle_state_and_parameters.html
    """

    while not vehicle.mode.name=="GUIDED":
        vehicle.mode = VehicleMode("GUIDED")

    rospy.logdebug("{} vel mode".format(vehicle.mode.name))
    # msg = vehicle.message_factory.set_position_target_local_ned_encode(
    #     0,  # time_boot_ms (not used)
    #     0, 0,  # target system, target component
    #     mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
    #     0b0000111111000110,  # type_mask (only speeds enabled)
    #     0, 0, 0,  # x, y, z positions (not used)
    #     velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
    #     0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
    #     0, 8.)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)

    msg2 = vehicle.message_factory.set_attitude_target_encode(
        0,
        0,                                         # target system
        0,                                         # target component
        0b11110001,                                # type mask: bit 1 is LSB
        [0., 0., 0., 0.],    # q
        2.,                                         # body roll rate in radian
        0.,                                         # body pitch rate in radian
        1.,                                  # body yaw rate in radian
        0)                                    # thrust
    # send command to vehicle on 1 Hz cycle
    for _ in range(0, 5):
        # vehicle.send_mavlink(msg)
        time.sleep(.1)
        vehicle.send_mavlink(msg2)
        time.sleep(.1)

    # @vehicle.on_message("*")
    # def handle_msg(veh, name, msg):
    #     print("on msg ", veh.mode.name, name, msg)


vehicle = connect('udpin:0.0.0.0:{}'.format(14561), wait_ready=True, baud=57600)

vehicle.home_location = LocationGlobal(
    lat=36.162263,
    lon=-96.835559,
    alt=315
)

print("velocity1")
print("---------------------")
send_ned_velocity(2, 1, -2)
print("velocity2")
print("---------------------")
send_ned_velocity(-2, 1, -4)
print("velocity3")
print("---------------------")
send_ned_velocity(2, 1, -2)
print("velocity done")
print("---------------------")


for _ in range(5):
    time.sleep(1)

print("done")