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
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from himher_ros.msg import BelParam, BelJointParam, BelJoint


def send_ned_velocity(velocity_x, velocity_y, velocity_z):
    """
    Move vehicle in direction based on specified velocity vectors.
    file:///home/alien/Downloads/dronekit/guide/vehicle_state_and_parameters.html
    """

    while not vehicle.mode.name == "GUIDED":
        vehicle.mode = VehicleMode("GUIDED")

    # rospy.logdebug("{} vel mode".format(tag, vehicle.mode.name))
    msg = vehicle.message_factory.set_position_target_local_ned_encode(
        0,  # time_boot_ms (not used)
        0, 0,  # target system, target component
        mavutil.mavlink.MAV_FRAME_LOCAL_NED,  # frame
        0b0000111111000111,  # type_mask (only speeds enabled)
        0, 0, 0,  # x, y, z positions (not used)
        velocity_x, velocity_y, velocity_z,  # x, y, z velocity in m/s
        0, 0, 0,  # x, y, z acceleration (not supported yet, ignored in GCS_Mavlink)
        0, 0)  # yaw, yaw_rate (not supported yet, ignored in GCS_Mavlink)
    # send command to vehicle on 1 Hz cycle
    for _ in range(0, 4):
        vehicle.send_mavlink(msg)
        time.sleep(.1)


def locGPS():
    return np.array([
        vehicle.location.global_frame.lon,
        vehicle.location.global_frame.lat,
        vehicle.location.global_frame.alt
    ])


def locNEU():
    return np.array([
        vehicle.location.local_frame.east,
        vehicle.location.local_frame.north,
        -vehicle.location.local_frame.down
    ])


def toLocMesh(neu):
    loc = np.round(neu)
    return np.where(np.logical_and(np.logical_and(xx == loc[0], yy == loc[1]), zz == loc[2]))


def phiFence(X):
    def calc(x):
        tol = 2
        if x[0] < minX + tol or x[0] > maxX - tol or \
                x[1] < minY + tol or x[1] > maxY - tol or \
                x[2] < minZ + tol or x[2] > maxZ - tol:
            return np.linalg.norm(x)
        return 0.

    res = np.array(map(calc, X))
    return res


def updateMyBel(nbsJointBel, nb):
    """:type nbsJointBel: BelJointParam"""

    if nbsJointBel.name == '': return
    # rospy.logdebug("{}<=={} start {}<=={}".format(
    #     tag, nb, [msgBelJoint.name, msgBelJoint.A.exist, msgBelJoint.B.exist, msgBelJoint.C.exist],
    #     [nbsJointBel.name, nbsJointBel.A.exist, nbsJointBel.B.exist, nbsJointBel.C.exist])
    # )

    if name != "A" and nbsJointBel.A.exist:  # no need to hear about itself and hear if any msg exist
        msgBelJoint.A = nbsJointBel.A
        msgBelJoint.A.src = nb

    if name != "B" and nbsJointBel.B.exist:
        msgBelJoint.B = nbsJointBel.B
        msgBelJoint.B.src = nb

    if name != "C" and nbsJointBel.C.exist:
        msgBelJoint.C = nbsJointBel.C
        msgBelJoint.C.src = nb

    # rospy.logdebug("{}<=={} done {}<=={}".format(
    #     tag, nb, [msgBelJoint.name, msgBelJoint.A.exist, msgBelJoint.B.exist, msgBelJoint.C.exist],
    #     [nbsJointBel.name, nbsJointBel.A.exist, nbsJointBel.B.exist, nbsJointBel.C.exist])
    # )


def fastNN(P):
    ceilRes = lambda x: resolution * (np.divmod(x, resolution)[0] + 1)
    floorRes = lambda x: resolution * (np.divmod(x, resolution)[0])
    minp = None
    mind = np.inf
    # print("P={}".format(P))
    for up in ["{0:03b}".format(i) for i in range(8)]:
        p = np.zeros(3, dtype=np.int)
        for i in range(3):
            p[i] = ceilRes(P[i]) if up[i] == '1' else floorRes(P[i])
        d = np.linalg.norm(P-p)
        if mind > d:
            mind = d
            minp = p.copy()
        # print(up, p, d, mind, minp)
    return mind, tuple(minp)

instanceOf = {
    "A": 0,
    "B": 1,
    "C": 2,
    "D": 3,
    "E": 4
}

portOf = {
    "A": 14551,
    "B": 14561,
    "C": 14571
}

###############################################################################
# Execution starts and ROS initiated here

rospy.init_node(name="quad", anonymous=True, log_level=rospy.DEBUG)

name = rospy.get_param("~name")
dim = int(rospy.get_param("/dim"))
minX = int(rospy.get_param("/minX"))
maxX = int(rospy.get_param("/maxX"))
minY = int(rospy.get_param("/minY"))
maxY = int(rospy.get_param("/maxY"))
minZ = int(rospy.get_param("/minZ"))
maxZ = int(rospy.get_param("/maxZ"))
resolution = int(rospy.get_param("/res"))
oLat = float(rospy.get_param("/origin_lat"))
oLon = float(rospy.get_param("/origin_lon"))
oAlt = float(rospy.get_param("/origin_alt"))
neighbors = rospy.get_param("~neighbors").split("_")
team = [c for c in "ABC"]
tag = "{}: ".format(name)
vel = 2.5  # m/s

###############################################################################
# Meshgrid Space
x = np.arange(minX, maxX + resolution, resolution)
y = np.arange(minY, maxY + resolution, resolution)
z = np.arange(minZ, maxZ + resolution, resolution)
xx, yy, zz = np.meshgrid(x, y, z)
X = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

vehicle = connect('udpin:0.0.0.0:{}'.format(portOf[name]), wait_ready=True, baud=57600)
vehicle.home_location = LocationGlobal(lat=oLat, lon=oLon, alt=oAlt)

rospy.sleep(3)
if name!="A": vehicle.mode = VehicleMode("GUIDED")

# initialize belief calculations
belExplore = np.zeros(xx.shape)
belPos = np.zeros(xx.shape)
belFence = np.zeros(xx.shape)

belExplore[0, 0, 0] = 1

# TODO
belHRI = np.zeros(xx.shape)
belTemp = np.zeros(xx.shape)
belHumidity = np.zeros(xx.shape)

J = np.zeros(xx.shape)

wCollision = 1.
wFence = .05
wExplore = 1.
zCollision = 2.
zFence = 1.

###############################################################################
# ROS main execution loop

recvBelJointParam = {}  # type:dict[str, BelJoint]
q_size = 10


def cbBelJointParam(msg):
    """:type msg BelJointParam"""
    recvBelJointParam[msg.name] = msg
    # rospy.logdebug("{}<--{} rcvs {}".format(tag, msg.name, recvBelJointParam.keys()))


pubsBelJointfParam = {}  # type: [dict, rospy.Publisher]

for nb in neighbors:
    pubsBelJointfParam[nb] = rospy.Publisher(
        name + '/bel_joint_param_{}'.format(nb), data_class=BelJointParam, queue_size=q_size
    )

pubBelJoint = rospy.Publisher(
    name + '/bel_joint', data_class=BelJoint, queue_size=q_size
)

for nb in neighbors:
    rospy.Subscriber(
        "/UAV/{}/bel_joint_param_{}".format(nb, name), callback=cbBelJointParam,
        data_class=BelJointParam, queue_size=q_size
    )

msgBelJoint = BelJointParam()
msgBelJoint.name = name
msgBelJoint.A = BelParam()
msgBelJoint.B = BelParam()
msgBelJoint.C = BelParam()

rate = rospy.Rate(1)

for nb in neighbors:
    rospy.logdebug("{} neighbors {}".format(tag, neighbors))
    while pubsBelJointfParam[nb].get_num_connections() < 1:
        rospy.logdebug("{} Waiting for nb {} to connect...".format(tag, nb))
        rospy.sleep(1)

while not rospy.is_shutdown():
    np.random.shuffle(neighbors)  # WARNING! don't deprive any neighbor because of its ordering
    msgOwnBel = BelParam()
    msgOwnBel.name = name
    msgOwnBel.src = name
    msgOwnBel.exist = True
    msgOwnBel.east = vehicle.location.local_frame.east
    msgOwnBel.north = vehicle.location.local_frame.east
    msgOwnBel.up = -vehicle.location.local_frame.down
    msgOwnBel.std_ned = 2 * resolution
    # msgBelSelfParam.temperature_reading = TODO
    # msgBelSelfParam.humidity_reading = TODO

    if name == "A": msgBelJoint.A = msgOwnBel
    if name == "B": msgBelJoint.B = msgOwnBel
    if name == "C": msgBelJoint.C = msgOwnBel

    # calc what robot thinks of others
    for nb in neighbors:
        nbsJointBel = BelJointParam()
        if nb == "A" and recvBelJointParam.has_key(nb): nbsJointBel = copy.deepcopy(recvBelJointParam[nb])
        if nb == "B" and recvBelJointParam.has_key(nb): nbsJointBel = copy.deepcopy(recvBelJointParam[nb])
        if nb == "C" and recvBelJointParam.has_key(nb): nbsJointBel = copy.deepcopy(recvBelJointParam[nb])
        updateMyBel(nbsJointBel, nb)

    for toNb in neighbors:
        msg = copy.deepcopy(msgBelJoint)
        #  dont relay back the msg thats been heard from the one in the first place
        if msg.A.src == toNb: msg.A.exist = False
        if msg.B.src == toNb: msg.B.exist = False
        if msg.C.src == toNb: msg.C.exist = False
        msg.A.src = name
        msg.B.src = name
        msg.C.src = name
        pubsBelJointfParam[nb].publish(msg)

    # Intention: Collision avoidance
    J = np.zeros(shape=xx.shape)
    np.random.shuffle(team)
    for uav in team:
        if uav == name: continue

        muPos = None
        if uav == "A":
            if msgBelJoint.A.exist:
                muPos = np.array([
                    msgBelJoint.A.east,
                    msgBelJoint.A.north,
                    msgBelJoint.A.up
                ])
        if uav == "B":
            if msgBelJoint.B.exist:
                muPos = np.array([
                    msgBelJoint.B.east,
                    msgBelJoint.B.north,
                    msgBelJoint.B.up
                ])
        if uav == "C":
            if msgBelJoint.C.exist:
                muPos = np.array([
                    msgBelJoint.C.east,
                    msgBelJoint.C.north,
                    msgBelJoint.C.up
                ])

        belPos = np.zeros(shape=xx.shape)
        if not muPos is None:
            covPos = 3 * resolution * np.identity(3)
            rv = multivariate_normal(muPos, covPos)
            belPos = -np.reshape(rv.pdf(X), newshape=xx.shape, order='C')
            belPos = (belPos - belPos.min()) / belPos.ptp()

        # Intention: Geo Fence
        belFence = -np.reshape(phiFence(X), newshape=xx.shape, order='C')
        belFence = (belFence - belFence.min()) / belFence.ptp()
        belFence = np.nan_to_num(belFence, 0.)

        # Intention: Exploration
        belExplore = (belExplore - belExplore.min()) / belExplore.ptp()
        belExplore = np.nan_to_num(belExplore, 0.)

        J += (
                wCollision * belPos
        )

    J /= zCollision
    J = (J - J.min()) / J.ptp()
    J = np.nan_to_num(J, 0.)

    wExplore = J.max()/2.

    J += (
            wFence * belFence +
            wExplore * belExplore
    )

    J = (J - J.min()) / J.ptp()

    loc = locNEU()
    dNN, indNNGrid = fastNN(loc)
    indNNTupOfArr = np.where(np.logical_and(np.logical_and(xx == indNNGrid[0], yy == indNNGrid[1]), zz == indNNGrid[2]))
    indNN = tuple(map(lambda arr: 0 if arr.size != 1 else arr[0], indNNTupOfArr))
    belExplore[indNN] = 1.
    # Gradient Calculation
    dFy, dFx, dFz = v, u, w = np.gradient(J)

    # the magnitude of the gradient is too small
    # However good for visualization
    gmag = np.sqrt(u**2 + v**2 + w**2)
    gmagNorm = (gmag - gmag.min())/gmag.ptp()
    uvwLoc = np.array([u[indNN], v[indNN], w[indNN]])
    rospy.logdebug("{} uvwOrg {}".format(tag, uvwLoc))
    if np.isclose(np.linalg.norm(uvwLoc), 0.):
        uvwLoc = np.random.uniform(low=-vel, high=vel, size=3)
        uvwLoc[2] = np.random.uniform(low=-.5, high=.0)
        rospy.logdebug("{} {} {} {} {} *{} {}".format(tag, loc, indNNGrid, indNN, dNN, uvwLoc, gmag[indNN]))
    else:
        uvwLoc = np.array(map(lambda x: -1. if x>0. else 1., uvwLoc))
        rospy.logdebug("{} {} {} {} {} {} {}".format(tag, loc, indNNGrid, indNN, dNN, uvwLoc, gmag[indNN]))

    # if name != "A": send_ned_velocity(vel * uvwLoc[0], vel * uvwLoc[1], vel * uvwLoc[2])
    msgHRI = BelJoint()
    msgHRI.name = name
    msgHRI.src = name
    msgHRI.locNED = loc.flatten()
    msgHRI.gnorm = gmag.flatten()
    msgHRI.u = u.flatten()
    msgHRI.v = v.flatten()
    msgHRI.w = w.flatten()
    msgHRI.uvwLoc = uvwLoc.flatten()
    pubBelJoint.publish(msgHRI)
    rate.sleep()
