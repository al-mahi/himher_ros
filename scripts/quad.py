#!/usr/bin/python

from __future__ import print_function
import time
from multiprocessing import Process
import copy
import numpy as np
from scipy.stats import multivariate_normal
from scipy.special import logit
from scipy.spatial import KDTree
from scipy.signal import unit_impulse
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.mixture.gaussian_mixture import _compute_precision_cholesky
import rospy
from dronekit import Vehicle, connect, LocationGlobalRelative, LocationGlobal, LocationLocal, VehicleMode
from pymavlink import mavutil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import timeit

from himher_ros.msg import BelParam, BelJointParam, BelJoint


def send_ned_velocity(uvwNormVel):
    """
    Move vehicle in direction based on specified velocity vectors.
    file:///home/alien/Downloads/dronekit/guide/vehicle_state_and_parameters.html
    """
    vel = 3.  # m/s
    uvwLocNormVel = np.nan_to_num(uvwNormVel, 0.)
    velocity_x, velocity_y, velocity_z = vel * uvwLocNormVel[1], vel * uvwLocNormVel[0], vel * uvwLocNormVel[2]

    if vehicle.mode.name != "GUIDED":
        rospy.logerr("{} send_ned_ve not in Guided mode {}".format(tag, vehicle.mode.name))
        return

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
    vehicle.send_mavlink(msg)
    # for _ in range(2):
    #     vehicle.send_mavlink(msg)
    #     rospy.sleep(rospy.Duration.from_sec(.15))


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


def indXtoJ(indX):
    """converts index from NED coordinate in array X to index of array J"""
    return np.unravel_index(indX % xx.size, xx.shape)


def phiFence():
    fence = np.zeros(xx.shape)
    for i in range(3):
        fence += 100. * (unit_impulse(xx.swapaxes(0, i).shape, (0,)) + unit_impulse(xx.swapaxes(0, i).shape, (-1,))).swapaxes(0, i)
        fence += 060. * (unit_impulse(xx.swapaxes(0, i).shape, (1,)) + unit_impulse(xx.swapaxes(0, i).shape, (-2,))).swapaxes(0, i)
        fence += 010. * (unit_impulse(xx.swapaxes(0, i).shape, (2,)) + unit_impulse(xx.swapaxes(0, i).shape, (-3,))).swapaxes(0, i)
    return fence


def normalizeArr(arr, maxVal=100.): return maxVal * np.nan_to_num((arr - arr.min()) / arr.ptp(), 0.)


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
rospy.logdebug("{} team {}".format(name, team))
tag = "{}: ".format(name)
scaleX = maxX-minX
scaleY = maxY-minY
scaleZ = maxZ-minZ
###############################################################################
# Meshgrid Space
x = np.arange(minX, maxX + resolution, resolution)
y = np.arange(minY, maxY + resolution, resolution)
z = np.arange(minZ, maxZ + resolution, resolution)
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
X = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

vehicle = connect('udpin:0.0.0.0:{}'.format(portOf[name]), wait_ready=True, baud=57600)
vehicle.home_location = LocationGlobal(lat=oLat, lon=oLon, alt=oAlt)

rospy.sleep(3)
if name!="A":
    vehicle.mode = VehicleMode("GUIDED")
    rospy.sleep(3)
    vehicle.mode = VehicleMode("GUIDED")

# initialize belief calculations
belExplore = np.zeros(xx.shape)

belExplore[0, :, :] = 1.
belExplore[:, 0, :] = 1.
belExplore[:, :, 0] = 1.
belExplore[-1, :, :] = 1.
belExplore[:, -1, :] = 1.
belExplore[:, :, -1] = 1.


observedTemp = np.zeros(xx.shape)
observedHumidity = np.zeros(xx.shape)

J = np.zeros(xx.shape)
covPos = .3 * np.array([
    [scaleX/resolution, 0, 0],
    [0., scaleY/resolution, 0],
    [0., 0, scaleZ/resolution]
])

rospy.logdebug("{} cov={}".format(tag, covPos))

wCollision = 1.
wFence = .5
wExplore = 1.


###############################################################################


###############################################################################
# ROS main execution loop
q_size = 2


def updateMyBel(nbsJointBel):
    """:type nbsJointBel: BelJointParam"""

    if nbsJointBel.name == '': return

    if name != "A" and nbsJointBel.A.exist:  # no need to hear about itself and hear if any msg exist
        msgBelJoint.A = nbsJointBel.A
        msgBelJoint.A.src = nbsJointBel.name

    if name != "B" and nbsJointBel.B.exist:
        msgBelJoint.B = nbsJointBel.B
        msgBelJoint.B.src = nbsJointBel.name

    if name != "C" and nbsJointBel.C.exist:
        msgBelJoint.C = nbsJointBel.C
        msgBelJoint.C.src = nbsJointBel.name

    # rospy.logdebug("{}<=={} done {}<=={}".format(
    #     tag, nb, [msgBelJoint.name, msgBelJoint.A.exist, msgBelJoint.B.exist, msgBelJoint.C.exist],
    #     [nbsJointBel.name, nbsJointBel.A.exist, nbsJointBel.B.exist, nbsJointBel.C.exist])
    # )


def cbBelJointParam(msg):
    """:type msg BelJointParam"""
    nbsJointBel = copy.deepcopy(msg)
    updateMyBel(nbsJointBel=nbsJointBel)


pubsBelJointfParam = {}  # type: [dict, rospy.Publisher]
subsBelJointfParam = {}  # type: [dict, rospy.SubscribeListener]

for nb in neighbors:
    pubsBelJointfParam[nb] = rospy.Publisher(
        name + '/bel_joint_param_{}'.format(nb), data_class=BelJointParam, queue_size=q_size
    )

pubBelJoint = rospy.Publisher(
    name + '/bel_joint', data_class=BelJoint, queue_size=q_size
)

for nb in neighbors:
    subsBelJointfParam[nb] = rospy.Subscriber(
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
    rospy.logdebug("{} pub neighbors {}".format(tag, neighbors))
    while pubsBelJointfParam[nb].get_num_connections() < 1:
        rospy.logdebug("{} Waiting for nb {} to connect...".format(tag, nb))
        rospy.sleep(1)

for nb in neighbors:
    rospy.logdebug("{} sub neighbors {}".format(tag, neighbors))
    while subsBelJointfParam[nb].get_num_connections() < 1:
        rospy.logdebug("{} Waiting for nb {} to connect...".format(tag, nb))
        rospy.sleep(1)

jFence = wFence * normalizeArr(phiFence())
while not rospy.is_shutdown():
# for _ in range(20):
    start = timeit.default_timer()
    msgOwnBel = BelParam()
    msgOwnBel.name = name
    msgOwnBel.src = name
    msgOwnBel.exist = True
    msgOwnBel.east = vehicle.location.local_frame.east
    msgOwnBel.north = vehicle.location.local_frame.north
    msgOwnBel.up = -vehicle.location.local_frame.down
    msgOwnBel.std_ned = 2 * resolution
    # msgBelSelfParam.temperature_reading = TODO
    # msgBelSelfParam.humidity_reading = TODO

    if name == "A": msgBelJoint.A = msgOwnBel
    if name == "B": msgBelJoint.B = msgOwnBel
    if name == "C": msgBelJoint.C = msgOwnBel

    np.random.shuffle(neighbors)
    for nb in neighbors:
        msgPropagate = copy.deepcopy(msgBelJoint)
        #  dont relay back the msg thats been heard from the one in the first place
        if msgPropagate.A.src == nb: msgPropagate.A.exist = False
        if msgPropagate.B.src == nb: msgPropagate.B.exist = False
        if msgPropagate.C.src == nb: msgPropagate.C.exist = False
        msgPropagate.A.src = name
        msgPropagate.B.src = name
        msgPropagate.C.src = name
        pubsBelJointfParam[nb].publish(msgPropagate)

    # Intention: Collision avoidance
    liveNbrs = []
    meansList = []
    covsList = []

    for uav in team:
        if uav == name: continue
        if uav == "A":
            if msgBelJoint.A.exist:
                meansList.append(
                    np.array([
                        msgBelJoint.A.east,
                        msgBelJoint.A.north,
                        msgBelJoint.A.up
                    ])
                )
                liveNbrs.append(uav)
                covsList.append(covPos)
        if uav == "B":
            if msgBelJoint.B.exist:
                meansList.append(
                    np.array([
                        msgBelJoint.B.east,
                        msgBelJoint.B.north,
                        msgBelJoint.B.up
                    ])
                )
                liveNbrs.append(uav)
                covsList.append(covPos)
        if uav == "C":
            if msgBelJoint.C.exist:
                meansList.append(
                    np.array([
                        msgBelJoint.C.east,
                        msgBelJoint.C.north,
                        msgBelJoint.C.up
                    ])
                )
                liveNbrs.append(uav)
                covsList.append(covPos)

    jExplore = wExplore * normalizeArr(belExplore)

    if len(liveNbrs) >= 1:
        gmm = GaussianMixture(n_components=len(liveNbrs), covariance_type='full')
        gmm.weights_ = np.ones(len(liveNbrs))
        gmm.means_ = np.array(meansList)
        gmm.precisions_cholesky_ = _compute_precision_cholesky(np.array(covsList), 'full')
        gmLgPdf = gmm.score_samples(X).reshape(xx.shape)

        jCollision = wCollision * normalizeArr(gmLgPdf)
        J = normalizeArr(jCollision + jExplore + jFence)
    else:

        J = normalizeArr(jExplore + jFence)

    loc = locNEU()
    nearestIndX = np.linalg.norm(X - loc, axis=1).argmin()
    indNN = indXtoJ(nearestIndX)

    belExplore[indNN] = 1.

    # Gradient Calculation
    dFx, dFy, dFz = u, v, w = np.gradient(J)

    uvwLoc = np.array([u[indNN], v[indNN], w[indNN]])
    uvwLocNormVel = -(uvwLoc / np.abs(uvwLoc).max())

    if name != "A": send_ned_velocity(uvwNormVel=uvwLocNormVel)

    msgHRI = BelJoint()
    msgHRI.name = name
    msgHRI.src = name
    msgHRI.locNED = loc.flatten()
    msgHRI.J = J.flatten()
    msgHRI.uvwLoc = uvwLoc.flatten()
    pubBelJoint.publish(msgHRI)

    if name != "A": rospy.logdebug("{} {} liveNb {} means {}  uvwLoc={} time={}".format(tag, loc, liveNbrs, meansList, uvwLoc, timeit.default_timer()-start))
    rate.sleep()
