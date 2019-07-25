#!/usr/bin/python
from __future__ import print_function
import numpy as np
from scipy.stats import multivariate_normal
import rospy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import copy
from himher_ros.msg import BelParam, BelJointParam, BelJoint

msgHRI = None  # type: msgHRI BelJoint


def update_all(num, unused):
    trans = lambda XX: np.reshape(np.asanyarray(XX, order='C'), xx.shape)
    global msgHRI  # type: msgHRI BelJoint
    if msgHRI is None: return
    rospy.logdebug("{} uvwLoc {}".format(tag, msgHRI.uvwLoc))
    J = trans(msgHRI.gnorm)
    u = trans(msgHRI.u)
    v = trans(msgHRI.v)
    w = trans(msgHRI.w)
    locNED = np.asanyarray(msgHRI.locNED)
    gp = np.asanyarray(msgHRI.uvwLoc)
    gpnorm = gp/gp.max()

    plt.cla()
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.xlim(minX, maxX)
    plt.ylim(minY, maxY)
    plt.title("E={:0>6.2f} N={:0>6.2f} Alt={:0>6.2f}".format(locNED[0], locNED[1], locNED[2]))

    rospy.logdebug("{} gp {}".format(tag, gp))
    plt.quiver(locNED[0], locNED[1], gpnorm[0], gpnorm[1])
    p = plt.contour(xx[:, :, 1], yy[:, :, 1], J[:, :, int(locNED[2]/resolution)])
    msgHRI = None


def visualization():
    fig = plt.figure("{}".format(name))
    unused = []
    ind = 0
    interval = 5000
    anims = [None]
    anims[ind] = animation.FuncAnimation(
        fig, update_all, 10000, fargs=(unused,),
        interval=interval, blit=False, repeat=False
    )
    plt.show()


def cbBelJoint(msg):
    """:type msg BelJoint"""
    global msgHRI
    msgHRI = copy.deepcopy(msg)


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

rospy.init_node(name="vis", anonymous=True, log_level=rospy.DEBUG)

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
tag = "{}vz: ".format(name)

###############################################################################
# Meshgrid Space
x = np.arange(minX, maxX + resolution, resolution)
y = np.arange(minY, maxY + resolution, resolution)
z = np.arange(minZ, maxZ + resolution, resolution)
xx, yy, zz = np.meshgrid(x, y, z)
X = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

q_size = 10
rospy.Subscriber(
    "/UAV/{}/bel_joint".format(name), callback=cbBelJoint,
    data_class=BelJoint, queue_size=q_size
)

rate = rospy.Rate(1)
visualization()
while not rospy.is_shutdown():
    rate.sleep()