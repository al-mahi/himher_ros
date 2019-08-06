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
    J = trans(msgHRI.J)
    locNED = np.asanyarray(msgHRI.locNED)
    gp = np.asanyarray(msgHRI.uvwLoc)
    gpnorm = gp/np.abs(gp).max()

    plt.cla()
    plt.xlabel('X')
    plt.ylabel('Y')
    # plt.xlim(minX, maxX)
    # plt.ylim(minY, maxY)
    plt.title("E={:0>6.2f} N={:0>6.2f} Alt={:0>6.2f}\n{}".format(locNED[0], locNED[1], locNED[2], np.around(gpnorm, decimals=4)))

    # rospy.logdebug("{} gp {}".format(tag, gp))
    plt.quiver(locNED[0], locNED[1], -gpnorm[0], -gpnorm[1])
    try:
        p = plt.contour(xx[:, :, 1], yy[:, :, 1], J[:, :, int((atAlt-minZ)/resolution)])
    except IndexError as e:
        rospy.logerr("{} IdexError {} msgHRI uvwLoc={} loc={}".format(tag,e.message, msgHRI.uvwLoc, msgHRI.locNED))
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
atAlt = int(rospy.get_param("~atAlt"))
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
xx, yy, zz = np.meshgrid(x, y, z, indexing='ij')
X = np.array([xx.flatten(), yy.flatten(), zz.flatten()]).T

q_size = 10
rospy.Subscriber(
    "/UAV/{}/bel_joint".format(name[0]), callback=cbBelJoint,
    data_class=BelJoint, queue_size=q_size
)

rate = rospy.Rate(10)
visualization()
while not rospy.is_shutdown():
    rate.sleep()