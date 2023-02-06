import numpy as np
from frankapy.franka_constants import FrankaConstants as FC

# -------------------------------------------------------------------------------- #
# Home joints and pose (default position to get images?). Use the `finetune_pose()`
# in `basic_franka_usage.py`. Let's define TOP as the top-most position when we get
# the images. Then WP1 is slightly lower, then WP2 is lower still, then we will do
# pick and place. Then, return to WP2, then WP1, then HOME.
# CAREFUL: all poses must be valid with the mounted camera w/out damaging cables!!
# -------------------------------------------------------------------------------- #
# `FC.HOME_JOINTS`: [-0.0003 -0.7851  0.0001 -2.3564 -0.0004  1.5705  0.7851]
# where the last value is pi/4, but CAREFUL the camera is in the way.
# -------------------------------------------------------------------------------- #
# Note: See https://frankaemika.github.io/docs/control_parameters.html
# JOINT_LIMITS_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
# JOINT_LIMITS_MAX = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]
# -------------------------------------------------------------------------------- #
JOINTS_RESET = np.copy(np.array(FC.HOME_JOINTS))
JOINTS_RESET[6] -= np.deg2rad(90)  # takes into account the camera

# --------------------------- TOP --------------------------- #
# Basically moving the joints around. Draft. Last joint can be -3*pi/4 = -2.3562 if
# we judge the reset pose to be at pi/4 (we want to flip 180 degrees).
#JOINTS_TOP = np.array([-0.1740, -0.3637, 0.0106, -1.4497, -0.0062, 1.0929, -2.3562])

# Fine-tuning to try and make the setup reachable for all poses.
#JOINTS_TOP = np.array([-0.5314, -0.3901, 0.4439, -1.6139, 0.1128, 1.2335, -2.4047])
#JOINTS_TOP = np.array([-0.5327, -0.3815, 0.5206, -1.6181, 0.1410, 1.2335, -2.3280])  # 01/30
JOINTS_TOP = np.array([-0.5337, -0.3563, 0.5253, -1.6232, 0.1394, 1.2847, -2.3275])  # 02/05

# --------------------------- WP1 --------------------------- #
# Got WP1 from the reset_joints() BUT adding pi/2 to EE (-pi/4 to pi/4).
JOINTS_WP1 = np.array([-0.0003, -0.7851, 0.0001, -2.3564, -0.0004, 1.5705, -0.7851])

# --------------------------- WP2 --------------------------- #
# Got WP2 from manually lowering and getting a reasonable set, w/same last joint.
# Not bad, I think this might actually be OK.
#JOINTS_WP2 = np.array([-0.2623, 0.0315, 0.1403, -2.6996, -0.0661, 2.7208, -0.7851])

# 01/30, try to move it down lower, tilt camera up a bit so it doesn't interfere?
JOINTS_WP2 = np.array([0.4127, 0.2938, -0.2986, -2.4853, 0.0676, 2.7421, -0.7049])
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# Other parameters for Transporters experiments.
# -------------------------------------------------------------------------------- #

# The closing doesn't quite matter that much as the frankapy should stop excess force.
GRIP_OPEN    = 0.040
GRIP_CLOSE   = 0.010
GRIP_RELEASE = 0.030  # smaller than GRIP_OPEN to avoid affecting cables too much

# The z height BEFORE and then DURING a pick.
Z_PRE_PICK  = 0.070
Z_PRE_PICK2 = 0.080  # 02/05: lifting w/cable seems to require a little more
Z_PICK      = 0.025  # 02/06, seems like 0.030 is too high
Z_PRE_PLACE = 0.070
Z_PLACE     = 0.035

# From Feb 05: test_pick_and_place.py
LIM_X_LO =  0.31
LIM_X_HI =  0.63
LIM_Y_LO = -0.31
LIM_Y_HI =  0.33
# -------------------------------------------------------------------------------- #

# -------------------------------------------------------------------------------- #
# https://github.com/DanielTakeshi/mixed-media-physical/blob/main/config.py
# Get from `rostopic echo /k4a/rgb/camera_info`. These are the _intrinsics_,
# which are just given to us (we don't need calibration for these). We do need
# a separate roslaunch command, and that directly affects these values.
#
# fx  0 x0
#  0 fy y0
#  0  0  1

K_matrices = {
    # From the Sawyer (scooping).
    #'k4a': np.array([
    #    [977.870,     0.0, 1022.401],
    #    [    0.0, 977.865,  780.697],
    #    [    0.0,     0.0,      1.0]
    #]),
    # From the Franka setup (iam-dopey).
    'k4a': np.array([
        [609.096,     0.0, 639.614],
        [    0.0, 608.888, 365.639],
        [    0.0,     0.0,     1.0]
    ]),
}
# -------------------------------------------------------------------------------- #

