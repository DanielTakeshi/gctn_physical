import numpy as np
from frankapy.franka_constants import FrankaConstants as FC

# -------------------------------------------------------------------------------- #
# Home joints and pose (default position to get images?). Use the `finetune_pose()`
# in `basic_franka_usage.py`. Let's define TOP as the top-most position when we get
# the images. Then WP1 is slightly lower, then WP2 is lower still, then we will do
# pick and place. Then, return to WP2, then WP1, then HOME.
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

# CAREFUL: all poses must be valid with the mounted camera and don't damage cables.

# Basically moving the joints around. Draft. Last joint can be -3*pi/4 = -2.3562 if
# we judge the reset pose to be at pi/4 (we want to flip 180 degrees).
#JOINTS_TOP = np.array([-0.1740, -0.3637, 0.0106, -1.4497, -0.0062, 1.0929, -2.3562])

# 01/30. Fine-tuning to try and make the setup reachable for all poses.
#JOINTS_TOP = np.array([-0.5314, -0.3901, 0.4439, -1.6139, 0.1128, 1.2335, -2.4047])
JOINTS_TOP = np.array([-0.5327, -0.3815, 0.5206, -1.6181, 0.1410, 1.2335, -2.3280])

# Got WP1 from the reset_joints() BUT adding pi/2 to EE (-pi/4 to pi/4).
JOINTS_WP1 = np.array([-0.0003, -0.7851, 0.0001, -2.3564, -0.0004, 1.5705, -0.7851])

# Got WP2 from manually lowering and getting a reasonable set, w/same last joint.
JOINTS_WP2 = np.array([-0.2623,  0.0315, 0.1403, -2.6996, -0.0661, 2.7208, -0.7851])

# Older values -- these were hitting the limits.
#JOINTS_TOP = np.array([-2.8367, 0.2063, 2.7082, -1.1855, -0.1566, 0.9927, -2.5034])
#JOINTS_WP1 = np.array([-2.8866, 0.4845, 2.8205, -1.6213, -0.2449, 1.1501, -2.4354])
#JOINTS_WP2 = np.array([-2.8901, 0.1476, 2.6970, -2.7584, -0.2457, 2.6129, -2.0506])
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

