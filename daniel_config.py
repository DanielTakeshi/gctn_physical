import numpy as np
from frankapy.franka_constants import FrankaConstants as FC

# -------------------------------------------------------------------------------- #
# Home joints and pose (default position to get images?). Use the `finetune_pose()`
# in `basic_franka_usage.py`. Let's define TOP as the top-most position when we get
# the images. Then WP1 is slightly lower, then WP2 is lower still, then we will do
# pick and place. Then, return to WP2, then WP1, then HOME.
# -------------------------------------------------------------------------------- #
JOINTS_RESET = np.copy(np.array(FC.HOME_JOINTS))
JOINTS_RESET[6] -= np.deg2rad(90)  # takes into account the camera

EE_TOP = np.array([0.3744, -0.1146, 0.7233, -0.0074, 0.0290, 0.9993, -0.0218])
EE_WP1 = np.array([0.3314, -0.1157, 0.6654, -0.0097, 0.0250, 0.9989, -0.0378])
EE_WP2 = np.array([0.3813, -0.0687, 0.1529, -0.0009, 0.1402, 0.9897, -0.0298])

# Note: See https://frankaemika.github.io/docs/control_parameters.html
# JOINT_LIMITS_MIN = [-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973]
# JOINT_LIMITS_MAX = [ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973]

JOINTS_TOP = np.array([-2.8367, 0.2063, 2.7082, -1.1855, -0.1566, 0.9927, -2.5034])
JOINTS_WP1 = np.array([-2.8866, 0.4845, 2.8205, -1.6213, -0.2449, 1.1501, -2.4354])
JOINTS_WP2 = np.array([-2.8901, 0.1476, 2.6970, -2.7584, -0.2457, 2.6129, -2.0506])
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

