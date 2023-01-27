import numpy as np

# ---------------------------------------------------------------------------------- #
# Home joints and pose (default position to get images?). Use `finetune_pose()` method
# in `basic_franka_usage.py`
# ---------------------------------------------------------------------------------- #
# For taking images? Top-down view.
# Translation: [ 0.3744 -0.1146  0.7233] | Rotation: [-0.0074  0.029   0.9993 -0.0218]
# Joints: [-2.8367  0.2063  2.7082 -1.1855 -0.1566  0.9927 -2.5034]
#
# For lowering before rotations? This should also be a waypoint we reach after finishing
# an action since we don't want to risk hitting anything.
# Translation: [ 0.324  -0.1048  0.2282] | Rotation: [-0.0047  0.0177  0.9998 -0.0077]
# Joints: [-0.4803 -0.503   0.1917 -2.8445  0.104   2.3419 -2.7057]
# ---------------------------------------------------------------------------------- #
EE_HOME = np.array([0.3744, -0.1146, 0.7233, -0.0074, 0.029, 0.9993, -0.0218])
JOINTS_HOME = np.array([-2.8367, 0.2063, 2.7082, -1.1855, -0.1566, 0.9927, -2.5034])

EE_GRIP = np.array([0.324, -0.1048, 0.2282, -0.0047, 0.0177, 0.9998, -0.0077])
JOINTS_GRIP = np.array([-0.4803, -0.503, 0.1917, -2.8445, 0.104, 2.3419, -2.7057])
# ---------------------------------------------------------------------------------- #

# ---------------------------------------------------------------------------------- #
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
# ---------------------------------------------------------------------------------- #

