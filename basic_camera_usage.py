"""
Testing the camera stuff.
"""
import daniel_utils as DU
import time
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
np.set_printoptions(suppress=True, precision=4, linewidth=150)

# ---------------------------------------------------------------------------------- #
# Home joints and pose (default position to get images?).
# Get these by moving the robot (with e-stop on) to a 'reasonable' spot to get images.
# Will require some trial and error. FYI, it seems to be reliable in getting to the
# desired poses, down to about millimeter level precision w.r.t translation, and about
# 0.001 precision (in radians) for the joints (I think all are revolute). Nice...
#
# Possible home:
# Translation: [ 0.333  -0.107   0.6424] | Rotation: [-0.0291  0.029   0.9991 -0.01  ]
# Joints: [-0.5896 -0.4637  0.1914 -1.6885  0.0481  1.1862 -2.7011]
#
# Possible for lowering+testing rotations:
# Translation: [ 0.324  -0.1048  0.2282] | Rotation: [-0.0047  0.0177  0.9998 -0.0077]
# Joints: [-0.4803 -0.503   0.1917 -2.8445  0.104   2.3419 -2.7057]
# ---------------------------------------------------------------------------------- #
EE_HOME = np.array([0.333, -0.107, 0.6424, -0.0291, 0.029, 0.9991, -0.01])
JOINTS_HOME = np.array([-0.5896, -0.4637, 0.1914, -1.6885, 0.0481, 1.1862, -2.7011])

EE_GRIP = np.array([0.324, -0.1048, 0.2282, -0.0047, 0.0177, 0.9998, -0.0077])
JOINTS_GRIP = np.array([-0.4803, -0.503, 0.1917, -2.8445, 0.104, 2.3419, -2.7057])
# ---------------------------------------------------------------------------------- #


def camera_stuff(fa):
    """How do we make use of a camera?

    Do we need two poses, for (base,EE) and (EE,hand)?
    """
    T_ee_world = fa.get_pose()
    joints = fa.get_joints()
    print('Translation: {} | Rotation: {}'.format(
            T_ee_world.translation, T_ee_world.quaternion))
    print('Joints: {}'.format(joints))

    # The calibration file; copy it from `/<HOME>/.ros/easy_handeye`.
    filename = 'cfg/easy_handeye_eye_on_hand.yaml'

    # Load transformation from calibration -- this is the EE to the camera?.

    # Also load transformation from the robot base to the EE? I think we need that?

    # Find pixels we want to use.

    # Then go to the pose?


if __name__ == "__main__":
    print('Creating the Franka Arm...')
    fa = FrankaArm()
    print(f'Done creating: {fa}')

    camera_stuff(fa)
    print('Finished with tests.')