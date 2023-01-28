"""
Test joint limits. Might be helpful for us as we have a camera which
poses some problems with the last joint (index 6) which is the EE.
I found some interesting findings from this, see comment below about
how to properly apply a rotation to the last joint. I think for our
case with the camera, we should start with FC.HOME_JOINTS and then
apply joints[6] += np.deg2rad(90) to get the 'reset' pose.
"""
import numpy as np
from frankapy import FrankaArm
from frankapy.franka_constants import FrankaConstants as FC
from autolab_core import RigidTransform
np.set_printoptions(precision=4, linewidth=150, suppress=True)
print('Joints: {} -- [from constants]'.format(np.array(FC.HOME_JOINTS)))

# Init robot and get starting joints.
fa = FrankaArm()
joints = fa.get_joints()
print('Joints: {} -- [starting joints]'.format(joints))

# Call reset_joints().
fa.reset_joints()
joints = fa.get_joints()
print('Joints: {} -- [after fa.reset_joints()]'.format(joints))

# ---------------------------------------------------------------------------- #
# PROBLEM: calling `fa.get_joints()` returns two possible joint values. The one
# that isn't equal to FC.HOME_JOINTS returns a more extrene `joints[6]`. Thus
# doing a rotation on that will cause excessive movement. So I think we need to
# stick with modifying a copy of FC.HOME_JOINTS.
# ---------------------------------------------------------------------------- #
# UPDATE: well actually restarting my control PC seems to work. But I think it
# would be safer just to start from FC.HOME_JOINTS in any case!
# ---------------------------------------------------------------------------- #

# Don't do this: `joints[6] -= np.deg2rad(15)`. Instead:
joints = np.copy(np.array(FC.HOME_JOINTS))
joints[6] -= np.deg2rad(90)
print('Joints: {} -- [after modifying joints array]'.format(joints))
fa.goto_joints(joints)

# Now let's read the joint values.
joints = fa.get_joints()
print('Joints: {} -- [after rotating]'.format(joints))
