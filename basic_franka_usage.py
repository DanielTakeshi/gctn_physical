"""More movement testing. Remember to avoid reset_pose() and reset_joints().

Purpose of this script:
- Get used to the frankapy interface.
- Move the robot around using `goto_pose` and `goto_joints`. Don't forget to set
    `use_impedance=False`!!
- Figure out a way to get default home poses nicely. We might have to repeat this
    multiple times, especially if it involves re-mounting the hand camera. We can
    also get to a reasonable approximation and fine-tune with rotations.
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


def ee_rotation_tests(fa):
    """Test the z-axis EE rotations.

    We have a mounted camera so we need to be careful not to rotate such that the
    wires hit the robot. Note: the poses are of type `autolab_core.RigidTransform`
    and can give us rotation matrices and quaternions. I think the simplest way to
    do this is to assume that the default pose (with camera mounted to me) is at
    'rotation angle 0' and then we just work from there with offsets.

    Careful: do rotations cause potential issues? Try this again and then do a MWE
    if needed. Wait, sometimes it works, sometimes it doesn't?
    """
    T_ee_world = fa.get_pose()
    joints = fa.get_joints()
    gripper_width = fa.get_gripper_width()

    print('Translation: {} | Rotation: {}'.format(
            T_ee_world.translation, T_ee_world.quaternion))
    print('Joints: {}'.format(joints))
    print('Gripper width: {:0.6f}'.format(gripper_width))

    # First, lower it if needed (I think lowering helps anyway).
    if False:
        print('Translating now with EE pose control.')
        T_ee_world = fa.get_pose()
        T_ee_world.translation += [0., 0.0, -0.0]
        fa.goto_pose(T_ee_world, use_impedance=False)

    # Adjust robot positions.
    #fa.goto_joints(JOINTS_HOME, use_impedance=False)
    #fa.goto_joints(JOINTS_GRIP, use_impedance=False)

    # Test rotations. Unfortunately, seems like -10 is all we can get in one direction.
    # But I think we can go about +145 in the other direction.
    #DU.rotate_EE_one_axis(fa, deg=-10, axis='z')
    DU.rotate_EE_one_axis(fa, deg=-90, axis='z')


def daniel_testing(fa):
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    # Get joint information
    joints = fa.get_joints()
    print('Joints: {}'.format(joints))

    # Get EE information
    gripper_width = fa.get_gripper_width()
    print('Gripper width: {:0.6f}'.format(gripper_width))

    # End-effector pose control (for translation).
    if False:
        print('Translating now with EE pose control.')
        T_ee_world = fa.get_pose()
        T_ee_world.translation += [0., 0.0, -0.05]
        fa.goto_pose(T_ee_world, use_impedance=False)

    # Test EE rotations, by `deg`, then returning back. For all 3 axes.
    if False:
        deg = 15
        print(f'Rotation in end-effector frame by {deg} deg in the x-axis.')
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.x_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
        T_ee_world_target = T_ee_world * T_ee_rot
        fa.goto_pose(T_ee_world_target, use_impedance=False)
        fa.goto_pose(T_ee_world, use_impedance=False)

        deg = 15
        print(f'Rotation in end-effector frame by {deg} deg in the y-axis.')
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
        T_ee_world_target = T_ee_world * T_ee_rot
        fa.goto_pose(T_ee_world_target, use_impedance=False)
        fa.goto_pose(T_ee_world, use_impedance=False)

        deg = 15
        print(f'Rotation in end-effector frame by {deg} deg in the z-axis.')
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.z_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
        T_ee_world_target = T_ee_world * T_ee_rot
        fa.goto_pose(T_ee_world_target, use_impedance=False)
        fa.goto_pose(T_ee_world, use_impedance=False)

    # Try to go to a pre-designated home pose.
    if False:
        print(f'Move to joints: {JOINTS_HOME}')
        fa.goto_joints(JOINTS_HOME, use_impedance=False)


if __name__ == "__main__":
    print('Creating the Franka Arm...')
    fa = FrankaArm()
    print(f'Done creating: {fa}')

    #daniel_testing(fa)
    #ee_rotation_tests(fa)
    print('Finished with tests.')