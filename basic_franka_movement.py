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
#Translation: [ 0.333  -0.107   0.6424] | Rotation: [-0.0291  0.029   0.9991 -0.01  ]
#Joints: [-0.5896 -0.4637  0.1914 -1.6885  0.0481  1.1862 -2.7011]
# ---------------------------------------------------------------------------------- #
JOINTS_HOME = np.array([-0.5896, -0.4637, 0.1914, -1.6885, 0.0481, 1.1862, -2.7011])
EE_HOME = np.array([0.333, -0.107, 0.6424, -0.0291, 0.029, 0.9991, -0.01])
# ---------------------------------------------------------------------------------- #


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

    daniel_testing(fa)
    print('Finished with tests.')