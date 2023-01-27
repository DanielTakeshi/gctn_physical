"""More movement testing. Remember to avoid reset_pose() and reset_joints().

Purpose of this script:
- Get used to the frankapy interface.
- Move the robot around using `goto_pose` and `goto_joints`. Don't forget to set
    `use_impedance=False`!!
- Figure out a way to get default home poses nicely. We might have to repeat this
    multiple times, especially if it involves re-mounting the hand camera. We can
    also get to a reasonable approximation and fine-tune with rotations.
"""
import daniel_config as DC
import daniel_utils as DU
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
np.set_printoptions(suppress=True, precision=4, linewidth=150)


def ee_rotation_tests(fa):
    """Test the z-axis EE rotations.

    We have a mounted camera so we need to be careful not to rotate such that the
    wires hit the robot. Note: the poses are of type `autolab_core.RigidTransform`
    and can give us rotation matrices and quaternions. I think the simplest way to
    do this is to assume that the default pose (with camera mounted to me) is at
    'rotation angle 0' and then we just work from there with offsets.

    Unfortunately the way the camera is mounted means when we rotate the gripper, we
    are close to one of the extremes of the rotation of the EE.

    Careful: do rotations cause potential issues? Try this again and then do a MWE
    if needed. Sometimes it works, sometimes it doesn't. Tip from the iam-lab: try
    increasing the duration of the command.
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

    # Adjust robot positions. Note: careful, this WILL rotate the EE to default.
    #fa.goto_joints(JOINTS_HOME, use_impedance=False)
    #fa.goto_joints(JOINTS_GRIP, duration=20, use_impedance=False)

    # Test rotations. Unfortunately, seems like -10 is all we can get in one direction.
    # But I think we can go about +145 in the other direction.
    #DU.rotate_EE_one_axis(fa, deg=-10, axis='z')
    DU.rotate_EE_one_axis(fa, deg=5, axis='z', duration=10)


def daniel_testing(fa):
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    # # Get joint information
    # joints = fa.get_joints()
    # print('Joints: {}'.format(joints))

    # # Get EE information
    # gripper_width = fa.get_gripper_width()
    # print('Gripper width: {:0.6f}'.format(gripper_width))

    # End-effector pose control (for translation).
    if False:
        print('Translating now with EE pose control.')
        T_ee_world = fa.get_pose()
        T_ee_world.translation -= [0., 0.0, 0.10]
        fa.goto_pose(T_ee_world, use_impedance=True)
        T_ee_world = fa.get_pose()
        print('Translation: {} | Rotation: {}'.format(
                T_ee_world.translation, T_ee_world.quaternion))

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


def finetune_pose(fa):
    """Get a good starting pose for experiments.

    Just keep tweaking this? As of 01/27 that's what I'm doing.
    Tip: have k4aviewer open while running this. :)
    """
    T_ee_world = fa.get_pose()
    joints = fa.get_joints()
    print('Translation: {} | Rotation: {}'.format(
            T_ee_world.translation, T_ee_world.quaternion))
    print('Joints: {}'.format(joints))

    # Tweak the robot as needed, or comment out to get the final pose.

    #DU.rotate_EE_one_axis(fa, deg=-2, axis='x', duration=10)
    #DU.rotate_EE_one_axis(fa, deg=-4, axis='y', duration=10)
    #DU.rotate_EE_one_axis(fa, deg=-3, axis='z', duration=10)

    #T_ee_world.translation += [0., -0.015, -0.0]
    #T_ee_world.translation += [0., -0.0, -0.010]
    #T_ee_world.translation += [-0.015, -0.0, -0.0]
    #fa.goto_pose(T_ee_world, use_impedance=True)
    return


def goto_start_pose(fa):
    """Go to the starting pose obtained from `finetune_pose()`."""
    T_ee_world = fa.get_pose()
    joints = fa.get_joints()
    print('Translation: {} | Rotation: {}'.format(
            T_ee_world.translation, T_ee_world.quaternion))
    print('Joints: {}'.format(joints))

    # Go to a pre-designated home pose. I think this should work.
    T_ee_world.translation = DC.EE_HOME[:3]
    T_ee_world.quaternion = DC.EE_HOME[3:]
    print(f'Move to joints: {DC.EE_HOME}')
    fa.goto_pose(T_ee_world, use_impedance=True, duration=10)


if __name__ == "__main__":
    print('Creating the Franka Arm...')
    fa = FrankaArm()
    print(f'Done creating: {fa}')

    #daniel_testing(fa)
    #ee_rotation_tests(fa)

    # We could iteratively fine-tune this to get good starting pose.
    finetune_pose(fa)

    print('Finished with tests.')