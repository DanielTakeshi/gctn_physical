"""Test our pick and place action primitive.

Note that we really will put this in `daniel_utils` but put in this stand
alone script until we get it working.
"""
import daniel_config as DC
import daniel_utils as DU
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
np.set_printoptions(suppress=True, precision=4, linewidth=150)


def get_z_rotation_delta(img=None):
    """Compute a z rotation, in degrees.

    We assume the waypoints (WP2) has the 'default' rotation and we compute
    an offset from that conditioned on what the mask image shows at the pick
    point. If we don't have an image, just put a fake (but valid!) rotation.
    """
    if img is None:
        return -70
    raise NotImplementedError()


def test_pick_and_place(z_offset=0.050):
    """Go to a sequence of waypoints to test intermediate grasp poses.

    See daniel_config for details. Reason why I need to do this is that it
    might be more precise if we have shorter length movements to get back to
    the top pose (for which I want to get an accurate and consistent pose).
    Also sometimes I get errors with EE control if I move too much.

    But for moving joints, I get this error:
    ValueError: Target joints in collision with virtual walls!
    """
    assert z_offset > 0, z_offset

    # Be careful about how we compute these from our actions!
    p0_x_delta =  0.200
    p0_y_delta = -0.150
    p1_x_delta =  0.000
    p1_y_delta =  0.350
    p0_norm = np.linalg.norm(np.array([p0_x_delta, p0_y_delta]))
    p1_norm = np.linalg.norm(np.array([p1_x_delta, p1_y_delta]))
    p0_dur = int(max(3, p0_norm*20))
    p1_dur = int(max(3, p1_norm*20))
    print('(First)  Pull norm: {:0.3f}, p0_dur: {}'.format(p0_norm, p0_dur))
    print('(Second) Pull norm: {:0.3f}, p1_dur: {}\n'.format(p1_norm, p1_dur))

    fa = FrankaArm()
    print(f'Done creating: {fa}')
    T_ee_world = fa.get_pose()
    joints = fa.get_joints()
    print('Starting Translation: {} | Rotation: {}'.format(
            T_ee_world.translation, T_ee_world.quaternion))
    print('Starting Joints: {}'.format(joints))

    # Go to top. NOTE(daniel): duration should be distance-dependent!!
    print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
    DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_TOP, duration=5, ignore_virtual_walls=True)

    # Go to (first) waypoint.
    print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
    DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP1, duration=5)

    # Go to (second) waypoint, use a longer duration.
    print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
    DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP2, duration=10)

    # Rotate.
    # NOTE(daniel): should be based on the image/action, using fake values.
    z_delta = get_z_rotation_delta()
    print(f'\nRotate by z delta: {z_delta}')
    DU.wait_for_enter()
    DU.rotate_EE_one_axis(fa, deg=z_delta, axis='z', use_impedance=True, duration=5)

    # Translate to be above the picking point.
    # NOTE(daniel): should be based on the image/action, using fake values.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [p0_x_delta, p0_y_delta, 0.0]
    print(f'\nTranslate to be above picking point: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world, duration=p0_dur)

    # Open the gripper.
    #fa.open_gripper()

    # Lower to actually grasp.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.0, 0.0, -z_offset]
    print(f'\nLower to grasp: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world)

    # Close the gripper.
    #fa.close_gripper()

    # Raise by a z offset.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.0, 0.0, z_offset]
    print(f'\nRaise by z offset: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world)

    # Go to the placing point.
    # NOTE(daniel): should be based on the image/action, using fake values.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [p1_x_delta, p1_y_delta, 0.0]
    print(f'\nTranslate to be above PLACING point: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world, duration=p1_dur)

    # Lower by z.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.0, 0.0, -z_offset]
    print(f'\nLower by z offset: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world)

    # Open the gripper.
    #fa.open_gripper()

    # Raise by z offset again.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.0, 0.0, z_offset]
    print(f'\nRaise by z offset: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world)

    # Return to (second) waypoint. This should revert the rotation.
    print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
    DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP2, duration=5)

    # Return to (first) waypoint, use a longer duration.
    print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
    DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP1, duration=10)

    # Return to top.
    print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
    DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_TOP, duration=5, ignore_virtual_walls=True)

    return fa


if __name__ == "__main__":
    fa = test_pick_and_place()

    # The pose sometimes seems inconsistent, unfortunately.
    print('\nFinished with tests.')
    T_ee_world = fa.get_pose()
    joints = fa.get_joints()
    print('FINAL Translation: {} | Rotation: {}'.format(
            T_ee_world.translation, T_ee_world.quaternion))
    print('FINAL Joints: {}'.format(joints))
