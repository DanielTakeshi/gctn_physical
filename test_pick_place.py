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

    As of 01/30, the 'default' rotation (at WP2) has the camera pointing to the
    left, so we probably want some negative rotation deltas.
    """
    if img is None:
        return -30
    raise NotImplementedError()


def test_pick_and_place(
        z_off_1=0.030,
        z_off_2=0.070,
        open_close_gripper=True
    ):
    """Go to a sequence of waypoints to test intermediate grasp poses.

    See daniel_config for details. Reason why I need to do this is that it
    might be more precise if we have shorter length movements to get back to
    the top pose (for which I want to get an accurate and consistent pose).
    Also sometimes I get errors with EE control if I move too much.

    For current (working) values of the delta terms, the angle of rotation, z
    offsets, etc., please see my Notion.

    Parameters
    ----------
    z_off_1: offset for lowering (after reaching the pick point and opening).
    z_off_2: offset for raising after grasping (empirically we need this, likely
        due to impedance control and extra cable weight).
    open_close_gripper: whether to test with opening and closing grippers.
    """
    assert 0 < z_off_1 <= 0.100, z_off_1
    assert 0 < z_off_2 <= 0.100, z_off_2

    # Be careful about how we compute these from our actions!
    p0_x_delta =  0.190
    p0_y_delta =  0.320
    p1_x_delta = -0.360
    p1_y_delta =  0.000
    p0_norm = np.linalg.norm(np.array([p0_x_delta, p0_y_delta]))
    p1_norm = np.linalg.norm(np.array([p1_x_delta, p1_y_delta]))
    p0_dur = int(max(3, p0_norm*20))
    p1_dur = int(max(3, p1_norm*20))
    print('(First)  Pull norm: {:0.3f}, p0_dur: {}'.format(p0_norm, p0_dur))
    print('(Second) Pull norm: {:0.3f}, p1_dur: {}\n'.format(p1_norm, p1_dur))

    # Create the robot and get initial information.
    fa = FrankaArm()
    print(f'Done creating: {fa}, closing gripper...')
    fa.goto_gripper(DC.GRIP_OPEN)
    fa.close_gripper()
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
    #DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP1, duration=5)

    # Go to (second) waypoint, use a longer duration.
    print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
    #DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP2, duration=10)

    # Rotate.
    # NOTE(daniel): should be based on the image/action, using fake values.
    z_delta = get_z_rotation_delta()
    print(f'\nRotate by z delta: {z_delta}')
    #DU.wait_for_enter()
    DU.rotate_EE_one_axis(fa, deg=z_delta, axis='z', use_impedance=True, duration=5)

    # Translate to be above the picking point.
    # NOTE(daniel): should be based on the image/action, using fake values.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [p0_x_delta, p0_y_delta, 0.0]
    print(f'\nTranslate to be above picking point: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world, duration=p0_dur)

    # Open the gripper.
    if open_close_gripper:
        fa.goto_gripper(width=DC.GRIP_OPEN)

    # Lower to actually grasp.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.0, 0.0, -z_off_1]
    print(f'\nLower to grasp: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world)

    # Close the gripper. Careful! Need `grasp=True`.
    if open_close_gripper:
        fa.goto_gripper(width=DC.GRIP_CLOSE, grasp=True, force=10.)

    # Raise by a z offset. NOTE(daniel): after tests, seems like it was not
    # raising enough, maybe due to extra weight from grasping the cable?
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [0.0, 0.0, z_off_2]
    print(f'\nRaise by z offset: {T_ee_world}')
    #DU.wait_for_enter()
    fa.goto_pose(T_ee_world)

    # Go to the placing point.
    # NOTE(daniel): should be based on the image/action, using fake values.
    T_ee_world = fa.get_pose()
    T_ee_world.translation += [p1_x_delta, p1_y_delta, 0.0]
    print(f'\nTranslate to be above PLACING point: {T_ee_world}')
    DU.wait_for_enter()
    fa.goto_pose(T_ee_world, duration=p1_dur)

    # Lower by z. NOTE(daniel): from testing, do we actually want this?
    if False:
        T_ee_world = fa.get_pose()
        T_ee_world.translation += [0.0, 0.0, -z_off_1]
        print(f'\nLower by z offset: {T_ee_world}')
        DU.wait_for_enter()
        fa.goto_pose(T_ee_world)

    # Open the gripper.
    if open_close_gripper:
        print('\nOpening gripper (should release cable)...')
        fa.goto_gripper(width=DC.GRIP_OPEN)

    # Raise by z offset. NOTE(daniel): if remove lowering, then remove this.
    if False:
        T_ee_world = fa.get_pose()
        T_ee_world.translation += [0.0, 0.0, z_off_1]
        print(f'\nRaise by z offset: {T_ee_world}')
        DU.wait_for_enter()
        fa.goto_pose(T_ee_world)

    # Return to (second) waypoint. This should revert the rotation.
    print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
    #DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP2, duration=8)

    # Return to (first) waypoint, use a longer duration.
    print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
    #DU.wait_for_enter()
    fa.goto_joints(DC.JOINTS_WP1, duration=10)

    # Return to top.
    print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
    #DU.wait_for_enter()
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
