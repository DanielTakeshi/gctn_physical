"""Utilities file to support GCTN experiments.

This is meant to be imported in our `main.py` script which runs experiments.
"""
import cv2
import sys
import numpy as np
from autolab_core import RigidTransform
np.set_printoptions(suppress=True, precision=4, linewidth=150)
import open3d as o3d
import yaml
import daniel_config as DC


def bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def rgb_to_bgr(img):
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def uv_to_world_pos(T_BC, u, v, z, return_meters=False, camera_ns='k4a',
        debug_print=False):
    """Transform from image coordinates and depth to world coordinates.

    Copied this from my 'mixed media' code, but likely have to change.
    https://github.com/DanielTakeshi/mixed-media-physical/blob/63ce1b2118d77f3757452540feeabd733fb9a9f4/utils_robot.py#L106

    TODO(daniel): can we keep track of the units carefully?
    I think the K matrix uses millimeters, and so does the depth camera.
    But does T_BC need to use millimeters or meters?

    Parameters
    ----------
    T_BC: should transform from camera to world. This was the ordering we had it
        earlier, we created a `matrix_camera_to_world`.
    u, v: image coordinates
    z: depth value

    Returns
    -------
    world coordinates at pixels (u,v) and depth z.
    """
    matrix_camera_to_world = T_BC.matrix  # NOTE(daniel): I think units are OK.

    # Get 4x4 camera intrinsics matrix.
    K = DC.K_matrices[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Will this work? Need to check. From SoftGym, and also they flip u,v here...
    one = np.ones(u.shape)
    x = (v - u0) * z / fx
    y = (u - v0) * z / fy

    # If x,y,z came from scalars, makes (1,4) matrix.
    # For others it produces (N,4) data for N data points.
    cam_coords = np.stack([x, y, z, one], axis=1)

    # TODO(daniel): which one? The second one?
    #world_coords = matrix_camera_to_world.dot(cam_coords.T)  # (4,4) x (4,1)
    world_coords = cam_coords.dot(matrix_camera_to_world.T)

    if debug_print:
        print('\n(cam_coords before converting to world)')
        print(cam_coords)

    # Returns (N,4). Ignore the last column.
    if return_meters:
        world_coords /= 1000.0
    return world_coords


def load_transformation(filename, as_rigid_transform=False):
    """Load camera calibration (from Wenxuan)."""
    if '.yaml' in filename:
        calibration = yaml.load(open(filename, 'rb'), Loader=yaml.FullLoader)
        calibration = calibration['transformation']
        trans = np.array([calibration['x'], calibration['y'], calibration['z']])
        quat = np.array([calibration['qw'], calibration['qx'], calibration['qy'], calibration['qz']])
        R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans
        transformation = T
    elif '.npy' in filename:
        with open(filename, 'rb') as f:
            transformation = np.load(f)

    # Might be easier just to do the RigidTransform here?
    if as_rigid_transform:
        T_cam_ee = RigidTransform(
            rotation=transformation[:3, :3],
            translation=transformation[:3, 3],
            from_frame='camera',
            to_frame='franka_tool',
        )
        return T_cam_ee
    else:
        return transformation


def get_rigid_transform_from_7D(nparr):
    """Convenience in case we want to go to 7D poses (translation,quaternion)."""
    assert len(nparr) == 7, nparr
    T_target = RigidTransform(
        translation=nparr[:3],
        rotation=RigidTransform.rotation_from_quaternion(q_wxyz=nparr[3:]),
        from_frame='franka_tool',
        to_frame='world',
    )
    return T_target


def wait_for_enter():
    if sys.version_info[0] < 3:
        raw_input('Enter to continue:')
    else:
        input('Enter to continue, CTRL+C to exit:')


def triplicate(img, to_int=False):
    """Stand-alone `triplicate` method."""
    w,h = img.shape
    new_img = np.zeros([w,h,3])
    for i in range(3):
        new_img[:,:,i] = img
    if to_int:
        new_img = new_img.astype(np.uint8)
    return new_img


def process_depth(orig_img, cutoff=2000):
    """Make a raw depth image human-readable by putting values in [0,255).

    Careful if the cutoff is in meters or millimeters!
    This might depend on the ROS topic. If using:
        rospy.Subscriber('k4a_top/depth_to_rgb/image_raw', ...)
    then it seems like it is in millimeters.
    """
    img = orig_img.copy()

    # Useful to turn the background into black into the depth images.
    def depth_to_3ch(img, cutoff):
        w,h = img.shape
        new_img = np.zeros([w,h,3])
        img = img.flatten()
        img[img>cutoff] = 0.0
        img = img.reshape([w,h])
        for i in range(3):
            new_img[:,:,i] = img
        return new_img

    def depth_scaled_to_255(img):
        # This was being printed constantly during rollouts.
        #if np.max(img) <= 0.0:
        #    print('Warning, np.max: {:0.3f}'.format(np.max(img)))
        img = 255.0/np.max(img)*img
        img = np.array(img,dtype=np.uint8)
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return img

    img = depth_to_3ch(img, cutoff)  # all values above 255 turned to white
    img = depth_scaled_to_255(img)   # correct scaling to be in [0,255) now
    return img


def sample_distribution(prob, n_samples=1):
    """Sample data point from a custom distribution."""
    flat_prob = np.ndarray.flatten(prob) / np.sum(prob)
    rand_ind = np.random.choice(
        np.arange(len(flat_prob)), n_samples, p=flat_prob, replace=False)
    rand_ind_coords = np.array(np.unravel_index(rand_ind, prob.shape)).T
    return np.int32(rand_ind_coords.squeeze())


def rotate_EE_one_axis(fa, deg, axis, use_impedance=True, duration=12):
    """Just do one EE axis rotation.

    Empirically, seems like abs(deg) has to be at least 4 to observe any
    notable robot movement. Also, I'd increase the duration from the
    default of 3.

    We should use impedance True as default.

    Default duration is set to be a bit long just in case.
    """
    assert np.abs(deg) <= 145., f'deg: {deg}, careful that is a lot...'
    print(f'Rotation in EE frame by {deg} deg, axis: {axis}.')
    T_ee_world = fa.get_pose()

    if axis == 'x':
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.x_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
    elif axis == 'y':
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
    elif axis == 'z':
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.z_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
    else:
        raise ValueError(axis)

    T_ee_world_target = T_ee_world * T_ee_rot
    fa.goto_pose(
        T_ee_world_target, duration=duration, use_impedance=use_impedance
    )


def pick_and_place(fa, pick_w, place_w, z_delta, starts_at_top=False):
    """Our pick and place action primitive.

    Take note of the ordering in pix0 and pix1.
    This should also probably determine the rotation angles.

    See daniel_config for details. Reason why I need to do this is that it
    might be more precise if we have shorter length movements to get back to
    the top pose (for which I want to get an accurate and consistent pose).
    Also sometimes I get errors with EE control if I move too much.

    For current (working) values of the delta terms, the angle of rotation, z
    offsets, etc., please see my Notion.

    Parameters
    ----------
    fa: FrankaArm
    pick_w: World coords of picking.
    place_w: World coords of placing.
    z_delta: offset for rotation (this is a hack for us). The 'default' rotation
        is when the camera faces to Eddie's desk, so it's like a '90 deg' rot.
        So, a z_delta=-90 would cause the camera to face my desk but that also
        increases the risks of kinematic errors.
    """
    def get_duration(x_delta, y_delta):
        #p0_norm = np.linalg.norm(np.array([p0_x_delta, p0_y_delta]))
        #p1_norm = np.linalg.norm(np.array([p1_x_delta, p1_y_delta]))
        #p0_dur = int(max(3, p0_norm*20))
        #p1_dur = int(max(3, p1_norm*20))
        #print('(First)  Pull norm: {:0.3f}, p0_dur: {}'.format(p0_norm, p0_dur))
        #print('(Second) Pull norm: {:0.3f}, p1_dur: {}\n'.format(p1_norm, p1_dur))
        # We have to change the above.
        pass

    # TODO(daniel) -- need to fix a lot of the durations. Shouldn't be hard.
    p0_dur = 5
    p1_dur = 5

    # TODO(daniel): check positional bounds.
    # (we should probably add safety checks)

    # Go to top, open/close grippers.
    if not starts_at_top:
        print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
        wait_for_enter()
        fa.goto_joints(DC.JOINTS_TOP, duration=10, ignore_virtual_walls=True)
        fa.goto_gripper(DC.GRIP_OPEN)
        fa.close_gripper()

    # Go to (first) waypoint.
    print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
    fa.goto_joints(DC.JOINTS_WP1, duration=5)

    # Go to (second) waypoint, use a longer duration.
    print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
    fa.goto_joints(DC.JOINTS_WP2, duration=10)

    # Rotate about z axis.
    print(f'\nRotate by z delta: {z_delta}')
    rotate_EE_one_axis(fa, deg=z_delta, axis='z', use_impedance=True, duration=5)

    # Translate to be above the picking point.
    prepick_w = np.copy(pick_w)
    prepick_w[2] = DC.Z_PRE_PICK
    T_ee_world = fa.get_pose()
    T_ee_world.translation = prepick_w
    print(f'\nTranslate to be above picking point, press ENTER:\n{T_ee_world}')
    fa.goto_pose(T_ee_world, duration=p0_dur)

    # Open the gripper.
    fa.goto_gripper(width=DC.GRIP_OPEN)

    # Lower to actually grasp.
    picking_w = np.copy(pick_w)
    picking_w[2] = DC.Z_PICK
    T_ee_world.translation = picking_w
    print(f'\nLower to grasp: {T_ee_world}')
    fa.goto_pose(T_ee_world)

    # Close the gripper. Careful! Need `grasp=True`.
    fa.goto_gripper(width=DC.GRIP_CLOSE, grasp=True, force=10.)

    # Return to pre-pick. NOTE(daniel): maybe due to extra weight, I had to
    # add some more height to make it raise sufficiently.
    prepick_w_2 = np.copy(pick_w)
    prepick_w_2[2] = DC.Z_PRE_PICK2
    T_ee_world.translation = prepick_w_2
    print(f'\nBack to pre-pick: {T_ee_world}')
    fa.goto_pose(T_ee_world)

    # Go to the pre-placing point.
    # NOTE(daniel): should be based on the image/action, using fake values.
    preplace_w = np.copy(place_w)
    preplace_w[2] = DC.Z_PRE_PLACE
    T_ee_world.translation = preplace_w
    print(f'\nTranslate to be above PLACING point: {T_ee_world}')
    fa.goto_pose(T_ee_world, duration=p1_dur)

    # Lower to place gently.
    placing_w = np.copy(place_w)
    placing_w[2] = DC.Z_PLACE
    T_ee_world.translation = placing_w
    print(f'\nLower to place: {T_ee_world}')
    fa.goto_pose(T_ee_world)

    # Open the gripper.
    print('\nOpening gripper (should release cable)...')
    fa.goto_gripper(width=DC.GRIP_RELEASE)

    # Return to pre-placing point. NOTE(daniel): if remove lowering, remove this.
    T_ee_world.translation = preplace_w
    print(f'\nReturn to pre-placing: {T_ee_world}')
    fa.goto_pose(T_ee_world)

    # Return to (second) waypoint. This should revert the rotation.
    print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
    fa.goto_joints(DC.JOINTS_WP2, duration=6)

    # Return to (first) waypoint, use a longer duration.
    print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
    fa.goto_joints(DC.JOINTS_WP1, duration=10)

    # Return to top.
    print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
    fa.goto_joints(DC.JOINTS_TOP, duration=5, ignore_virtual_walls=True)

    return fa


if __name__ == "__main__":
    pass
