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


def uv_to_world_pos(T_BC, u, v, z, camera_ns='k4a', debug_print=False):
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

    # If x,y,z came from scalars, makes (1,4) matrix. Need to test for others.
    cam_coords = np.stack([x, y, z, one], axis=1)

    # TODO(daniel): which one? The second one?
    #world_coords = matrix_camera_to_world.dot(cam_coords.T)  # (4,4) x (4,1)
    world_coords = cam_coords.dot(matrix_camera_to_world.T)

    if debug_print:
        # Camera axis has +x pointing to me, +y to wall, +z downwards.
        #print('\nMatrix camera to world')
        #print(matrix_camera_to_world)
        #print('\n(inverse of that matrix)')
        #print(np.linalg.inv(matrix_camera_to_world))
        print('\n(cam_coords before converting to world)')
        print(cam_coords)
        print('')

    return world_coords  # (n,4) but ignore last row


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
        raw_input('Press Enter to continue:')
    else:
        input('Press Enter to continue:')


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
        if np.max(img) <= 0.0:
            print('Warning, np.max: {:0.3f}'.format(np.max(img)))
        img = 255.0/np.max(img)*img
        img = np.array(img,dtype=np.uint8)
        for i in range(3):
            img[:,:,i] = cv2.equalizeHist(img[:,:,i])
        return img

    img = depth_to_3ch(img, cutoff)  # all values above 255 turned to white
    img = depth_scaled_to_255(img)   # correct scaling to be in [0,255) now
    return img


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


def pick_only(fa, pix0, img_c, z_delta):
    """A pick primitive to test things.

    Take note of the ordering in pix0 and pix1.
    This should also probably determine the rotation angles.

    fa: FrankaArm
    pix0: tuple of picking pixels.
    img_c: current image (assume a binary mask).
    z_delta: delta of height for picking and placing.
    """
    assert z_delta > 0, z_delta

    raise NotImplementedError


def pick_and_place(fa, pix0, pix1, img_c, img_g, z_delta):
    """Our pick and place action primitive.

    Take note of the ordering in pix0 and pix1.
    This should also probably determine the rotation angles.

    fa: FrankaArm
    pix0: tuple of picking pixels.
    pix1: tuple of placing pixels.
    img_c: current image (assume a binary mask).
    img_g: goal image (assume a binary mask).
    z_delta: delta of height for picking and placing.
    """
    assert z_delta > 0, z_delta

    raise NotImplementedError


if __name__ == "__main__":
    pass
