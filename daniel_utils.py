"""Utilities file to support GCTN experiments.

This is meant to be imported in our `main.py` script which runs experiments.
"""
import cv2
import numpy as np
from autolab_core import RigidTransform
np.set_printoptions(suppress=True, precision=4, linewidth=150)


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
