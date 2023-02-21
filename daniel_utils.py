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
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)


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
    print(f'Rotation in EE frame by {deg:0.2f} deg, axis: {axis}.')
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


def pick_and_place(fa, pix0, pix1, pick_w, place_w, z_delta=0.0,
        starts_at_top=False):
    """Our pick and place action primitive.

    Take note of the ordering in pix0 and pix1.
    This should also probably determine the rotation angles.

    See daniel_config for details. Reason why I need to do this is that it
    might be more precise if we have shorter length movements to get back to
    the top pose (for which I want to get an accurate and consistent pose).
    Also sometimes I get errors with EE control if I move too much.

    For current (working) values of the delta terms, the angle of rotation, z
    offsets, etc., please see my Notion.

    02/20/2023: let's adjust z offset based on how far the robot moves. I
    think we should reduce the offset with smaller z.

    Parameters
    ----------
    fa: FrankaArm
    pick_w: World coords of picking.
    place_w: World coords of placing. If this is None, we exit. Actually maybe
        let's not do this complexity.
    z_delta: offset for rotation (this is a hack for us). The 'default' rotation
        is when the camera faces to Eddie's desk, so it's like a '90 deg' rot.
        So, a z_delta=-90 would cause the camera to face my desk but that also
        increases the risks of kinematic errors.
    """
    def get_duration(curr_xy, targ_xy):
        norm_xy = np.linalg.norm(curr_xy - targ_xy)
        p_dur = int(max(3, norm_xy*20))
        return p_dur

    # TODO(daniel): check positional bounds.
    # (we should probably add safety checks)

    pixel_diff = np.linalg.norm(pix0 - pix1)
    print(f'Pixel difference norm: {pixel_diff:0.3f}')

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
    print(f'\nRotate by z delta: {z_delta:0.2f}')
    rotate_EE_one_axis(fa, deg=z_delta, axis='z', use_impedance=True, duration=5)

    # Translate to be above the picking point, now w/calibration correction.
    prepick_w = np.copy(pick_w)
    prepick_w[2] = DC.Z_PRE_PICK
    prepick_w = DC.calibration_correction(pix_w=pix0, pick_w=prepick_w)
    T_ee_world = fa.get_pose()
    curr_xy = T_ee_world.translation[:2]
    targ_xy = prepick_w[:2]
    p0_dur = get_duration(curr_xy, targ_xy)
    T_ee_world.translation = prepick_w
    print(f'\nTranslate to pre-pick, dur. {p0_dur}.')
    fa.goto_pose(T_ee_world, duration=p0_dur)

    # Open the gripper.
    fa.goto_gripper(width=DC.GRIP_OPEN)

    # Lower to actually grasp. We want to copy prepick_w and adjust z axis.
    picking_w = np.copy(prepick_w)
    #picking_w[2] = DC.Z_PICK
    picking_w[2] += (DC.Z_PICK - DC.Z_PRE_PICK)
    T_ee_world.translation = picking_w
    print(f'\nLower to grasp.')
    fa.goto_pose(T_ee_world)

    # Close the gripper. Careful! Need `grasp=True`.
    fa.goto_gripper(width=DC.GRIP_CLOSE, grasp=True, force=10.)

    # Return to pre-pick. NOTE(daniel): maybe due to extra weight, I had to
    # add some more height to make it raise sufficiently. Also this doesn't
    # need calibration correction and I think copying pick_w is OK.
    prepick_w_2 = np.copy(pick_w)
    prepick_w_2[2] = DC.Z_PRE_PICK2
    T_ee_world.translation = prepick_w_2
    print(f'\nBack to pre-pick.')
    fa.goto_pose(T_ee_world)

    # Go to the pre-placing point. This will need calibration correction.
    # Yes, with pick_w as the preplace... sorry for the bad naming.
    preplace_w = np.copy(place_w)
    preplace_w[2] = DC.Z_PRE_PLACE
    preplace_w = DC.calibration_correction(pix_w=pix1, pick_w=preplace_w)
    T_ee_world = fa.get_pose()
    curr_xy = T_ee_world.translation[:2]
    targ_xy = preplace_w[:2]
    p1_dur = get_duration(curr_xy, targ_xy)
    T_ee_world.translation = preplace_w
    print(f'\nTranslate to pre-place, dur. {p1_dur}')
    fa.goto_pose(T_ee_world, duration=p1_dur)

    # Lower to place gently. Copy preplace_w and adjust the z axis.
    placing_w = np.copy(preplace_w)
    #placing_w[2] = DC.Z_PLACE
    placing_w[2] += (DC.Z_PLACE - DC.Z_PRE_PLACE)
    T_ee_world.translation = placing_w
    print(f'\nLower to place.')
    fa.goto_pose(T_ee_world)

    # Open the gripper.
    print('\nOpening gripper (should release cable)...')
    fa.goto_gripper(width=DC.GRIP_RELEASE)

    # Return to pre-placing point. NOTE: if remove lowering, remove this. But
    # lowering seems to help empirically. Also this doesn't need calibration.
    T_ee_world.translation = preplace_w
    print(f'\nReturn to pre-placing.')
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


def determine_rotation_from_mask(mask, pick, place=None):
    """Determines a rotation from the mask.

    See `test_mask_rotations.py` for more details and tests.
    The `pick` (and possibly `place` if we consider this) should be tuples
    that show image pixels.

    Returns
    -------
    A dict with this information plus a bunch of debugging images.
    """
    assert len(pick) == 2, pick
    p0, p1 = pick
    assert mask.shape == (160, 320), mask.shape

    # Align with `test_mask_rotations.py`, `mask` should be 3 channels, but
    # `mask_prob` should have 1, and types should be uint8.
    mask = mask.astype(np.uint8)  # float64 to uint8
    mask = triplicate(mask, to_int=True)
    mask_prob = mask[:, :, 0]
    mask_copy = np.copy(mask)

    # For bounding box stuff.
    ss = 10

    # Keep adding and returning stuff here.
    stuff = {'pick': pick, 'mask_original': mask}

    # ------------------------- visualize pick+box --------------------------- #
    # Circle the picking point and label it.
    cv2.circle(mask_copy, center=(pick[1],pick[0]), radius=5, color=RED, thickness=2)
    cv2.putText(
        img=mask_copy,
        text="{}".format(pick),
        org=(10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=GREEN,
        thickness=1
    )

    # Put a bounding box around the picking point. Blue with BGR imgs. If it's
    # close to the boundary, then some of the bounding box might not be visible.
    cv2.rectangle(mask_copy, (p1-ss,p0-ss), (p1+ss, p0+ss), BLUE, 2)

    # Make it consistent with viewing and saving.
    mask_copy = rgb_to_bgr(mask_copy)
    stuff['mask_pick_and_local_crop'] = mask_copy
    # ------------------------------------------------------------------------ #

    # ---------------------- Contours crop image ----------------------------- #
    # Now crop and get exactly desired image sizes (with black as padded region).
    mask_prob_crop = mask_prob[p0-ss:p0+ss, p1-ss:p1+ss]  # grayscale
    mask_crop      =      mask[p0-ss:p0+ss, p1-ss:p1+ss]  # 'rgb'

    # It could be smaller due to cropping near the boundary of images.
    # assert mask_prob_crop.shape == (ss*2,ss*2), mask_prob_crop.shape
    # assert mask_crop.shape == (ss*2,ss*2,3), mask_crop.shape

    # Draw contours.
    contours, hierarchy = cv2.findContours(
        image=mask_prob_crop, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    # Show contours on the cropped image.
    image_copy = mask_crop.copy()
    cv2.drawContours(
        image=image_copy,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    stuff['mask_contours_crop'] = image_copy
    # ------------------------------------------------------------------------ #

    # ----------------------------- Fit line? -------------------------------- #
    # This is the first contour, I think biggest?
    cnt = contours[0]

    # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    # Wow this actually works nicely. :) This is the tangent line.
    # Well we might need something here in case of vx or vy as 0?
    # vx and vy returns a normalized vector, so just have to make sure that
    # they aren't all 0?
    image_copy = mask_crop.copy()
    rows, cols = mask_prob_crop.shape
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    print(f'After cv2.fitLine, vx, vy: {vx} {vy}, x, y: {x} {y}')
    if vx == 0 or vy == 0:
        print('Warning! vx or vy are 0, need to re-normalize.')
        # Not sure how principled this is but might be OK?
        if vx == 0:
            vx += 1e-4
        if vy == 0:
            vy += 1e-4
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image_copy, (cols-1,righty), (0,lefty), GREEN, 2)

    # I think we want to get the _perpendicular_ line segment.
    tmp = vx
    vx = vy
    vy = -tmp
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image_copy, (cols-1,righty), (0,lefty), BLUE, 2)
    stuff['mask_contours_crop_fitline'] = image_copy
    # ------------------------------------------------------------------------ #

    # -------------------- Perp line, put on orig img ------------------------ #
    # Let's get the perpendicular line, and put all the info on the original image.
    # This is mask_copy, BTW, but need to be back to rgb now.
    mask_copy = bgr_to_rgb(mask_copy)

    # Simple: just override the cropped region? YES! Nice.
    #assert image_copy.shape == (int(ss*2), int(ss*2), 3), f'{image_copy.shape} {ss}'
    if image_copy.shape != (int(ss*2), int(ss*2), 3):
        print(f'Warning: {image_copy.shape}, means a crop smaller than usual.')
    mask_copy[p0-ss:p0+ss, p1-ss:p1+ss] = image_copy

    # Let's pad the image. Don't pad the 3rd channel! Pad with 255 (i.e., white).
    # This is STRICTLY FOR READABILITY later when we insert the angles (in deg and
    # radian). This is NOT the same as padding for handling crops/grasps close to
    # the image boundary (which would need black values, i.e., 0, for padding).
    mask_copy = np.pad(mask_copy, ((40,40), (40,40), (0,0)), constant_values=255)
    print(f'After padding, mask copy: {mask_copy.shape}')

    # Also before we do this let's annotate the angle. Same color as perpendicular
    # line segment, for consistency (with all the potential BGR vs RGB confusions).
    # https://stackoverflow.com/questions/66839398/python-opencv-get-angle-direction-of-fitline
    # Note that we overrode vx and vy to be the perpendicular case.
    x_axis      = np.array([1, 0])    # unit vector in the same direction as the x axis
    your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
    dot_product = np.dot(x_axis, your_line)
    angle_2_x   = np.arccos(dot_product)
    angle_deg   = np.rad2deg(angle_2_x)
    cv2.putText(
        img=mask_copy,
        text="{:.1f} (rad)".format(angle_2_x[0]),
        org=(20, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=BLUE,
        thickness=1
    )
    cv2.putText(
        img=mask_copy,
        text="{:.1f} (deg)".format(angle_deg[0]),
        org=(120, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=BLUE,
        thickness=1
    )

    stuff['angle_raw_rad'] = angle_2_x
    stuff['angle_raw_deg'] = angle_deg
    # ------------------------------------------------------------------------ #

    # ------------------------------------------------------------------------ #
    # Last parts: need to consider the existing workspace, limits, etc.
    # ------------------------------------------------------------------------ #
    angle_deg = angle_deg[0]  # length one array
    assert 0 <= angle_deg < 180.

    # Offset by 90, cap negative at -25.
    angle_deg_new = angle_deg - 90.
    angle_deg_new = max(angle_deg_new, -25)

    # Negate everything since we counter-clockwise rotation is negative.
    angle_deg_revised = -angle_deg_new

    cv2.putText(
        img=mask_copy,
        text="{:.1f} (revised)".format(angle_deg_revised),
        org=(240, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=BLUE,
        thickness=1
    )

    mask_copy = rgb_to_bgr(mask_copy)
    #cv2.imwrite('mask_contours_all_info_orig.png', mask_copy)
    stuff['mask_contours_crop_all_info_orig'] = mask_copy
    stuff['angle_deg_revised'] = angle_deg_revised
    return stuff


def evaluate_masks(mask_curr, mask_goal):
    """Since we have binary masks let's just compare pixels.

    Just report which pixels are equal. Or which of the white ones are equal.
    Actually since most of the image is just black, we might want to prioritize
    the white pixels.
    """
    assert mask_curr.shape == mask_goal.shape == (160,320), \
        f'{mask_curr.shape} {mask_goal.shape}'
    curr_binary = mask_curr > 0
    goal_binary = mask_goal > 0
    white_curr = np.sum(curr_binary)
    white_goal = np.sum(goal_binary)

    # Equal, considering both white and black pixels.
    equals = np.equal(curr_binary, goal_binary)

    # Only 1 (True) if both pixels are white.
    equals_white = np.logical_and(curr_binary, goal_binary)

    metrics = {}

    # This number will be high since most pixels are black.
    metrics['pix_eq_overall'] = np.sum(equals) / np.prod(equals.shape)

    # Might be a better metric to report? Divide by the minimum.
    metrics['pix_eq_white'] = np.sum(equals_white) / min(white_curr, white_goal)

    # These are mostly just FYI.
    metrics['pix_both_white'] = np.sum(equals_white)
    metrics['pix_white_curr'] = white_curr
    metrics['pix_white_goal'] = white_goal

    return metrics


if __name__ == "__main__":
    pass
