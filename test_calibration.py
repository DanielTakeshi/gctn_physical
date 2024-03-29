"""A script to try and improve calibration.

We might have a chessboard or another pattern, and try to get the
robot to move there. Some references:
https://www.tutorialspoint.com/how-to-find-patterns-in-a-chessboard-using-opencv-python
"""
import cv2
import argparse
import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=150)
from frankapy import FrankaArm
from data_collect import DataCollector
import daniel_config as DC
import daniel_utils as DU

SAVEDIR = 'logs/'
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# Careful.
WIDTH = 320
HEIGHT = 160

REAL_OUTPUT = 'real_output/'


def test_calib(args, fa, dc, T_cam_ee):
    """Test calibration. Careful with the `nline` and `ncol` choices.

    NOTE: if using 4,3 respectively this means we detect the 4x3 inner corners
    (12 total).  If I print a calibration grid, this ignores the corners at the
    very edges. So compute the number of rows/columns of the full grid, and
    subtract one. Also I can swap nline & ncol, and it's OK (it just makes the
    drawing a different direction).
    """
    print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
    fa.goto_joints(DC.JOINTS_TOP, duration=10, ignore_virtual_walls=True)
    T_ee_world = fa.get_pose()
    print(f'T_ee_world:\n{T_ee_world}\n')

    # I _think_ this seems OK? It passes my sanity checks.
    T_cam_world = T_ee_world * T_cam_ee
    T_cam_world.translation *= 1000.0
    print(f'T_cam_world now w/millimeters:\n{T_cam_world}\n')

    # Get the aligned color and depth images.
    img_dict = dc.get_images()
    c_img = img_dict['color_raw']
    c_img_proc = img_dict['color_proc']  # use this
    c_img_bbox = img_dict['color_bbox']
    d_img = img_dict['depth_raw']

    # Depends on the chessboard.
    nline = 7
    ncol = 4
    img = np.copy(c_img_proc)
    print(f'Searching for corners on img: {img.shape}')

    # Find the chessboard corners. ret is True if corners are detected.
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (nline, ncol), None)
    print(f'Corners detected? {ret}')
    if ret:
        print(f'Shape: {corners.shape}')
    else:
        return

    # This should draw the inner corners.
    cv2.drawChessboardCorners(img, (nline, ncol), corners,ret)
    wname = f'{nline} x {ncol} chessboard'
    cv2.imshow(wname, img)
    cv2.waitKey(0)  # Press ESC
    cv2.destroyAllWindows()
    img_savedir = f'c_img_all_corners.png'
    cv2.imwrite(img_savedir, img)

    # Convert to ints (pixels). Note: `img` is (160,320,3) but these pixel
    # values seem to be such that 1st row in `corners_pix` goes up to 320.
    corners_pix = corners.squeeze().astype(np.int)
    print(f'Corners pixels: {corners_pix.shape}')
    print(corners_pix)  # (num_corners, 2)

    picks_pix = []
    picks_world = []
    for idx in range(corners_pix.shape[0]):
        # pix0[0] ranges from (0,160), shorter axis.
        # pix0[1] ranges from (0,320), longer axis.
        pix0 = np.array( [corners_pix[idx,1], corners_pix[idx,0] ] ).astype(np.int)
        print(f'On corner {idx}, pick pixels: {pix0}')
        picks_pix.append(pix0)

        # ------------------------------------------------------------------- #
        # ----------------------- Convert pixel to world -------------------- #
        # ------------------------------------------------------------------- #
        # Intuition: increasing v means moving in the -y direction wrt robot base.
        # CAREFUL: this depends on cropping values in data_collect.py!
        pick  = pix0 * 2
        pick[0] += dc.crop_y
        pick[1] += dc.crop_x

        # Resume calculations.
        uu = np.array([pick[0]]).astype(np.int64)
        vv = np.array([pick[1]]).astype(np.int64)
        depth_uv = d_img[uu, vv]
        #print(f'At (uu,vv), depth:\nu={uu}\nv={vv})\ndepth={depth_uv}')

        # Convert pixels to EE world coordinates. Shape (N,4)
        world_coords = DU.uv_to_world_pos(
            T_cam_world, u=uu, v=vv, z=depth_uv, return_meters=True
        )
        #print(f'world_coords ({world_coords.shape}):\n{world_coords}')

        # Use all xyz but later we can use pre-selected z values.
        pick_world  = world_coords[0, :3]
        print(f'pick point: {pick_world}')
        picks_world.append(pick_world)
        # ------------------------------------------------------------------- #

        # Let's save this (color images with predictions) in the target directory.
        img_copy = np.copy(c_img_bbox)
        cv2.circle(img_copy, center=(pick[1],pick[0]), radius=5, color=GREEN, thickness=2)
        img_savedir = f'c_img_pre_action_time_{str(idx).zfill(2)}.png'
        cv2.imwrite(img_savedir, img_copy)

    if not args.runrobot:
        return

    # Well actualy let's just get all the world positions, then we can do this.
    for idx in range(corners_pix.shape[0]):
        pix_w = picks_pix[idx]
        pick_w = picks_world[idx]

        # Hopefully it's consistent among rotations.
        z_rot_delta = 0.0

        # Additional debugging.
        print('\nPlanning to execute:')
        print('Pick:  {}  --> World {}'.format(pix_w, pick_w))
        print('Z rotation (delta): {:.1f}'.format(z_rot_delta))

        # Go to first two waypoints. Well we might only do this sometimes?
        if idx == 0:
            print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
            fa.goto_joints(DC.JOINTS_WP1, duration=5)
        print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
        fa.goto_joints(DC.JOINTS_WP2, duration=10)

        # Handle rotations.
        if z_rot_delta != 0:
            DU.rotate_EE_one_axis(
                fa, deg=z_rot_delta, axis='z', use_impedance=True, duration=5
            )

        # Translate to be above the picking point on the chess board.
        prepick_w = np.copy(pick_w)
        prepick_w[2] = DC.Z_PRE_PICK
        prepick_w = DC.calibration_correction(
            pix_w=pix_w, pick_w=prepick_w, z_rot_delta=z_rot_delta
        )
        T_ee_world = fa.get_pose()
        T_ee_world.translation = prepick_w
        print(f'Corrected prepick_w value: {prepick_w}')
        fa.goto_pose(T_ee_world)

        # Lower to be on the chess board. Note: THIS NEEDS TO COPY `prepick_w`.
        # Not the pick_w. Also if we want the z axis hack we have to do it here. :/
        # Or we can adjust this to just be a delta? Sigh. Yeah let's do delta.
        # We increment (really, decrease since it's negative) z by some amount.
        picking_w = np.copy(prepick_w)
        _change_z = DC.Z_PICK - DC.Z_PRE_PICK
        picking_w[2] += _change_z
        #picking_w[2] = DC.Z_PICK
        T_ee_world.translation = picking_w
        fa.goto_pose(T_ee_world)

        # Return to second waypoint, first waypoint, then the top.
        print(f'\nMove to JOINTS_WP2:\n{DC.JOINTS_WP2}')
        fa.goto_joints(DC.JOINTS_WP2, duration=6)
        #print(f'\nMove to JOINTS_WP1:\n{DC.JOINTS_WP1}')
        #fa.goto_joints(DC.JOINTS_WP1, duration=10)
        #print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
        #fa.goto_joints(DC.JOINTS_TOP, duration=5, ignore_virtual_walls=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--runrobot', type=int, default=1,
        help='use 0 if I just want to check the chessboard detection process')
    args = p.parse_args()

    print(f'Creating FrankaArm...')
    fa = FrankaArm()
    fa.close_gripper()

    print(f'Creating DataCollector...')
    dc = DataCollector()

    # The calibration file, copied from `/<HOME>/.ros/easy_handeye`.
    #filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_v04.yaml'  # 02/05/2023
    #filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_v05.yaml'  # 02/10/2023
    #filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_v06.yaml'  # 02/12/2023
    filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_10cm_v04.yaml'  # 02/17/2023
    #filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_10cm_v05.yaml'  # 02/17/2023
    T_cam_ee = DU.load_transformation(filename, as_rigid_transform=True)
    print(f'Loaded transformation from {filename}:\n{T_cam_ee}\n')

    test_calib(args, fa=fa, dc=dc, T_cam_ee=T_cam_ee)