"""Our main experiment script. This should be for rollouts.

Actually to make it simpler, let's also use this for just testing the
pick and place from images? We have test_pick_place.py but that one
assumes we hard-code the position to go. Here, we should hande the full
image pixel to world manipulation pipeline.
"""
import os
from os.path import join
import cv2
import time
import json
import pickle
import datetime
import argparse
import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=150)
from collections import defaultdict

from frankapy import FrankaArm
from autolab_core import RigidTransform
from data_collect import DataCollector
import daniel_config as DC
import daniel_utils as DU

SAVEDIR = 'logs/'
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)

# For `click_and_crop`.
POINTS          = []
CENTER_OF_BOXES = []
STACKED_IMAGES  = []

# Careful.
WIDTH = 320
HEIGHT = 160


# TODO(daniel): adapted from GCTN code, should test if we want human labels here.
def click_and_crop(event, x, y, flags, param):
    global POINTS, CENTER_OF_BOXES, STACKED_IMAGES

    assert len(STACKED_IMAGES) >= 1, STACKED_IMAGES
    stacked_copy = np.copy(STACKED_IMAGES[-1])

    # If left mouse button clicked, record the starting (x,y) coordinates
    # and indicate that cropping is being performed
    if event == cv2.EVENT_LBUTTONDOWN:
        POINTS.append((x,y))

    # Check to see if the left mouse button was released
    elif event == cv2.EVENT_LBUTTONUP:
        # Record ending (x,y) coordinates, indicate that cropping is finished, save center!
        POINTS.append((x,y))

        upper_left = POINTS[-2]
        lower_right = POINTS[-1]
        assert upper_left[0] < lower_right[0]
        assert upper_left[1] < lower_right[1]
        center_x = int(upper_left[0] + (lower_right[0]-upper_left[0])/2)
        center_y = int(upper_left[1] + (lower_right[1]-upper_left[1])/2)
        CENTER_OF_BOXES.append( (center_x,center_y) )

        # Draw rectangle around the region of interest.
        cv2.rectangle(img=stacked_copy,
                      pt1=POINTS[-2],
                      pt2=POINTS[-1],
                      color=(0,0,255),
                      thickness=2)
        cv2.circle(img=stacked_copy,
                   center=CENTER_OF_BOXES[-1],
                   radius=3,
                   color=(0,0,255),
                   thickness=-1)
        cv2.putText(img=stacked_copy,
                    text="{}".format(CENTER_OF_BOXES[-1]),
                    org=CENTER_OF_BOXES[-1],
                    fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                    fontScale=0.5,
                    color=(0,0,255),
                    thickness=1)

        # A separate pop-up shows the pixel labels, x-axis (horizontal) first.
        cent = CENTER_OF_BOXES[-1]
        window_name_2 = f"click_and_crop, center: {cent}. Press any key."
        cv2.namedWindow(window_name_2, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name_2, stacked_copy.shape[1]*2, stacked_copy.shape[0]*2)
        cv2.imshow(window_name_2, stacked_copy)


# TODO(daniel): adapted from GCTN code, should test if we want human labels here.
def _human_label(cimg_curr, dimg_curr, mask_curr, cimg_next, dimg_next, mask_next,
        cimg_goal, dimg_goal, mask_goal, demo_dir, timestep):
    """We might have to do some manual data labeling, unfortunately.
    The current pipeline if we have to label:
    - We go through each episode, and each time step within an episode.
    - For each time, that's when we call this method.
    - First, drag a box around the RGB image (top left) to get the picking spot.
    - A pop-up window will show up to confirm that, move it aside.
    - Second, on the ORIGINAL set of images (NOT the pop-up), in the upper right, draw
        another box around the RGB image to get the placing spot.
    - Then click any key to exit (assuming it looks good).
    - Check afterwards all the debugging images. Do this BEFORE going to the next demo.
    IF I MADE A MISTAKE, check which ones I made a mistake on and fix code (not in
        this method) to just fix the mistaken trials. Also delete any act_{timestep}.pkl
        that were just created.
    Image coordinate conventions:
    When we use `pixels0` and `pixels1`, values are (x,y) where x is positive x in
    the usual direction (horizontal) but where +y is pointing downwards. The (0,0)
    position will be visualized (with cv2.draw... methods) in the upper left corner.
    https://stackoverflow.com/questions/57068928/
    TODO(daniel) -- what about Transporters training?
    """
    global STACKED_IMAGES

    # Show stacked image, (time t, time t+1, goal). The last action is at: t+1 = goal.
    stacked = np.hstack((
        np.vstack((cimg_curr, dimg_curr, mask_curr)),
        np.vstack((cimg_next, dimg_next, mask_next)),
        np.vstack((cimg_goal, dimg_goal, mask_goal)),
    ))
    STACKED_IMAGES.append(stacked)

    # Apply callback and drag a box around the end-effector on the (updated!) image.
    window_name = f'Stacked image size: {stacked.shape}'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, stacked.shape[1]*2, stacked.shape[0]*2)
    cv2.setMouseCallback(window_name, click_and_crop) # Record clicks to this window!
    cv2.imshow(window_name, stacked)
    key = cv2.waitKey(0)

    # With two bounding boxes on our original image, we'll get the pick and place.
    # But the pixels1 is a separate image, hence need to subtract the width (360).
    pixels0 = CENTER_OF_BOXES[-2]
    pixels1 = CENTER_OF_BOXES[-1]
    pixels1 = (pixels1[0] - WIDTH, pixels1[1])
    print(f'Actions (pixels): pick {pixels0} --> place {pixels1}')
    assert pixels0[0] >= 0 and pixels0[1] >= 0, pixels0
    assert pixels1[0] >= 0 and pixels1[1] >= 0, pixels1
    assert pixels0[0] < WIDTH and pixels0[1] < HEIGHT, pixels0
    assert pixels1[0] < WIDTH and pixels1[1] < HEIGHT, pixels1

    # To copy this on the second stacked image.
    _pixels0 = (pixels0[0] + WIDTH, pixels0[1])
    _pixels1 = (pixels1[0] + WIDTH, pixels1[1])

    # Save a debugging image which visualizes pick (blue) and place (red).
    debug_img = np.copy(stacked)
    debug_name = os.path.join(demo_dir, f'debug_acts_{str(timestep).zfill(3)}.png')

    # For the original current image.
    cv2.circle(img=debug_img, center=pixels0, radius=2, color=(255,0,0), thickness=-1)
    cv2.circle(img=debug_img, center=pixels1, radius=2, color=(0,0,255), thickness=-1)
    cv2.circle(img=debug_img, center=_pixels0, radius=2, color=(255,0,0), thickness=-1)
    cv2.circle(img=debug_img, center=_pixels1, radius=2, color=(0,0,255), thickness=-1)

    # Actually for the text we don't really care _where_ it is as long as its legible.
    # So I'm going to shift it by -x and +y a bit.
    cv2.putText(img=debug_img,
                text="{}".format(pixels0),
                org=(pixels0[0]-25, pixels0[1]+25),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(255,0,0),
                thickness=1)
    cv2.putText(img=debug_img,
                text="{}".format(pixels1),  # the real label
                org=(_pixels1[0]-25, _pixels1[1]+25),  # has underscore, so shifted
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=(0,0,255),
                thickness=1)

    # Upscale size by 2 and then write.
    debug_img = cv2.resize(debug_img, (debug_img.shape[1]*2, debug_img.shape[0]*2))
    cv2.imwrite(debug_name, debug_img)
    print(f'Saved to: {debug_name}')
    cv2.destroyAllWindows()

    # For training Transporters.
    act_stuff = {'pixels0': pixels0, 'pixels1': pixels1}
    return act_stuff


# TODO(daniel): stop delaying and implement this!
def get_z_rot(m_img, pix0, pix1):
    """Need to get the z rotation!"""
    return 0.0


def get_demos_human(robot):
    """Get images from a human. Press 'y' to keep getting images.

    Originally from:
    https://github.com/thomasweng15/bimanual_folding/blob/daniel_franka_dec2022/bimanual_ros/transporters/main.py

    Also, can debug to check that images look OK. Before each session with the
    robot, check to see that the bounding box is aligned with the tape I put on
    the foam to mark the workspace limits.
    """
    im_dir = 'images/'
    n_sub_dirs = len([x for x in os.listdir(im_dir) if 'demo_' in x])
    sub_dir = 'demo_{}'.format(str(n_sub_dirs).zfill(3))
    im_dir = join(im_dir, sub_dir)
    print('Collecting images, will put them in: {}'.format(im_dir))
    os.mkdir(im_dir)

    while True:
        usr_input = raw_input("Record images? (y). Else, exit and finish demo (n): ")
        while (usr_input.lower() != "y") and (usr_input.lower() != "n"):
            usr_input = raw_input("Please enter a valid option. (y/n)").lower()
        if usr_input.lower() == "n":
            break

        images = robot.get_dict_of_images()
        n_imgs = len([x for x in os.listdir(im_dir) if 'mask' in x and '.png' in x])
        n_imgs = str(n_imgs).zfill(3)
        cimg   = images['c_image']
        cimg_b = images['c_image_bbox']
        cimg_p = images['c_image_proc']
        dimg   = images['d_image']
        dimg_p = images['d_image_proc']
        mask   = images['m_image']
        cimg_path   = join(im_dir, 'cimg_{}.png'.format(n_imgs))
        cimg_b_path = join(im_dir, 'cimg_b_{}.png'.format(n_imgs))
        cimg_p_path = join(im_dir, 'cimg_p_{}.png'.format(n_imgs))
        dimg_path   = join(im_dir, 'dimg_{}.png'.format(n_imgs))
        dimg_p_path = join(im_dir, 'dimg_p_{}.png'.format(n_imgs))
        mask_path   = join(im_dir, 'mask_{}.png'.format(n_imgs))
        U.save_image(cimg, cimg_path)
        U.save_image(cimg_b, cimg_b_path)
        U.save_image(cimg_p, cimg_p_path)
        U.save_image(dimg, dimg_path)
        U.save_image(dimg_p, dimg_p_path)
        U.save_image(mask, mask_path)
        print('Collected and saved images at index: {}'.format(n_imgs))
        print('  cimg:   {}'.format(cimg.shape))
        print('  cimg_p: {}'.format(cimg_p.shape))
        print('  dimg:   {}'.format(dimg.shape))
        print('  dimg_p: {}'.format(dimg_p.shape))


def check_imgs(img_dict):
    cimg = img_dict['color_raw']
    dimg = img_dict['depth_raw']
    dimg_proc = img_dict['depth_proc']
    mask_im = img_dict['mask_img']
    assert cimg is not None
    assert dimg is not None
    assert dimg_proc is not None
    assert mask_im is not None
    assert cimg.shape == dimg_proc.shape, f'{cimg.shape}, {dimg_proc.shape}'


def save_stuff(args, trial_info):
    """Saves info from trial, at each time step (overwriting if needed).

    Currently just saving images and actions each time step, both as dicts.
    We will save 1 more set of images, after the last action, to get the
    final images.
    """
    p_fname = join(args.savedir, 'trial_info.pkl')
    with open(p_fname, 'wb') as fh:
        pickle.dump(trial_info, fh)


def run_trial(args, count, fa, dc, T_cam_ee):
    """Runs one trial.

    Also supports a mode where a human picks at the pixel for the robot to go
    and grasp at, or if the pixel is chosen randomly.

    How to convert from pixel to world action? Remember: pix0, pix1 are w.r.t. the
    (160,320) cropped and resized image. But we need d_img which is from the orig.
    (720,1280) image and shows distances in millimeters. However we also cropped
    that to (320,640) before resizing. So, it feels like we take pix0, pix1 and
    double them, and we pretend that's the pixel location in a "(320,640)" image.

    Then for the first value (along shorter axis, length 320) we add by DC.crop_y
    which is the value we offset it. Since it was originally 720, we chopped off
    200 pix on either side, hence offset by that amount. Similarly for the second
    value (along longer axis, length 640) we add by DC.crop_x. This way the (0,0)
    in the (320,640) image turns into a pixel of (DC.crop_x, DC.crop_y) in the
    original d_img and that's where we can query depth. I think this might work.

    TODO(daniel): need to supply goal images!!!!!!! But that can come later once
    we have done more data collection.
    """
    trial_info = defaultdict(list)

    for t in range(args.max_T):
        # Start with moving robot to home position, compute EE pose.
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
        check_imgs(img_dict)
        c_img = img_dict['color_raw']
        c_img_proc = img_dict['color_proc']
        c_img_bbox = img_dict['color_bbox']
        d_img = img_dict['depth_raw']
        d_img_proc = img_dict['depth_proc']
        m_img = img_dict['mask_img']  # (160,320)
        m_img_tr = DU.triplicate(m_img)  # (160,320,3)
        trial_info['img_dict'].append(img_dict)

        # Determine the action.
        pix0, pix1 = None, None

        if args.method == 'human':
            raise NotImplementedError()

        elif args.method == 'random':
            # Randomly pick a valid point on the m_img (on the cable).
            pix0 = DU.sample_distribution(m_img)

            # For placing, let's also pick a random point but not at boundary?
            mask_place = np.zeros_like(m_img)
            mask_place[20:160-20, 20:320-20] = 1
            #pix1 = np.int32([80,260])  # or we can just hard-code it ...
            pix1 = DU.sample_distribution(mask_place)

            # Annotate, remember that we need the `center` reversed.
            cv2.circle(m_img_tr, center=(pix0[1],pix0[0]), radius=5, color=GREEN, thickness=2)
            cv2.circle(m_img_tr, center=(pix1[1],pix1[0]), radius=5, color=BLUE, thickness=2)
            cv2.putText(
                img=m_img_tr,
                text="{}".format(pix0),
                org=(10, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=GREEN,
                thickness=1
            )
            cv2.putText(
                img=m_img_tr,
                text="{}".format(pix1),
                org=(90, 20),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=0.5,
                color=BLUE,
                thickness=1
            )

            # Create stacked image for visualization.
            if False:
                stacked = np.hstack((m_img_tr, c_img_proc))
                cv2.imwrite('test.png', stacked)  # shows c_img_proc
                wname = 'pix0: {}, pix1: {}'.format(pix0, pix1)
                cv2.imshow(wname, stacked)  # doesn't show c_img_proc??
                key = cv2.waitKey(0)  # Press ESC

        elif args.method == 'gctn':
            raise NotImplementedError()

        else:
            raise ValueError(args.method)

        # ------------------------------------------------------------------- #
        # ----------------------- Convert pixel to world -------------------- #
        # ------------------------------------------------------------------- #
        # Intuition: increasing v means moving in the -y direction wrt robot base.

        # CAREFUL: this depends on cropping values in data_collect.py!
        pick  = pix0 * 2
        place = pix1 * 2
        pick[0] += dc.crop_y
        pick[1] += dc.crop_x
        place[0] += dc.crop_y
        place[1] += dc.crop_x

        # Check 'revised' pick and place on the original (720,1280)-sized images.
        if False:
            cv2.circle(c_img, center=(pick[1],pick[0]), radius=5, color=GREEN, thickness=2)
            cv2.circle(c_img, center=(place[1],place[0]), radius=5, color=BLUE, thickness=2)
            cv2.imwrite('c_img.png', c_img)

        # Resume calculations.
        uu = np.array([pick[0], place[0]]).astype(np.int64)
        vv = np.array([pick[1], place[1]]).astype(np.int64)
        depth_uv = d_img[uu, vv]
        print(f'At (uu,vv), depth:\nu={uu}\nv={vv})\ndepth={depth_uv}')

        # Convert pixels to EE world coordinates. Shape (N,4)
        world_coords = DU.uv_to_world_pos(
            T_cam_world, u=uu, v=vv, z=depth_uv, return_meters=True
        )
        print(f'world_coords ({world_coords.shape}):\n{world_coords}')

        # Use all xyz but later in pick and place, we'll use pre-selected z values.
        pick_world  = world_coords[0, :3]
        place_world = world_coords[1, :3]
        # ------------------------------------------------------------------- #

        # Don't forget to compute a rotation! We'll annotate this to the image.
        z_rot = get_z_rot(m_img, pix0, pix1)

        # Additional debugging.
        print('\nPlanning to execute:')
        print('Pick:  {}  --> World {}'.format(pix0, pick_world))
        print('Place: {}  --> World {}'.format(pix1, place_world))
        cv2.circle(c_img_bbox, center=(pick[1],pick[0]), radius=5, color=GREEN, thickness=2)
        cv2.circle(c_img_bbox, center=(place[1],place[0]), radius=5, color=BLUE, thickness=2)
        #cv2.imwrite('c_img_pre_action.png', c_img_bbox)

        # Show pre-act image. Press ESC. Then press ENTER (or CTRL+C to abort).
        wname = 'pick: {} --> {}, place: {} --> {}'.format(pix0, pick, pix1, place)
        cv2.imshow(wname, c_img_bbox)
        _ = cv2.waitKey(0)  # Press ESC
        cv2.destroyAllWindows()
        DU.wait_for_enter()  # PRESS ENTER

        # Save repeatedly so that we can save trials in prog or which failed.
        act_dict = {
            'pix0': pix0,    # pick pixels on (160,320)
            'pix1': pix1,    # place pixels on (160,320)
            'pick': pick,    # pick pixels on (720,1280)
            'place': place,  # place pixels on (720,1280)
            'pick_w': pick_world,
            'place_w': place_world,
            'z_rot': z_rot,
        }
        trial_info['act_dict'].append(act_dict)
        save_stuff(args, trial_info)

        # The moment of truth ... :-)
        DU.pick_and_place(fa, pick_world, place_world, z_rot, starts_at_top=True)

    # Save the _final_ images. The pick and place should move robot back to top.
    img_dict = dc.get_images()
    trial_info['img_dict'].append(img_dict)
    save_stuff(args, trial_info)


if __name__ == "__main__":
    # Save within args.outdir subdirs: gctn_{k}, human_{k}, random_{k}, etc.
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='data')
    p.add_argument('--method', type=str, default='random')
    p.add_argument('--max_T', type=int, default=8)
    args = p.parse_args()

    # Which trial? Assume we count and then add to data dir.
    assert os.path.exists(args.outdir), args.outdir
    trial_head = join(args.outdir, args.method)
    if not os.path.exists(trial_head):
        os.mkdir(trial_head)
    count = len([x for x in os.listdir(trial_head) if 'trial_' in x])
    args.date = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    suffix = f'trial_{str(count).zfill(3)}__{args.date}'
    args.savedir = join(trial_head, suffix)

    # Dump info to the save dir ASAP.
    os.mkdir(args.savedir)
    args.savejson = join(args.savedir, 'args.json')
    with open(args.savejson, 'w') as fh:
        json.dump(vars(args), fh, indent=4)

    # random: pick any location on the cable.
    # gctn: must have one running on my other machine
    assert args.method in ['random', 'human', 'gctn']

    print(f'Creating FrankaArm...')
    fa = FrankaArm()
    fa.close_gripper()

    print(f'Creating DataCollector...')
    dc = DataCollector()

    # The calibration file, copied from `/<HOME>/.ros/easy_handeye`.
    filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_v04.yaml'  # from 02/05/2022
    T_cam_ee = DU.load_transformation(filename, as_rigid_transform=True)
    print(f'Loaded transformation from {filename}:\n{T_cam_ee}\n')

    print('='*100)
    print('='*100)
    print(f'RUNNING TRIAL {count}!!')
    print('='*100)
    print('='*100)
    run_trial(args, count, fa=fa, dc=dc, T_cam_ee=T_cam_ee)
    print(f'Done with trial! See the savedir:\n\t{args.savedir}')