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

# Careful.
WIDTH = 320
HEIGHT = 160

# Put input and output here for real experiments with GCTN.
REAL_INPUT = 'real_input/'
REAL_OUTPUT = 'real_output/'


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


def print_eval_metrics(eval_metrics):
    print(f'Eval metrics:')
    for key in list(eval_metrics.keys()):
        print('  {}: {:0.3f}'.format(key, eval_metrics[key]))


def run_trial(args, fa, dc, T_cam_ee, goal_info=None):
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

    Currently saving these keys in `trial_info`, maps each key to a list.
        'img_dict'             --- length: # actions + 1
        'eval_metrics'         --- length: # actions + 1
        'gctn_dict'            --- length: # actions
        'act_dict'             --- length: # actions
        'stuff_dict_rotation'  --- length: # actions
    The first two have information collected AFTER all actions.

    Args:
        goal_info: a dict with the goal image information.
    """
    trial_info = defaultdict(list)

    # Stays fixed at each time step.
    mask_goal = goal_info['mask_trip'][None,...]
    _mask_goal = mask_goal[0, :, :, 0]  # makes (160,320)

    for t in range(args.max_T):
        print(f'\n********* On time t={t+1} (1-index) / {args.max_T}. *********')

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

        # Get evaluation metric. Index 0 since mask_goal minibatch is size 1.
        eval_metrics = DU.evaluate_masks(mask_curr=m_img, mask_goal=_mask_goal)
        trial_info['eval_metrics'].append(eval_metrics)
        print_eval_metrics(eval_metrics)

        # SAVE HERE! If we do CTRL+C w/GCTN, then that's fine (we have all we need).
        save_stuff(args, trial_info)

        # Determine the action.
        pix0, pix1 = None, None

        if args.method == 'random':
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
            # ---------------------------------------------------------------------- #
            # Look at `REAL_INPUT` and `REAL_OUTPUT` from `perform_physical_rollout()`
            # https://github.com/DanielTakeshi/pybullet-def-envs/blob/physical/load.py
            # These 'REAL' dirs are SEPARATE from the logs where we store `trial_info`,
            # but `trial_info` should have all the same information anyway.
            # ---------------------------------------------------------------------- #
            # Somewhat annoyingly, we have to pass in triplicated images to GCTN. But
            # in GCTN when we _process_ the masks, we only take the 1st channel! :/
            # Also, both mask_obs and mask_goal should be of the same type (float64)?
            # and both have 255 as the nonzero value (not 1).
            # ---------------------------------------------------------------------- #
            mask_obs = m_img_tr[None,...]
            assert mask_obs.shape  == (1, 160, 320, 3), mask_obs.shape
            assert mask_goal.shape == (1, 160, 320, 3), mask_goal.shape
            assert mask_obs.dtype == mask_goal.dtype

            # Clunky: after we do this, we need to scp this data over to `takeshi`.
            in_fname = join(REAL_INPUT, f'in_{str(t).zfill(2)}.pkl')
            gctn_input = {'obs': mask_obs, 'goal': mask_goal}
            with open(in_fname, 'wb') as fh:
                pickle.dump(gctn_input, fh)
            print(f'Saved input to GCTN: {in_fname} ...')

            # Wait for the correct data for time `t` (0-indexed). FYI, we often exit
            # here and that's OK as we've already saved the prior images and metrics.
            out_fname = join(REAL_OUTPUT, f'out_{str(t).zfill(2)}.pkl')
            print(f'Waiting for output from GCTN: {out_fname}; doing CTRL+C is OK!')
            while not os.path.exists(out_fname):
                pass

            # Extract correct data. TODO(daniel): check transposes, etc.
            # Since GCTN has (320,160) images I do think we have to transpose.
            time.sleep(2)  # to prevent 'ran out of input'
            with open(out_fname, 'rb') as fh:
                gctn_dict = pickle.load(fh)
            pix0 = gctn_dict['act_pred']['params']['pixels0']
            pix1 = gctn_dict['act_pred']['params']['pixels1']

            # So that we can assign to pick, place later.
            print(f'From GCTN: pick {pix0}, place {pix1}')
            pix0 = np.int32([pix0[1], pix0[0]])
            pix1 = np.int32([pix1[1], pix1[0]])
            print(f'Revised: pick {pix0}, place {pix1}')

            # Actually we should save ALL this in its own output.
            trial_info['gctn_dict'].append(gctn_dict)

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
        stuff_dict = DU.determine_rotation_from_mask(mask=m_img, pick=pix0)
        z_rot_delta = stuff_dict['angle_deg_revised']

        # Additional debugging.
        print('\nPlanning to execute:')
        print('Pick:  {}  --> World {}'.format(pix0, pick_world))
        print('Place: {}  --> World {}'.format(pix1, place_world))
        print('Z rotation (delta): {:.1f}'.format(z_rot_delta))
        cv2.circle(c_img_bbox, center=(pick[1],pick[0]), radius=5, color=GREEN, thickness=2)
        cv2.circle(c_img_bbox, center=(place[1],place[0]), radius=5, color=BLUE, thickness=2)

        # Let's save this (color images with predictions) in the target directory.
        img_savedir = join(args.savedir, f'c_img_pre_action_time_{str(t).zfill(2)}.png')
        cv2.imwrite(img_savedir, c_img_bbox)

        # Show pre-act image. Press ESC. Then press ENTER (or CTRL+C to abort).
        wname = 'zrot: {:.1f}, pick: {} --> {}, place: {} --> {}'.format(
                z_rot_delta, pix0, pick, pix1, place)
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
            'z_rot': z_rot_delta,
        }
        trial_info['act_dict'].append(act_dict)
        trial_info['stuff_dict_rotation'].append(stuff_dict)
        save_stuff(args, trial_info)

        # The moment of truth ... :-)
        DU.pick_and_place(fa, pick_world, place_world, z_rot_delta, starts_at_top=True)

    # ----------------------------------------------------------------------- #
    # Save the _final_ images. The pick and place should move robot back to top.
    # Also save the final metrics, etc. However, I don't think we will use this
    # much since CTRL+C exits the code, and we already saved beforehand.
    # ----------------------------------------------------------------------- #
    img_dict = dc.get_images()
    trial_info['img_dict'].append(img_dict)

    # Final metrics.
    m_img = img_dict['mask_img']
    eval_metrics = DU.evaluate_masks(mask_curr=m_img, mask_goal=_mask_goal)
    trial_info['eval_metrics'].append(eval_metrics)
    print_eval_metrics(eval_metrics)

    # Now save everything, overriding stuff we saved (as usual).
    save_stuff(args, trial_info)


if __name__ == "__main__":
    # Save within args.outdir subdirs: gctn_{k}, human_{k}, random_{k}, etc.
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='data')
    p.add_argument('--method', type=str, default='random')
    p.add_argument('--max_T', type=int, default=10)
    p.add_argument('--goal_idx', type=int, default=0)
    args = p.parse_args()

    # Bells and whistles, makes data processing a bit easier.
    files_inp = [join(REAL_INPUT,x) for x in os.listdir(REAL_INPUT) if '.pkl' in x]
    files_out = [join(REAL_OUTPUT,x) for x in os.listdir(REAL_OUTPUT) if '.pkl' in x]
    n_inp = len(files_inp)
    n_out = len(files_out)
    if n_inp > 0 or n_out > 0:
        print(f'Removing {n_inp} and {n_out} items in in/out directories.')
        for ff in files_inp:
            print(f'  removing: {ff}')
            os.remove(ff)
        for ff in files_out:
            print(f'  removing: {ff}')
            os.remove(ff)

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

    # random: pick any location on the cable, no need for a goal image.
    # gctn: pick one model to run on my other machine.
    assert args.method in ['random', 'gctn']
    goal_info = {}
    if args.method == 'gctn':
        goal_img_path = join(
            DC.GOAL_IMG_DIR, f'goal_{str(args.goal_idx).zfill(3)}_mask_trip.png'
        )
        assert os.path.exists(goal_img_path), goal_img_path
        goal_mask = cv2.imread(goal_img_path).astype('float')  # need float
        goal_info['mask_trip'] = goal_mask
        assert goal_mask.shape == (160,320,3), goal_mask.shape
        assert len(np.unique(goal_mask)) == 2, goal_mask
        assert np.max(goal_mask) > 1, np.max(goal_mask)

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
    T_cam_ee = DU.load_transformation(filename, as_rigid_transform=True)
    print(f'Loaded transformation from {filename}:\n{T_cam_ee}\n')

    print('='*100)
    print('='*100)
    print(f'RUNNING TRIAL {count}!!')
    print('='*100)
    print('='*100)
    run_trial(args, fa=fa, dc=dc, T_cam_ee=T_cam_ee, goal_info=goal_info)
    print(f'Done with trial! See the savedir:\n\t{args.savedir}')