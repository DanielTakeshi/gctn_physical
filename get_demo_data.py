"""
A very simple script to collect data. I did this for the first dataset
I collected in early December. Remember, we do have to manually label.

Tips:

0. Don't forget to save the beginning image! We hit CTRL+C when we are DONE.

1. Do this for ONE EPISODE, then stop and move the robot with e-stop. Thus each
time we do this the robot should have to manually reset. This will improve the
robustness of the data to noise in the robot's motion since the camera position
is never exactly the same.

2. Put some tape (assuming it will be masked out) on the workspace, to stabilize
the 'goal' configuration, so that we can induce some structure in the demo data.

3. Based on what I've seen tentatively, we may have to be careful about having
too much of the data be on the very endpoints of the cable; minor errors can
cause the robot to completely miss those endpoints. So be mindful of that.

4. Please avoid being too close to the edges of the workspace. We want to make
sure that the robot can easily reach anywhere, and some spots on the edges may
be hard combined with rotations.

5. After each trial, inspect the resulting images IMMEDIATELY! If it looks bad,
delete the directory so the number can stay consistent.

6. When performing the demonstration, do not rotate my finger grasp after the
initial grasp! Just have it move in the same direction, that's probably how it
will be for the robot.
"""
import os
from os.path import join
import cv2
import pickle
import numpy as np
np.set_printoptions(suppress=True, precision=3)

from frankapy import FrankaArm
from data_collect import DataCollector
import daniel_config as DC
import daniel_utils as DU
DEMO_DIR = 'demos/'


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


def save_image(img, path, debug=False):
    if img is None:
        return
    cv2.imwrite(path, img)
    if debug:
        print("Saved: {} of size {}".format(path, img.shape))


def get_demos_human(save_dir, dc):
    """Get images from a human."""
    while True:
        print('ENTER to get images, CTRL+C to exit')
        DU.wait_for_enter()

        # Numbering.
        n_imgs = len([x for x in os.listdir(save_dir) if 'mask' in x and '.png' in x])
        n_imgs = str(n_imgs).zfill(4)

        # This technically has all I need but it might be easier to also save
        # the separate images because we can cycle w/out using pickle loading.
        images = dc.get_images()
        dict_fname = join(save_dir, 'dc_image_dict_{}'.format(n_imgs))
        with open(dict_fname, 'wb') as fh:
            pickle.dump(images, fh)

        # Save individual images.
        color_raw = images['color_raw']
        color_proc = images['color_proc']
        color_bbox = images['color_bbox']
        depth_proc = images['depth_proc']
        mask_img = images['mask_img']  # (160,320)
        c_raw_path  = join(save_dir, 'cimg_raw_{}.png'.format(n_imgs))
        c_proc_path = join(save_dir, 'cimg_proc_{}.png'.format(n_imgs))
        c_bbox_path = join(save_dir, 'cimg_bbox_{}.png'.format(n_imgs))
        d_proc_path = join(save_dir, 'dimg_proc_{}.png'.format(n_imgs))
        m_img_path  = join(save_dir, 'mask_img_{}.png'.format(n_imgs))
        save_image(color_raw,  c_raw_path)
        save_image(color_proc, c_proc_path)
        save_image(color_bbox, c_bbox_path)
        save_image(depth_proc, d_proc_path)
        save_image(mask_img,   m_img_path)
        print('Collected and saved images at index: {}'.format(n_imgs))
        print('  color_raw:  {}'.format(color_raw.shape))
        print('  color_proc: {}'.format(color_proc.shape))
        print('  depth_proc: {}'.format(depth_proc.shape))
        print('  mask_img:   {}'.format(mask_img.shape))


if __name__ == "__main__":
    if not os.path.exists(DEMO_DIR):
        os.mkdir(DEMO_DIR)

    # Bells and whistles.
    n_sub_dirs = len([x for x in os.listdir(DEMO_DIR) if 'demo_' in x])
    sub_dir = 'demo_{}'.format(str(n_sub_dirs).zfill(4))
    save_dir = join(DEMO_DIR, sub_dir)
    print('Collecting demos to: {}'.format(save_dir))
    assert not os.path.exists(save_dir)
    os.mkdir(save_dir)

    print(f'Creating FrankaArm...')
    fa = FrankaArm()
    fa.close_gripper()

    # Move robot to top.
    print(f'\nMove to JOINTS_TOP:\n{DC.JOINTS_TOP}')
    fa.goto_joints(DC.JOINTS_TOP, duration=10, ignore_virtual_walls=True)
    T_ee_world = fa.get_pose()
    print(f'T_ee_world:\n{T_ee_world}\n')

    print(f'Creating DataCollector...')
    dc = DataCollector()

    get_demos_human(save_dir, dc)
