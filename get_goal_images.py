"""Use this script to get goal images.

Well it's pretty simple we just manually set the cable on the plane,
Query the images, and save the _triplicated_ images.
"""
import os
from os.path import join
import cv2
import time
import numpy as np
np.set_printoptions(suppress=True, precision=4, linewidth=150)

from frankapy import FrankaArm
from data_collect import DataCollector
import daniel_config as DC
import daniel_utils as DU

# Put goal images here. We'll also put in color, depth, etc.
SAVEDIR = DC.GOAL_IMG_DIR


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


if __name__ == "__main__":
    # NOTE! be sure that I have ALREADY put the cable on the workspace!
    if not os.path.exists(SAVEDIR):
        os.mkdir(SAVEDIR)
    count = len(
        [x for x in os.listdir(SAVEDIR) if 'goal_' in x and 'mask_trip.png' in x]
    )
    name = f'goal_{str(count).zfill(3)}'

    # Bells and whistles. Move the robot to the same 'home' position.
    fa = FrankaArm()
    fa.close_gripper()
    dc = DataCollector()
    fa.goto_joints(DC.JOINTS_TOP, duration=10, ignore_virtual_walls=True)
    time.sleep(2)

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

    # Save images. We really want `m_img_tr` but we'll save others anyway.
    cv2.imwrite( join(SAVEDIR, f'{name}_cimg.png'),      c_img)
    cv2.imwrite( join(SAVEDIR, f'{name}_cimg_proc.png'), c_img_proc)
    cv2.imwrite( join(SAVEDIR, f'{name}_cimg_bbox.png'), c_img_bbox)
    cv2.imwrite( join(SAVEDIR, f'{name}_dimg.png'),      d_img)
    cv2.imwrite( join(SAVEDIR, f'{name}_dimg_proc.png'), d_img_proc)
    cv2.imwrite( join(SAVEDIR, f'{name}_mask.png'),      m_img)
    cv2.imwrite( join(SAVEDIR, f'{name}_mask_trip.png'), m_img_tr)

    print(f'Done! See files in {SAVEDIR} w/naming: {name}')
