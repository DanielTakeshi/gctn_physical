"""Use this script to inspect any results.

Report any test-time statistics, report qualitatively all the
images (and save to directories for easy viewing), etc.
"""
import os
from os.path import join
import cv2
import json
import pickle
import argparse
import numpy as np
import daniel_config as DC
#import daniel_utils as DU

# Trials which didn't record FINAL images.
EXCLUDE = [
    'trial_000__2023-02-16_14-54-52',  # doesn't have eval metrics as well
    'trial_001__2023-02-16_16-05-19',
]


def incorrect_length(dir_one):
    _, tail = os.path.split(dir_one)
    return tail in EXCLUDE


def saveimg(imgdir, img, transpose=False, debugprint=False, bgr_rgb=False,
        rgb_bgr=False):
    """Just to reduce amount of typing, etc."""
    assert not (bgr_rgb and rgb_bgr)
    if bgr_rgb:
        pass
    if rgb_bgr:
        pass
    if transpose:
        img = img.transpose(1,0,2)
    cv2.imwrite(imgdir, img)
    if debugprint:
        print(f'  saved: {imgdir}')


def inspect(dir_one, goal_mask):
    """Inspect one trial's directory.

    Might return some statistics so we can compute method-wide stats.
    Note: lengths of img_dict and eval_metrics should be 1 more than the
    others, but I didn't do this for a few initial trials.

    Ideally this can be just a few images that maybe I copy and paste into
    my Notion, or will put in a set of slides to show images.

    Args:
        dir_one: directory for one trial, starts with 'data/gctn'.
    """
    info_dir = join(dir_one, 'trial_info.pkl')
    with open(info_dir, 'rb') as fh:
        info = pickle.load(fh)
    print(f'Loaded info, keys: {info.keys()}')
    for key in list(info.keys()):
        print(f'key: {key}, len: {len(info[key])}')
    num_acts = len(info['gctn_dict'])

    print('Observable images, NOT output of GCTN (that is for actions).')
    for t in range(num_acts + 1):
        tt = str(t).zfill(2)
        img_dict = info['img_dict'][t]

        # The (160,320,3) color image. Probabaly what we'll show in slides.
        cproc = img_dict['color_proc']
        imgdir = join(dir_one, f'img_dict_{tt}_color_proc.png')
        saveimg(imgdir, cproc, debugprint=False)

        # But we really should also show the mask as that's the input.
        mask = img_dict['mask_img']
        imgdir = join(dir_one, f'img_dict_{tt}_mask_img.png')
        saveimg(imgdir, mask, debugprint=False)

    # NOTE: these have to be transposed!
    print('GCTN attention/transport heat maps, ACTIONS, etc.')
    for t in range(num_acts):
        tt = str(t).zfill(2)
        gctn_dict = info['gctn_dict'][t]

        # Attention heat map.
        hmap_attn = gctn_dict['extras']['attn_heat_bgr']
        imgdir = join(dir_one, f'gctn_dict_{tt}_heat_attn.png')
        saveimg(imgdir, hmap_attn, transpose=True, debugprint=False)

        # Transport heat map.
        hmap_tran_l = gctn_dict['extras']['tran_heat_bgr']
        hmap_tran = hmap_tran_l[0]  # in case we had >1 rotations
        imgdir = join(dir_one, f'gctn_dict_{tt}_heat_tran.png')
        saveimg(imgdir, hmap_tran, transpose=True, debugprint=False)

    # Unfortunately we might have to retroactively compute the IoU.
    print('Metric: (1) ... no name for it, (2) cable mask IoU:')
    for t in range(num_acts + 1):
        tt = str(t).zfill(2)
        img_dict = info['img_dict'][t]

        # Pixel Mask IoU
        mask_t = img_dict['mask_img']  # (160,320)
        binary_curr = mask_t > 0
        binary_goal = goal_mask > 0
        pix_inter = np.sum( np.logical_and(binary_curr, binary_goal) )
        pix_union = np.sum( np.logical_or( binary_curr, binary_goal) )
        pix_iou = pix_inter / pix_union

        # Get metrics
        metrics = info['eval_metrics'][t]
        pix_eq = metrics['pix_eq_white']

        # Double check that it's equal. We didn't store this before trial 020.
        if 'cable_mask_iou' in metrics:
            assert metrics['cable_mask_iou'] == pix_iou, \
                '{:0.3f} {:0.3f}'.format(metrics['cable_mask_iou'], pix_iou)

        print(f'  time {tt}: {pix_eq:0.3f}, {pix_iou:0.3f}')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument('--outdir', type=str, default='data')
    p.add_argument('--method', type=str, default='gctn')
    args = p.parse_args()

    # Iterate through directories to inspect.
    targ_dir = join(args.outdir, args.method)
    dirs = sorted(
        [join(targ_dir,x) for x in os.listdir(targ_dir) if 'trial_' in x]
    )

    # We might save images in each dir, though we could have also done this
    # while running `main.py` so there might be some duplication. But this is
    # faster to run as it doesn't involve the robot moving.
    for dir_one_trial in dirs:
        _, tail = os.path.split(dir_one_trial)
        if tail in EXCLUDE:
            print(f'\nSkipping: {dir_one_trial}')
            continue

        # Get the args.json to tell us which goal image we used.
        json_f = [
            join(dir_one_trial,x) for x in os.listdir(dir_one_trial) if 'args.json'==x
        ]
        assert len(json_f) == 1, json_f
        with open(json_f[0], 'rb') as fh:
            args_from_run = json.load(fh)
        goal_idx = args_from_run['goal_idx']

        # Load the goal image mask. Will be binary with 0s and 255s. Needed to
        # compute the IoU retroactively.
        goal_img_path = join(
            DC.GOAL_IMG_DIR, f'goal_{str(goal_idx).zfill(3)}_mask_trip.png'
        )
        assert os.path.exists(goal_img_path), goal_img_path
        goal_mask = cv2.imread(goal_img_path).astype('float')  # need float
        assert goal_mask.shape == (160,320,3), goal_mask.shape
        goal_mask = goal_mask[:, :, 0].astype(np.int32)
        assert len(np.unique(goal_mask)) == 2, goal_mask
        assert np.max(goal_mask) > 1, np.max(goal_mask)

        # Inspect!
        print('')
        print('='*100)
        print(f'Inspecting: {dir_one_trial}, goal_idx: {goal_idx}')
        print('='*100)
        inspect(dir_one_trial, goal_mask)