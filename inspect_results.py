"""Use this script to inspect any results.

Report any test-time statistics, report qualitatively all the
images (and save to directories for easy viewing), etc.
"""
import os
from os.path import join
import cv2
import pickle
import argparse
import numpy as np
import daniel_config as DC
import daniel_utils as DU


def inspect(dir_one_trial):
    """Inspect one trial's directory.

    Might return some statistics so we can compute method-wide stats.
    Note: lengths of img_dict and eval_metrics should be 1 more than the
    others, but I didn't do this for a few initial trials.
    """
    info_dir = join(dir_one_trial, 'trial_info.pkl')
    with open(info_dir, 'rb') as fh:
        info = pickle.load(fh)
    print(f'Loaded info, keys: {info.keys()}')
    for key in list(info.keys()):
        print(f'key: {key}, len: {len(info[key])}')


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
        print(f'\nInspecting: {dir_one_trial}')
        inspect(dir_one_trial)