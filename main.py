"""Our main experiment script. This should be for rollouts."""
import os
from os.path import join
import argparse
import time
import numpy as np
from frankapy import FrankaArm
import daniel_utils as DU
np.set_printoptions(suppress=True, precision=4, linewidth=150)

SAVEDIR = 'logs/'


def save_stuff():
    """TODO"""
    pass


def run_trial(fa):
    """TODO"""
    pass


if __name__ == "__main__":
    print('Running experiments')
    fa = FrankaArm()

    # TODO(daniel)