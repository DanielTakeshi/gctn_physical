# Physical GCTN Experiments.

Physical experiments for our GCTN paper: https://berkeleyautomation.github.io/bags/

For the simulation code, see https://github.com/DanielTakeshi/deformable-ravens

Disclaimer: a lot of this code depends on our physical robot and some modifications to the GCTN training code. We recommend using the code as a reference, but not actually running it.


## Usage

- Make sure we have a model of GCTN available.
- Make sure `takeshi` is running with a script.
- Make sure we're in our Python 3.6 conda env (I name it `franka`).

Run the following commands:

```
python main.py [args]
```

See that script for more, and `daniel_utils.py` for a bunch of utility methods.

Run this to get demonstrations:

```
python get_demo_data.py
```

See the `scripts/` directory (and the associated README) for testing a bunch of
things.

Directories:

- `data/` is meant for output from different methods.
- `demos/` is meant for saving demonstration data.
- `images/` is meant for debugging `data_collector.py`'s image querying.
- `goals_real_cable_line_notarget/` has the goal images for test time rollouts.
- `real_input/` and `real_output/` are for handling GCTN synchronization.

NOTE: we technically have a termination time step but we are planning to just
use CTRL+C for exiting. This should still save the right amount of data.

The main script will save at:

```
data/gctn/trial_002__2023-02-16_17-57-33:
total 104M
-rw-rw-r-- 1 testseita testseita  246 Feb 16 17:57 args.json
-rw-rw-r-- 1 testseita testseita 817K Feb 16 17:58 c_img_pre_action_time_00.png
-rw-rw-r-- 1 testseita testseita 816K Feb 16 18:00 c_img_pre_action_time_01.png
-rw-rw-r-- 1 testseita testseita 821K Feb 16 18:01 c_img_pre_action_time_02.png
-rw-rw-r-- 1 testseita testseita 816K Feb 16 18:03 c_img_pre_action_time_03.png
-rw-rw-r-- 1 testseita testseita 814K Feb 16 18:05 c_img_pre_action_time_04.png
-rw-rw-r-- 1 testseita testseita 817K Feb 16 18:06 c_img_pre_action_time_05.png
-rw-rw-r-- 1 testseita testseita  99M Feb 16 18:08 trial_info.pkl
```

Then later we might inspect things and populate those directories with more
images.

## My Setup

To get images:

- I use the robot e-stop to move its gripper to a reasonable spot, then `python
  basic_franka_usage.py` to fine-tune it with code. I want a good top-down
  image.
- Then test with `python data_collect.py` to check bounding boxes and cropping
  convention and segmentation masks.
- Then check that the robot can actually reach the extremes of the workspace.

Only once those are done, then I can actually collect 'real' demonstration
data. I do this by hand since it's a LOT faster than having the robot move
there.


## Some random tips from Oliver's students:

From Kevin:

- The issue is that you mounted a kinect on the end-effector which affects the
  weight of the robot. With our impedance controller, we need to set the
  accurate weight of the robot. When setting `use_impedance=False`, you would
  be using the Franka's internal ik and stiffness which are better at
  counteracting small differences in weight

- Yeah so `use_impedance` is set to true by default because in certain joint
  configurations the inbuilt ik doesn't work, but usually the inbuilt ik
  results in smoother motions.
