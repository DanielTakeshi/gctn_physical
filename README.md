# Physical GCTN Experiments.

## Usage

- Make sure we have a model of GCTN available.
- Make sure `takeshi` is running with a script.
- Make sure we're in our Python 3.6 conda env (I name it `franka`).

Run the following commands:

```
python main.py [args]
```

See that script for more, and `daniel_utils.py` for a bunch of utility methods.


See the `scripts/` directory (and the associated README) for testing a bunch of
things.

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
