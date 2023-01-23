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
