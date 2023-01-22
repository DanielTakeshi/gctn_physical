# gctn_physical

Physical GCTN


Some random tips from Oliver's students:

From Kevin:

- The issue is that you mounted a kinect on the end-effector which affects the
  weight of the robot. With our impedance controller, we need to set the
  accurate weight of the robot. When setting `use_impedance=False`, you would
  be using the Franka's internal ik and stiffness which are better at
  counteracting small differences in weight

- Yeah so `use_impedance` is set to true by default because in certain joint
  configurations the inbuilt ik doesn't work, but usually the inbuilt ik
  results in smoother motions.
