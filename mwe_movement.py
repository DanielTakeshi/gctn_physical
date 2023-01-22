import numpy as np
np.set_printoptions(precision=4)
from frankapy import FrankaArm
from autolab_core import RigidTransform

fa = FrankaArm()
T_ee_world = fa.get_pose()
deg = 120
T_ee_rot = RigidTransform(
    rotation=RigidTransform.z_axis_rotation(np.deg2rad(deg)),
    from_frame='franka_tool', to_frame='franka_tool'
)
T_ee_world_target = T_ee_world * T_ee_rot
fa.goto_pose(T_ee_world_target, use_impedance=False)
