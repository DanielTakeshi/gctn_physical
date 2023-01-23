import numpy as np
from frankapy import FrankaArm
from autolab_core import RigidTransform

fa = FrankaArm()
deg = 120

# First rotation.
print(f'First rotation of {deg}...')
T_ee_world = fa.get_pose()
T_ee_rot = RigidTransform(
    rotation=RigidTransform.z_axis_rotation(np.deg2rad(deg)),
    from_frame='franka_tool', to_frame='franka_tool'
)
T_ee_world_target = T_ee_world * T_ee_rot
fa.goto_pose(T_ee_world_target, duration=20)
