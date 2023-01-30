import numpy as np
from frankapy import FrankaArm
from autolab_core import RigidTransform

fa = FrankaArm()
# deg = 120
#
# # First rotation.
# print(f'First rotation of {deg}...')
# T_ee_world = fa.get_pose()
# T_ee_rot = RigidTransform(
#     rotation=RigidTransform.z_axis_rotation(np.deg2rad(deg)),
#     from_frame='franka_tool', to_frame='franka_tool'
# )
# T_ee_world_target = T_ee_world * T_ee_rot
# fa.goto_pose(T_ee_world_target, duration=20)

#print(fa.get_joints())
#print(fa.get_pose())

# If this isn't working, restart the machine.
# Note: if we are grasping something, set `grasp=True`.
print('Closing...')
fa.goto_gripper(width=0.01, grasp=True, force=10.)
print('Opening...')
fa.goto_gripper(width=0.05)

#fa.goto_joints(
#    np.array([-0.2623,  0.0315, 0.1403, -2.6996, -0.0661, 2.7208, -0.7851])
#)
#
#print(fa.get_joints())