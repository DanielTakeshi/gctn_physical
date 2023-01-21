import time
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
np.set_printoptions(suppress=True, precision=4, linewidth=150)

JOINTS = np.array([-0.2965, -0.5919, 0.0356, -1.9614, 0.0217, 1.3333, -2.5998])


def daniel_testing(fa):
    # read functions
    T_ee_world = fa.get_pose()
    print('Translation: {} | Rotation: {}'.format(T_ee_world.translation, T_ee_world.quaternion))

    # get joint information
    joints = fa.get_joints()
    print('Joints: {}'.format(joints))

    # get EE information
    gripper_width = fa.get_gripper_width()
    print('Gripper width: {:0.6f}'.format(gripper_width))

    # end-effector pose control
    print('Translating now...')
    T_ee_world = fa.get_pose()

    T_ee_world.translation += [0., 0., 0.05]
    fa.goto_pose(T_ee_world)
    joints = fa.get_joints()
    print(f'Finished first translation. Joints: {joints}')
    time.sleep(4)

    T_ee_world.translation -= [0., 0., 0.05]
    fa.goto_pose(T_ee_world)
    joints = fa.get_joints()
    print(f'Finished second translation. Joints: {joints}')


if __name__ == "__main__":
    print('Creating the Franka Arm...')
    fa = FrankaArm()
    print(f'Done creating: {fa}')

    daniel_testing(fa)
    print('Finished with tests.')