"""Basic movement tests.

Note that use_impedance=False is necessary since we mounted a camera.
Well, not strictly necessary but I think it makes more sense in our case.
"""
import numpy as np
np.set_printoptions(precision=4)
from frankapy import FrankaArm


if __name__ == "__main__":
    print('Creating the Franka Arm...')
    fa = FrankaArm()

    # Starting pose.
    T_ee_world = fa.get_pose()
    print('Starting pose:\n\tTranslation: {} | Rotation: {}'.format(
        T_ee_world.translation, T_ee_world.quaternion))

    # Translate z upwards.
    T_ee_world.translation += [0., 0., 0.05]
    fa.goto_pose(T_ee_world, use_impedance=False)

    T_ee_world = fa.get_pose()
    print('Pose after translation.\n\tTranslation: {} | Rotation: {}'.format(
        T_ee_world.translation, T_ee_world.quaternion))
