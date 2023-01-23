"""Utilities file to support GCTN experiments.

This is meant to be imported in our `main.py` script which runs experiments.
"""
import numpy as np
from autolab_core import RigidTransform
np.set_printoptions(suppress=True, precision=4, linewidth=150)


def rotate_EE_one_axis(fa, deg, axis, use_impedance=False, duration=12):
    """Just do one EE axis rotation.

    Empirically, seems like abs(deg) has to be at least 4 to observe any
    notable robot movement. Also, I'd increase the duration from the
    default of 3.

    Default duration
    """
    assert np.abs(deg) <= 145., f'deg: {deg}, careful that is a lot...'
    print(f'Rotation in EE frame by {deg} deg, axis: {axis}.')
    T_ee_world = fa.get_pose()

    if axis == 'x':
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.x_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
    elif axis == 'y':
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.y_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
    elif axis == 'z':
        T_ee_rot = RigidTransform(
            rotation=RigidTransform.z_axis_rotation(np.deg2rad(deg)),
            from_frame='franka_tool', to_frame='franka_tool'
        )
    else:
        raise ValueError(axis)

    T_ee_world_target = T_ee_world * T_ee_rot
    fa.goto_pose(
        T_ee_world_target, duration=duration, use_impedance=use_impedance
    )


def pick_and_place():
    """Our pick and place action primitive."""
    pass


if __name__ == "__main__":
    pass
