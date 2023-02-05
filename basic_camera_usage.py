"""Testing the camera stuff. References:

https://github.com/IFL-CAMP/easy_handeye
https://github.com/Wenxuan-Zhou/frankapy_env/blob/main/frankapy_env/pointcloud.py
https://github.com/DanielTakeshi/mixed-media-physical/blob/main/utils_robot.py

Also remember OpenCV conventions.
https://www.notion.so/Progress-Report-Internal-5e92796ab6a94e66a70a6a77d2bbc4b6
"""
import cv2
import time
import daniel_config as DC
import daniel_utils as DU
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
from data_collect import DataCollector
np.set_printoptions(suppress=True, precision=5, linewidth=150)


# TODO(daniel): not tested / used yet.
def world_to_uv(T_CB, world_coordinate, camera_ns='k4a', debug_print=False):
    """Transform from world coordinates to image pixels.

    Copied this from my 'mixed media' code, but likely have to change.

    From a combination of Carl Qi and SoftGym code.
    See also the `project_to_image` method and my SoftGym code:
    https://github.com/Xingyu-Lin/softagent_rpad/blob/master/VCD/camera_utils.py
    https://github.com/mooey5775/softgym_MM/blob/dev_daniel/softgym/utils/camera_projections.py

    TODO(daniel): should be faster to pre-compute this, right? The transformation
    should not change.

    Parameters
    ----------
    T_CB. Transformation from camera to world (what order?).
    world_coordinate: np.array, shape (n x 3), specifying world coordinates, i.e.,
        we might get from querying the tool EE Position.

    Returns
    -------
    (u,v): specifies (x,y) coords, `u` and `v` are each np.arrays, shape (n,).
        To use it directly with a numpy array such as img[uv], we might have to
        transpose it. Unfortunately I always get confused about the right way.
    """
    matrix_world_to_camera = T_CB.matrix  # TODO check units

    # NOTE(daniel) rest of this is from SoftGym, minus how we get the K matrix.
    world_coordinate = np.concatenate(
        [world_coordinate, np.ones((len(world_coordinate), 1))], axis=1)  # n x 4
    camera_coordinate = np.dot(matrix_world_to_camera, world_coordinate.T)  # 4 x n
    camera_coordinate = camera_coordinate.T  # n x 4  (ignore the last col of 1s)

    if debug_print:
        #print('\nMatrix world to camera')
        #print(matrix_world_to_camera)
        #print('\n(inverse of that matrix)')
        #print(np.linalg.inv(matrix_world_to_camera))
        print('\nWorld coords (source)')
        print(world_coordinate)
        print('\nCamera coords (target), but these need to be converted to pixels')
        print(camera_coordinate)
        print("")

    # Get 4x4 camera intrinsics matrix.
    K = DU.K_matrices[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Convert to ints because we want the pixel coordinates.
    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")
    return (u,v)


# TODO(daniel): not tested / used yet.
def test_world_to_camera():
    """Test going from world coordinates to camera pixels."""
    # TODO(daniel): another exmpale for me to test
    # Camera images. Make sure the roslaunch file 'activates' the camera nodes.
    cimg1, dimg1 = robot.capture_image(args.data_dir, camera_ns='k4a',     filename='k4a')
    cimg2, dimg2 = robot.capture_image(args.data_dir, camera_ns='k4a_top', filename='k4a_top')

    # Get the robot EE position (w.r.t. world) and get the pixels.
    ee_pose = robot.get_ee_pose()
    ee_posi = ee_pose[:3]
    print('Current EE position: {}'.format(ee_posi))

    # Use the top camera, make Nx3 matrix with world coordinates to be converted to camera.
    points_world = np.array([
        [0.0, 0.0, 0.0],  # base of robot (i.e., this is the world center origin)
        [0.1, 0.0, 0.0],  # more in x direction
        [0.2, 0.0, 0.0],  # more in x direction
        [0.3, 0.0, 0.0],  # more in x direction
        [0.4, 0.0, 0.0],  # more in x direction
        [0.5, 0.0, 0.0],  # more in x direction
        [0.6, 0.0, 0.0],  # more in x direction
        [0.7, 0.0, 0.0],  # more in x direction
        [0.8, 0.0, 0.0],  # more in x direction
        [0.9, 0.0, 0.0],  # more in x direction
        [1.0, 0.0, 0.0],  # more in x direction

        [0.5, -0.6, 0.0],  # check y
        [0.5, -0.5, 0.0],  # check y
        [0.5, -0.4, 0.0],  # check y
        [0.5, -0.3, 0.0],  # check y
        [0.5, -0.2, 0.0],  # check y
        [0.5, -0.1, 0.0],  # check y
        [0.5,  0.1, 0.0],  # check y
        [0.5,  0.2, 0.0],  # check y
        [0.5,  0.3, 0.0],  # check y

        [0.6, -0.1, 0.0],  # check z
        [0.6, -0.1, 0.1],  # check z
        [0.6, -0.1, 0.2],  # check z
        [0.6, -0.1, 0.3],  # check z
        [0.6, -0.2, 0.0],  # check z
        [0.6, -0.2, 0.1],  # check z
        [0.6, -0.2, 0.2],  # check z
        [0.6, -0.2, 0.3],  # check z
        [0.6, -0.3, 0.0],  # check z
        [0.6, -0.3, 0.1],  # check z
        [0.6, -0.3, 0.2],  # check z
        [0.6, -0.3, 0.3],  # check z
        [0.6, -0.4, 0.0],  # check z
        [0.6, -0.4, 0.1],  # check z
        [0.6, -0.4, 0.2],  # check z
        [0.6, -0.4, 0.3],  # check z
        [0.6, -0.5, 0.0],  # check z
        [0.6, -0.5, 0.1],  # check z
        [0.6, -0.5, 0.2],  # check z
        [0.6, -0.5, 0.3],  # check z
        [0.6, -0.6, 0.0],  # check z
        [0.6, -0.6, 0.1],  # check z
        [0.6, -0.6, 0.2],  # check z
        [0.6, -0.6, 0.3],  # check z

        [0.667, -0.374, 0.662],  # center of the camera (actually not visible)
    ])

    # Convert to pixels for 'k4a_top'!
    uu,vv = world_to_uv(buffer=robot.buffer,
                        world_coordinate=points_world,
                        camera_ns='k4a_top')

    # Now write over the image. For cv2 we need to reverse (u,v), right?
    # Actually for some reason we don't have to do that ...
    cimg = cimg2.copy()
    for i in range(points_world.shape[0]):
        p_w = points_world[i]
        u, v = uu[i], vv[i]
        print('World: {}  --> Pixels {} {}'.format(p_w, u, v))
        if i < 11:
            color = (0,255,255)
        elif i < 20:
            color = (255,0,0)
        else:
            color = (0,0,255)
        cv2.circle(cimg, center=(u,v), radius=10, color=color, thickness=-1)

    # Compare original vs new. The overlaid points visualize world coordinates.
    print('See image, size {}'.format(cimg.shape))
    cv2.imwrite('original_annotate.png', img=cimg2)
    cv2.imwrite('original.png', img=cimg)


def uv_to_world_pos(T_BC, u, v, z, camera_ns='k4a', debug_print=False):
    """Transform from image coordinates and depth to world coordinates.

    Copied this from my 'mixed media' code, but likely have to change.
    https://github.com/DanielTakeshi/mixed-media-physical/blob/63ce1b2118d77f3757452540feeabd733fb9a9f4/utils_robot.py#L106

    TODO(daniel): can we keep track of the units carefully?
    I think the K matrix uses millimeters, and so does the depth camera.
    But does T_BC need to use millimeters or meters?

    Parameters
    ----------
    T_BC: should transform from camera to world. This was the ordering we had it
        earlier, we created a `matrix_camera_to_world`.
    u, v: image coordinates
    z: depth value

    Returns
    -------
    world coordinates at pixels (u,v) and depth z.
    """
    matrix_camera_to_world = T_BC.matrix  # NOTE(daniel): I think units are OK.

    # Get 4x4 camera intrinsics matrix.
    K = DC.K_matrices[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Will this work? Need to check. From SoftGym, and also they flip u,v here...
    one = np.ones(u.shape)
    x = (v - u0) * z / fx
    y = (u - v0) * z / fy

    # If x,y,z came from scalars, makes (1,4) matrix. Need to test for others.
    cam_coords = np.stack([x, y, z, one], axis=1)

    # TODO(daniel): which one?
    #world_coords = matrix_camera_to_world.dot(cam_coords.T)  # (4,4) x (4,1)
    world_coords = cam_coords.dot(matrix_camera_to_world.T)

    if debug_print:
        # Camera axis has +x pointing to me, +y to wall, +z downwards.
        #print('\nMatrix camera to world')
        #print(matrix_camera_to_world)
        #print('\n(inverse of that matrix)')
        #print(np.linalg.inv(matrix_camera_to_world))
        print('\n(cam_coords before converting to world)')
        print(cam_coords)
        print('')

    return world_coords  # (n,4) but ignore last row


def test_camera_to_world():
    """Test camera and pixel to world stuff.

    Do we need two poses, for (base,EE) and (EE,camera)? Then we combine them?
    I think the former is `fa.get_pose()` and the latter is from calibration.
    Be careful with units. Calibration yaml files and the transformations there
    are in meters, along with `fa.get_pose()` transformations.  But I think the
    camera information matrix puts units in millimeters.

    Careful: when we use the raw pixels these have to be for the original image,
    not the one resized. Si it has to be expressed in the raw image space. See
    the data_collector.py for more details.
    """
    dc = DataCollector()
    print('Started the data collector!')

    # The calibration file, copied from `/<HOME>/.ros/easy_handeye`.
    # I think this gives transformation from EE to camera?
    filename = 'cfg/easy_handeye_eye_on_hand__panda_EE_v04.yaml'  # from 02/05/2022
    T = DU.load_transformation(filename)
    print(f'Loaded transformation from {filename}:\n{T}\n')

    # Also load transformation from the robot base to the EE? I think we need that?
    # But does this require starting up the robot to get the pose? Yeah because it
    # has to depend on that, right?
    fa = FrankaArm()
    T_ee_world = fa.get_pose()
    print(f'T_ee_world:\n{T_ee_world}\n')

    # Might just do this one time?
    print('Translation: {} | Rotation: {}'.format(
        T_ee_world.translation, T_ee_world.quaternion))
    joints = fa.get_joints()
    print('Joints: {}'.format(joints))

    # Combine the transformations? Trying to follow `examples/move_robot.py`
    # and from the debugging that RigidTransform provides.
    T_cam_ee = RigidTransform(
        rotation=T[:3, :3],
        translation=T[:3, 3],
        from_frame='camera',
        to_frame='franka_tool',
    )
    print(f'T_cam_ee:\n{T_cam_ee}\n')

    # I _think_ this seems OK? It passes my sanity checks.
    T_cam_world = T_ee_world * T_cam_ee
    print(f'T_cam_world:\n{T_cam_world}\n')
    T_cam_world.translation *= 1000.0
    print(f'T_cam_world now w/millimeters:\n{T_cam_world}\n')

    # Get the aligned color and depth images.
    time.sleep(0.1)
    img_dict = dc.get_images()
    cimg = img_dict['color_raw']
    dimg = img_dict['depth_raw']
    dimg_proc = img_dict['depth_proc']
    assert cimg is not None
    assert dimg is not None
    assert dimg_proc is not None
    assert cimg.shape == dimg_proc.shape, f'{cimg.shape}, {dimg_proc.shape}'
    cv2.imwrite('cimg.png', cimg)
    cv2.imwrite('dimg.png', dimg_proc)

    # Find pixels we want to use. Also annotate them. Careful, if we pick pixels
    # by the boundary we might end up with something that has zeros. Increasing
    # value of v means moving in the negative y direction wrt the robot base.
    u = np.array([400, 400, 400]).astype(np.int64)
    v = np.array([400, 600, 800]).astype(np.int64)
    z = dimg[u, v]
    print(f'At (u,v), depth:\nu={u}\nv={v})\ndepth={z}')

    # As usual for plotting, swap u and v.
    cv2.circle(dimg_proc, center=(v[0],u[0]), radius=3, color=(255,0,0), thickness=3)
    cv2.circle(dimg_proc, center=(v[1],u[1]), radius=3, color=(0,255,0), thickness=3)
    cv2.circle(dimg_proc, center=(v[2],u[2]), radius=3, color=(255,255,255), thickness=3)
    cv2.imwrite('dimg_coord.png', dimg_proc)

    # Using T_cam_world, and camera info, should convert them to world points.
    pos_world_mm = uv_to_world_pos(T_cam_world, u, v, z, debug_print=False)
    pos_world_meter = pos_world_mm / 1000.
    print(f'\npos_world (mm):\n{pos_world_mm}')
    print(f'div by 1000 (m):\n{pos_world_meter}')

    # Fine-tune the robot positions.

    # Later, have EE go to those positions. Can do pick and place later.
    # TODO
    #DU.pick_only(
    #    fa=fa, pix0=None, img_c=None, z_delta=None,
    #)
    time.sleep(1)

    print('Finished with tests.')


if __name__ == "__main__":
    test_camera_to_world()
