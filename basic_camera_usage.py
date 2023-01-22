"""
Testing the camera stuff. References:

https://github.com/IFL-CAMP/easy_handeye
https://github.com/Wenxuan-Zhou/frankapy_env/blob/main/frankapy_env/pointcloud.py
https://github.com/DanielTakeshi/mixed-media-physical/blob/main/utils_robot.py
"""
import time
import daniel_utils as DU
import yaml
import open3d as o3d
import numpy as np
from autolab_core import RigidTransform
from frankapy import FrankaArm
np.set_printoptions(suppress=True, precision=4, linewidth=150)

# NOTE(daniel): these work without errors.
import rospy
from geometry_msgs.msg import Point
from geometry_msgs.msg import Pose
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Quaternion
from geometry_msgs.msg import Transform
from geometry_msgs.msg import TransformStamped
from geometry_msgs.msg import Vector3

# NOTE(daniel): these error due to something related to Python 2.
# I'm guessing we should not use these.
#import tf2_ros
#import tf.transformations as tr

# ---------------------------------------------------------------------------------- #
# Home joints and pose (default position to get images?).
# Get these by moving the robot (with e-stop on) to a 'reasonable' spot to get images.
# Will require some trial and error. FYI, it seems to be reliable in getting to the
# desired poses, down to about millimeter level precision w.r.t translation, and about
# 0.001 precision (in radians) for the joints (I think all are revolute). Nice...
#
# Possible home:
# Translation: [ 0.333  -0.107   0.6424] | Rotation: [-0.0291  0.029   0.9991 -0.01  ]
# Joints: [-0.5896 -0.4637  0.1914 -1.6885  0.0481  1.1862 -2.7011]
#
# Possible for lowering+testing rotations:
# Translation: [ 0.324  -0.1048  0.2282] | Rotation: [-0.0047  0.0177  0.9998 -0.0077]
# Joints: [-0.4803 -0.503   0.1917 -2.8445  0.104   2.3419 -2.7057]
# ---------------------------------------------------------------------------------- #
EE_HOME = np.array([0.333, -0.107, 0.6424, -0.0291, 0.029, 0.9991, -0.01])
JOINTS_HOME = np.array([-0.5896, -0.4637, 0.1914, -1.6885, 0.0481, 1.1862, -2.7011])

EE_GRIP = np.array([0.324, -0.1048, 0.2282, -0.0047, 0.0177, 0.9998, -0.0077])
JOINTS_GRIP = np.array([-0.4803, -0.503, 0.1917, -2.8445, 0.104, 2.3419, -2.7057])
# ---------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------- #
# https://github.com/DanielTakeshi/mixed-media-physical/blob/main/config.py
# Get from `rostopic echo /k4a_top/rgb/camera_info`. These are the _intrinsics_,
# which are just given to us (we don't need calibration for these).
# TODO(daniel): how do we get these in our case now? Do we have to do a roslaunch
# command separately from these?
K_matrices = {
    'k4a': np.array([
        [977.870,     0.0, 1022.401],
        [    0.0, 977.865,  780.697],
        [    0.0,     0.0,      1.0]
    ]),
    'k4a_top': np.array([
        [977.005,     0.0, 1020.287],
        [    0.0, 976.642,  782.864],
        [    0.0,     0.0,      1.0]
    ]),
}
# ---------------------------------------------------------------------------------- #


# ---------------------------------------------------------------------------------------- #
# https://answers.ros.org/question/332407/transformstamped-to-transformation-matrix-python/
# ---------------------------------------------------------------------------------------- #

def pose_to_pq(msg):
    """Convert a C{geometry_msgs/Pose} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.position.x, msg.position.y, msg.position.z])
    q = np.array([msg.orientation.x, msg.orientation.y,
                  msg.orientation.z, msg.orientation.w])
    return p, q


def pose_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/PoseStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return pose_to_pq(msg.pose)


def transform_to_pq(msg):
    """Convert a C{geometry_msgs/Transform} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    p = np.array([msg.translation.x, msg.translation.y, msg.translation.z])
    q = np.array([msg.rotation.x, msg.rotation.y,
                  msg.rotation.z, msg.rotation.w])
    return p, q


def transform_stamped_to_pq(msg):
    """Convert a C{geometry_msgs/TransformStamped} into position/quaternion np arrays

    @param msg: ROS message to be converted
    @return:
      - p: position as a np.array
      - q: quaternion as a numpy array (order = [x,y,z,w])
    """
    return transform_to_pq(msg.transform)


def msg_to_se3(msg):
    """Conversion from geometric ROS messages into SE(3)

    @param msg: Message to transform. Acceptable types - C{geometry_msgs/Pose}, C{geometry_msgs/PoseStamped},
    C{geometry_msgs/Transform}, or C{geometry_msgs/TransformStamped}
    @return: a 4x4 SE(3) matrix as a numpy array
    @note: Throws TypeError if we receive an incorrect type.
    """
    if isinstance(msg, Pose):
        p, q = pose_to_pq(msg)
    elif isinstance(msg, PoseStamped):
        p, q = pose_stamped_to_pq(msg)
    elif isinstance(msg, Transform):
        p, q = transform_to_pq(msg)
    elif isinstance(msg, TransformStamped):
        p, q = transform_stamped_to_pq(msg)
    else:
        raise TypeError("Invalid type for conversion to SE(3)")
    norm = np.linalg.norm(q)
    if np.abs(norm - 1.0) > 1e-3:
        raise ValueError(
            "Received un-normalized quaternion (q = {0:s} ||q|| = {1:3.6f})".format(
                str(q), np.linalg.norm(q)))
    elif np.abs(norm - 1.0) > 1e-6:
        q = q / norm
    g = tr.quaternion_matrix(q)
    g[0:3, -1] = p
    return g


def uv_to_world_pos(buffer, u, v, z, camera_ns, debug_print=False):
    """Transform from image coordinates and depth to world coordinates.

    Copied this from my 'mixed media' code, but likely have to change.

    Parameters
    ----------
    u, v: image coordinates
    z: depth value
    camera_params:

    Returns
    -------
    world coordinates at pixels (u,v) and depth z.
    """

    # We name this as T_BC so that we go from camera to base.
    while not rospy.is_shutdown():
        try:
            if camera_ns == "k4a":
                T_BC = buffer.lookup_transform(
                        'base', 'rgb_camera_link', rospy.Time(0))
            elif camera_ns == "k4a_top":
                T_BC = buffer.lookup_transform(
                        'base', 'top_rgb_camera_link', rospy.Time(0))
            #print("Transformation, Camera -> Base:\n{}".format(T_BC))
            break
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            continue

    # Convert this to a 4x4 homogeneous matrix (borrowed code from ROS answers).
    matrix_camera_to_world = msg_to_se3(T_BC)

    # Get 4x4 camera intrinsics matrix.
    K = K_matrices[camera_ns]
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
    #world_coords = matrix_camera_to_world.dot(cam_coords.T)  # (4,4) x (4,1)
    #print(world_coords)
    # TODO(daniel) check filter_points, see if this is equivalent.
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


def world_to_uv(buffer, world_coordinate, camera_ns, debug_print=False):
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
    buffer: from ROS, so that we can look up camera transformations.
    world_coordinate: np.array, shape (n x 3), specifying world coordinates, i.e.,
        we might get from querying the tool EE Position.
    Returns
    -------
    (u,v): specifies (x,y) coords, `u` and `v` are each np.arrays, shape (n,).
        To use it directly with a numpy array such as img[uv], we might have to
        transpose it. Unfortunately I always get confused about the right way.
    """

    # We name this as T_CB so that we go from base to camera.
    while not rospy.is_shutdown():
        try:
            if camera_ns == "k4a":
                T_CB = buffer.lookup_transform(
                        'rgb_camera_link', 'base', rospy.Time(0))
            elif camera_ns == "k4a_top":
                T_CB = buffer.lookup_transform(
                        'top_rgb_camera_link', 'base', rospy.Time(0))
            #print("Transformation, Base -> Camera:\n{}".format(T_CB))
            break
        except (tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException) as e:
            continue

    # Convert this to a 4x4 homogeneous matrix (borrowed code from ROS answers).
    matrix_world_to_camera = msg_to_se3(T_CB)

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
    K = K_matrices[camera_ns]
    u0 = K[0, 2]
    v0 = K[1, 2]
    fx = K[0, 0]
    fy = K[1, 1]

    # Convert to ints because we want the pixel coordinates.
    x, y, depth = camera_coordinate[:, 0], camera_coordinate[:, 1], camera_coordinate[:, 2]
    u = (x * fx / depth + u0).astype("int")
    v = (y * fy / depth + v0).astype("int")
    return (u,v)


def load_transformation(filename):
    """
    Load camera calibration (from Wenxuan).
    """
    if '.yaml' in filename:
        calibration = yaml.load(open(filename, 'rb'), Loader=yaml.FullLoader)
        calibration = calibration['transformation']
        trans = np.array([calibration['x'], calibration['y'], calibration['z']])
        quat = np.array([calibration['qw'], calibration['qx'], calibration['qy'], calibration['qz']])
        R = o3d.geometry.get_rotation_matrix_from_quaternion(quat)
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = trans
        transformation = T
    elif '.npy' in filename:
        with open(filename, 'rb') as f:
            transformation = np.load(f)
    return transformation


def camera_stuff():
    """How do we make use of a camera?

    Do we need two poses, for (base,EE) and (EE,camera)? Then we combine them?
    I think the former is `fa.get_pose()` and the latter is from calibration.
    """

    # The calibration file; copy it from `/<HOME>/.ros/easy_handeye`.
    # I think this gives transformation from EE to camera?
    filename = 'cfg/easy_handeye_eye_on_hand.yaml'
    T = load_transformation(filename)
    print(f'Loaded transformation from {filename}:\n{T}\n')

    # Also load transformation from the robot base to the EE? I think we need that?
    # But does this require starting up the robot to get the pose? Yeah because it
    # has to depend on that, right?
    fa = FrankaArm()
    T_ee_world = fa.get_pose()
    print(f'T_ee_world:\n{T_ee_world}\n')

    # Wait, we might not want the tool offset since the calibration we did does NOT
    # consider that. Unfortunately this seems to be the same as earlier since I
    # don't think we have set it anywhere?
    T_ee_world_nooff = fa.get_pose(include_tool_offset=False)
    print(f'T_ee_world_nooff:\n{T_ee_world_nooff}\n')

    # This is the identity matrix...
    T_toolbasepose = fa.get_tool_base_pose()
    print(f'fa.get_tool_base_pose():\n{T_toolbasepose}\n')

    # Combine the transformations? Trying to do this following examples/move_robot.py
    # and from the debugging that RigidTransform provides.
    T_cam_ee = RigidTransform(
        rotation=T[:3, :3],
        translation=T[:3, 3],
        from_frame='camera',
        to_frame='franka_tool',
    )
    print(f'T_cam_ee:\n{T_cam_ee}\n')

    # This is _almost_ there! But I think there is something missing with the
    # tool offset that's causing the issue. OR Maybe I can re-run calibration but
    # with a different pose which more closely matches the tool pose?
    T_cam_world = T_ee_world * T_cam_ee
    print(f'T_cam_world:\n{T_cam_world}\n')

    ## OK now let's see if we can get the image. Nope, not working.
    ## I think we have to use another setting.
    ## NOTE(daniel): I had to install this. Not sure why VSCode complains.
    #from perception_utils.kinect import KinectSensorBridged
    #print('Creating KinectSensorBridged...')
    #ks = KinectSensorBridged()
    #print('starting KinectSensorBridged...')
    #ks.start()
    #print('Done starting...')

    # TODO

    # Find pixels we want to use.
    # TODO

    # Using T_cam_world, and camera info, should convert them to EE poses.
    # TODO

    # Then go to the pose? Can do pick-and-place later.
    # TODO
    time.sleep(3)


if __name__ == "__main__":
    camera_stuff()
    print('Finished with tests.')