"""A class which polls ROS topics to get images.

Meant to be called from other code. This is a standalone thing which
obtains color and depth images. Various references:

https://github.com/sjyk/my-new-project
https://github.com/DanielTakeshi/debridement-code
https://github.com/DanielTakeshi/IL_ROS_HSR/blob/master/src/il_ros_hsr/core/sensors.py
https://github.com/DanielTakeshi/mixed-media-physical/blob/main/utils_robot.py
https://github.com/iamlab-cmu/perception-utils/blob/master/perception_utils/kinect.py

Other notes:
http://wiki.ros.org/cv_bridge
https://github.com/microsoft/Azure_Kinect_ROS_Driver/blob/melodic/docs/usage.md
https://github.com/eric-wieser/ros_numpy

A few notes on how this code works:

- (TODO) It's designed to be run in parallel with the SawyerRobot() class. In rospy as soon
  as we call `rospy.Subsriber()`, it spins off another thread.
- From my old ROS question: https://answers.ros.org/question/277209/
  Generally it seems good practice to NOT do much image processing in callbacks.
- (TODO) It depends on what ROS topics we use, but for using color (rgb/image_rect_raw) and
  depth (depth/image_raw), they are NOT pixel aligned and have these sizes and types.
    color: (1536, 2048, 3), dtype('uint8')
    depth: (1024, 1024),    dtype('<u2')
  Thus, to make sure they are in the same space, we use `depth_to_rgb`. From GitHub:
    The depth image, transformed into the color camera co-ordinate space by the
    Azure Kinect Sensor SDK.  This image has been resized to match the color
    image resolution. Note that since the depth image is now transformed into
    the color camera co-ordinate space, some depth information may have been
    discarded if it was not visible to the depth camera.
- It works with Python3.
    https://github.com/ros-perception/vision_opencv/issues/207

I have to call my launcher first to get the Kinect topics. For example:
    testseita@TheCat:~/github/frankapy$ roslaunch bimanual_ros sensor_kinect_v01.launch
This is in a Python2 terminal window. Then, I can call this in Python3.
We can run this stand-alone main method to pick a reasonable camera pose before
doing any frankapy stuff.
"""
import os
from os.path import join
import cv2
import time
import numpy as np
import daniel_utils as DU
np.set_printoptions(suppress=True, precision=4, linewidth=150)
import rospy
import ros_numpy
from sensor_msgs.msg import Image
SAVEDIR = 'images/'


# For cropping images.
CROP_X = 840
CROP_Y = 450
CROP_W = 300
CROP_H = 300


class DataCollector:

    def __init__(self):
        self.timestep = 0
        self.record_img = False
        self.record_pcl = False
        self.debug_print = False

        # Store color and depth images.
        self.c_image = None
        self.c_image_proc = None
        self.c_image_l = []
        self.c_image_proc_l = []
        self.d_image = None
        self.d_image_proc = None
        self.d_image_l = []
        self.d_image_proc_l = []

        # Point clouds (segmented, subsampled)? FlowBot3D used 1200, Dough used 1000.
        self.max_pts = 5000
        self.pcl = None
        self.pcl_l = []

        # Storing other info. Here, `_b` means w.r.t. the 'base'.
        self.ee_poses_b = []
        self.tool_poses_b = []

        ## For cropped images. The w,h indicate width,height of cropped images.
        self.crop_x = CROP_X
        self.crop_y = CROP_Y
        self.crop_w = CROP_W
        self.crop_h = CROP_H

        # Segment the items. If using color, in BGR mode (not RGB) but HSV seems
        # better. See segmentory.py for more details.
        self.targ_lo = np.array([ 15,  70, 170], dtype='uint8')
        self.targ_up = np.array([ 60, 255, 255], dtype='uint8')
        self.targ_mask = None
        self.targ_mask_l = []

        self.dist_lo = np.array([ 70,  70,  70], dtype='uint8')
        self.dist_up = np.array([155, 230, 255], dtype='uint8')
        self.dist_mask = None
        self.dist_mask_l = []

        self.tool_lo = np.array([  0,   0,   0], dtype='uint8')
        self.tool_up = np.array([255, 255,  45], dtype='uint8')
        self.tool_mask = None
        self.tool_mask_l = []

        self.area_lo = np.array([  0,  70,   0], dtype='uint8')
        self.area_up = np.array([255, 255, 255], dtype='uint8')
        self.area_mask = None
        self.area_mask_l = []

        # "Depth to RGB". Don't do RGB to Depth, produces lots of empty space.
        # TODO(daniel) I don't have image_rect_color, I only have image_raw?
        # Is there a difference between image_raw and image_rect_color?
        # It must be because of my launch file, look at what I launched.
        rospy.Subscriber('/k4a/rgb/image_raw',
                Image, self.color_image_callback, queue_size=1)
        rospy.Subscriber('/k4a/depth_to_rgb/image_raw',
                Image, self.depth_image_callback, queue_size=1)

    def color_image_callback(self, msg):
        """If `self.record`, then this saves the color images.

        Also amazingly, just calling the same ee pose code seems to work?
        Careful, this might decrease the rate that images get called, we don't
        want much computation here. Profile it?

        From trying both 'rgb8' and 'bgr8', and saving with `cv2.imwrite(...)`, the
        'bgr8' mode seems to preserve colors as we see it.
        """
        if rospy.is_shutdown():
            return

        #self.c_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        im = ros_numpy.numpify(msg)
        self.c_image = im[:, :, :3].astype(np.uint8)
        self.c_image_proc = self._process_color(self.c_image)

        # Segment the target(s) and distractor(s).
        self.hsv = cv2.cvtColor(self.c_image_proc, cv2.COLOR_BGR2HSV)
        self.targ_mask = cv2.inRange(self.hsv, self.targ_lo, self.targ_up)
        self.dist_mask = cv2.inRange(self.hsv, self.dist_lo, self.dist_up)
        self.tool_mask = cv2.inRange(self.hsv, self.tool_lo, self.tool_up)
        self.area_mask = cv2.inRange(self.hsv, self.area_lo, self.area_up)

        if self.record_img:
            if self.debug_print:
                rospy.loginfo('New color image, total: {}'.format(len(self.c_image_l)))
            self.c_image_l.append(self.c_image)
            self.c_image_proc_l.append(self.c_image_proc)
            #self.targ_mask_l.append(self.targ_mask)
            #self.dist_mask_l.append(self.dist_mask)
            #self.tool_mask_l.append(self.tool_mask)
            #self.area_mask_l.append(self.area_mask)

    def depth_image_callback(self, msg):
        """Callback for the depth image.

        We use a ROS topic that makes the depth image into the same coordinate space
        as the RGB image, so the depth at pixel (x,y) should correspond to the 'depth'
        of the pixel (x,y) in the RGB image.

        For encoding (32FC1): a 32-bit float (32F) with a single channel (C1).
        """
        if rospy.is_shutdown():
            return

        #self.d_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        self.d_image = ros_numpy.numpify(msg).astype(np.float32)
        self.d_image_proc = self._process_depth(self.d_image)  # not cropped

        if self.record_img:
            # Record the depth image. Not sure if we need the processed one?
            if self.debug_print:
                rospy.loginfo('New depth image, total: {}'.format(len(self.d_image_l)))
            self.d_image_l.append(self.d_image)
            self.d_image_proc_l.append(self.d_image_proc)

    def start_recording(self):
        self.record_img = True

    def stop_recording(self):
        self.record_img = False

    def get_color_image(self):
        return self.c_image

    def get_color_images(self):
        return self.c_image_l

    def get_depth_image(self):
        return self.d_image

    def get_depth_image_proc(self):
        return self.d_image_proc

    def get_depth_images(self):
        return self.d_image_l

    def get_ee_poses(self):
        return self.ee_poses_b

    def get_tool_poses(self):
        return self.tool_poses_b

    def _process_color(self, cimg):
        """Process the color image. Don't make this too computationally heavy."""
        cimg_crop = self.crop_img(cimg,
            x=self.crop_x, y=self.crop_y, w=self.crop_w, h=self.crop_h)
        return cimg_crop

    def _process_depth(self, dimg):
        """Process the depth image. Don't make this too computationally heavy."""
        return DU.process_depth(dimg)
        return dimg

    def put_bbox_on_img(self, img, x, y, w, h):
        """Test bounding boxes on images.

        When visualizing images (e.g., with `eog cimg.png`) coordinates (x,y) start at
        (0,0) in the upper LEFT corner. Increasing x moves RIGHTWARD, but increasing y
        moves DOWNWARD. The w is for the width (horizontal length) of the image.

        We use (x,y) here and use cropping code that does y first, then x. This means
        the bounding box we visualize can be interpreted as how the image is cropped.
        """
        new_img = img.copy()
        cv2.rectangle(new_img, (x,y), (x+w, y+h), (0,255,0), 3)
        return new_img

    def crop_img(self, img, x, y, w, h):
        """Crop the image. See documentation for bounding box."""
        new_img = img[y:y+h, x:x+w]
        return new_img


if __name__ == "__main__":
    dc = DataCollector()
    print('Created the Data Collector!')

    if not os.path.exists(SAVEDIR):
        os.mkdir(SAVEDIR)

    # This works in Python3. :)
    for t in range(5):
        time.sleep(1)
        print('\nTime t={}'.format(t))
        if dc.c_image is not None:
            print('DC image (rgb):\n{}'.format(dc.c_image.shape))
            fname = join(SAVEDIR, f'img_{str(t).zfill(3)}_color.png')
            cv2.imwrite(fname, dc.c_image)
        if dc.d_image is not None:
            print('DC image (depth):\n{}'.format(dc.d_image.shape))
            fname = join(SAVEDIR, f'img_{str(t).zfill(3)}_depthproc.png')
            cv2.imwrite(fname, dc.d_image_proc)
