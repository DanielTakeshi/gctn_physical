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


class DataCollector:

    def __init__(self, init_node=False, rosnode_name='DataCollector'):
        """Collects images from the ROS topics for the Azure Kinect camera.

        Set init_node=True if running this as a stand-alone script.
        In other cases (e.g., if we call from frankapy code), it should be false.
        """
        self.timestep = 0
        self.record_img = False
        self.debug_print = False

        # Store color and depth images.
        self.c_image = None
        self.c_image_bbox = None
        self.c_image_proc = None
        self.d_image = None
        self.d_image_proc = None
        self.c_recorded = 0
        self.d_recorded = 0
        self.mask_im = None

        # For cropping images to (w,h) BEFORE resizing to (160,320).
        self.crop_x = 0 + 180
        self.crop_y = 0 + 130
        self.crop_w = 1280 - (2*180)
        self.crop_h =  720 - (2*130)
        self.width  = 320
        self.height = 160

        # Get masked images. See 'segmentor.py' scripts.
        self.mask_lo = np.array([ 90,  35,   0], dtype='uint8')
        self.mask_up = np.array([255, 255, 255], dtype='uint8')
        self.mask_im = None

        if init_node:
            rospy.init_node(rosnode_name, anonymous=True)

        # "Depth to RGB". Don't do RGB to Depth, produces lots of empty space.
        # TODO(daniel) can double check these topics but they seem OK to me.
        rospy.Subscriber('/k4a/rgb/image_raw',
                Image, self.color_cb, queue_size=1)
        rospy.Subscriber('/k4a/depth_to_rgb/image_raw',
                Image, self.depth_cb, queue_size=1)

    def color_cb(self, msg):
        """Callback for the color images.

        From trying both 'rgb8' and 'bgr8', the 'bgr8' mode seems to be OK.
        """
        if rospy.is_shutdown():
            return

        # Make this change to get it working in Python3.
        #self.c_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        im = ros_numpy.numpify(msg)
        self.c_image = im[:, :, :3].astype(np.uint8)
        self.c_image_bbox = self._process_color(self.c_image, bbox_no_crop=True)
        self.c_image_crop = self._process_color(self.c_image, bbox_no_crop=False)

        # Segment the cable, using c_image_crop (we might re-name...).
        self.hsv = cv2.cvtColor(self.c_image_crop, cv2.COLOR_BGR2HSV)
        self.mask_im = cv2.inRange(self.hsv, self.mask_lo, self.mask_up)

    def depth_cb(self, msg):
        """Callback for the depth image.

        We use a ROS topic that makes the depth image into the same coordinate space
        as the RGB image, so the depth at pixel (x,y) should correspond to the 'depth'
        of the pixel (x,y) in the RGB image.

        For encoding (32FC1): a 32-bit float (32F) with a single channel (C1).
        """
        if rospy.is_shutdown():
            return

        # Make this change to get it working in Python3.
        #self.d_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="32FC1")
        self.d_image = ros_numpy.numpify(msg).astype(np.float32)
        self.d_image_proc = DU.process_depth(self.d_image)  # not cropped
        self.d_image_proc_crop = self._process_depth(self.d_image)  # cropped

    def start_recording(self):
        self.record_img = True

    def stop_recording(self):
        self.record_img = False

    def _process_color(self, cimg, bbox_no_crop=False):
        """Process the color image. Don't make this too computationally heavy.

        Currently we're going to support bounding boxes (for visualization) but
        normally we want to crop it and then resize. I think cropping first is
        critical to a 1:2 ratio, then we can resize. Tune the crop parameters.
        The starting RGB image shape is (720,1280,3). As of 01/27, cropping to
        (460,920,3).
        """
        if bbox_no_crop:
            cimg = self.put_bbox_on_img(
                cimg, x=self.crop_x, y=self.crop_y, w=self.crop_w, h=self.crop_h
            )
        else:
            cimg_crop = self.crop_img(
                cimg, x=self.crop_x, y=self.crop_y, w=self.crop_w, h=self.crop_h
            )
            cimg = cv2.resize(cimg_crop, (self.width, self.height))
        return cimg

    def _process_depth(self, dimg):
        """Process the depth image. Don't make this too computationally heavy.

        If cropping before processing, follow the same procedure for color images.
        """
        dimg_crop = self.crop_img(
            dimg, x=self.crop_x, y=self.crop_y, w=self.crop_w, h=self.crop_h
        )
        dimg_crop = DU.process_depth(dimg_crop)
        dimg = cv2.resize(dimg_crop, (self.width, self.height))
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
        """Crop the image. See documentation for bounding box.

        The y and x are the starting points for cropping.
        """
        new_img = img[y:y+h, x:x+w]
        return new_img

    def get_images(self):
        """Helps to clarify which images mean what."""
        images = dict(
            color_raw=self.c_image,
            depth_raw=self.d_image,
            depth_proc=self.d_image_proc,
        )
        return images


if __name__ == "__main__":
    # Create DC, wait a few seconds for it to start.
    dc = DataCollector(init_node=True)
    print('Created the Data Collector!')
    time.sleep(2)

    if not os.path.exists(SAVEDIR):
        os.mkdir(SAVEDIR)

    # This works in Python3. :)
    for t in range(5):
        time.sleep(1)
        print('\nTime t={}'.format(t))
        tt = str(t).zfill(3)

        if dc.c_image is not None:
            print('DC image (color):\n{}'.format(dc.c_image.shape))
            fname = join(SAVEDIR, f'img_{tt}_color.png')
            cv2.imwrite(fname, dc.c_image)
            fname = join(SAVEDIR, f'img_{tt}_color_bbox.png')
            cv2.imwrite(fname, dc.c_image_bbox)
            fname = join(SAVEDIR, f'img_{tt}_color_crop.png')
            cv2.imwrite(fname, dc.c_image_crop)

        if dc.d_image is not None:
            print('DC image (depth):\n{}'.format(dc.d_image.shape))
            fname = join(SAVEDIR, f'img_{tt}_depth_proc.png')
            cv2.imwrite(fname, dc.d_image_proc)
            fname = join(SAVEDIR, f'img_{tt}_depth_crop.png')
            cv2.imwrite(fname, dc.d_image_proc_crop)
