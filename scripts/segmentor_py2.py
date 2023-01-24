"""Use to get tuned values for segmentation from a GUI. Run in Python 2.

Updated for the cables and Transporters-related experiments.
See my Notion for promising values / segmentation examples.

Also I am using some tape to act as markers but I don't necessarily want the
tape visible.

NOTE: unfortunately I think we have to run a `roslaunch` command first, like
roslaunching rviz to get this working. So this is not meant to be run with
the frankapy code we are using. After my `setupf` alias I did:

testseita@TheCat:~/github/frankapy$ roslaunch bimanual_ros sensor_kinect_rviz_v01.launch

And then in another tab (again, Python 2), running this script works.
"""
import cv2
import rospy
import argparse
import numpy as np
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Segmenter():
    """Based on code from Thomas Weng."""

    def __init__(self, args):
        self.args = args
        rospy.init_node("fg_bg_segmenter")
        self.sub = rospy.Subscriber(args.img_topic, Image, self.cb)
        self.use_bgr = (args.bgr == 1)
        if self.use_bgr:
            self.title_window = 'Fore-Back ground Color Segmentor'
        else:
            self.title_window = 'Fore-Back ground HSV Segmentor'

        # If HSV segmentation.
        self.lower_hue =   0
        self.upper_hue = 255
        self.lower_sat =   0
        self.upper_sat = 255
        self.lower_val =   0
        self.upper_val = 255

        # If BGR segmentation.
        self.lower_B =   0
        self.upper_B = 255
        self.lower_G =   0
        self.upper_G = 255
        self.lower_R =   0
        self.upper_R = 255

        self.bgr = None
        self.hsv = None
        self.dst = None
        self.bridge = CvBridge()
        cv2.namedWindow(self.title_window)

        if self.use_bgr:
            cv2.createTrackbar('lower B', self.title_window,   0, 255, self.on_lower_b)
            cv2.createTrackbar('upper B', self.title_window, 255, 255, self.on_upper_b)
            cv2.createTrackbar('lower G', self.title_window,   0, 255, self.on_lower_g)
            cv2.createTrackbar('upper G', self.title_window, 255, 255, self.on_upper_g)
            cv2.createTrackbar('lower R', self.title_window,   0, 255, self.on_lower_r)
            cv2.createTrackbar('upper R', self.title_window, 255, 255, self.on_upper_r)
        else:
            cv2.createTrackbar('lower hue', self.title_window,   0, 255, self.on_lower_h)
            cv2.createTrackbar('upper hue', self.title_window, 255, 255, self.on_upper_h)
            cv2.createTrackbar('lower sat', self.title_window,   0, 255, self.on_lower_s)
            cv2.createTrackbar('upper sat', self.title_window, 255, 255, self.on_upper_s)
            cv2.createTrackbar('lower val', self.title_window,   0, 255, self.on_lower_v)
            cv2.createTrackbar('upper val', self.title_window, 255, 255, self.on_upper_v)

    def update(self):
        # Keep pixel from 'bgr' image on (white) if the mask is 1 at the pixel.
        if self.use_bgr:
            lower = np.array([self.lower_B, self.lower_G, self.lower_R], dtype='uint8')
            upper = np.array([self.upper_B, self.upper_G, self.upper_R], dtype='uint8')
            mask = cv2.inRange(self.bgr, lower, upper)
            print("lower BGR: {}, upper BGR: {}".format(lower, upper))
        else:
            lower = np.array([self.lower_hue, self.lower_sat, self.lower_val], dtype='uint8')
            upper = np.array([self.upper_hue, self.upper_sat, self.upper_val], dtype='uint8')
            mask = cv2.inRange(self.hsv, lower, upper)
            print("lower HSV: {}, upper HSV: {}".format(lower, upper))
        # kernel = np.ones((9,9),np.uint8)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        self.dst = cv2.bitwise_and(np.stack([mask, mask, mask], axis=2), self.bgr)

    # ============================== BGR ============================== #

    def on_lower_b(self, val):
        self.lower_B = val
        self.update()

    def on_upper_b(self, val):
        self.upper_B = val
        self.update()

    def on_lower_g(self, val):
        self.lower_G = val
        self.update()

    def on_upper_g(self, val):
        self.upper_G = val
        self.update()

    def on_lower_r(self, val):
        self.lower_R = val
        self.update()

    def on_upper_r(self, val):
        self.upper_R = val
        self.update()

    # ============================== HSV ============================== #

    def on_lower_h(self, val):
        self.lower_hue = val
        self.update()

    def on_upper_h(self, val):
        self.upper_hue = val
        self.update()

    def on_lower_s(self, val):
        self.lower_sat = val
        self.update()

    def on_upper_s(self, val):
        self.upper_sat = val
        self.update()

    def on_lower_v(self, val):
        self.lower_val = val
        self.update()

    def on_upper_v(self, val):
        self.upper_val = val
        self.update()

    # ============================== Update window ============================== #

    def cb(self, msg):
        # In BGR mode. If we do `cv2.imwrite(..., im)` we get 'correct' colors.
        im = self.bridge.imgmsg_to_cv2(msg, 'bgr8')

        # Make sure this matches what we are actually using! I'm cropping to 320x160.
        x = 0
        y = 0 + 40
        w = 1280
        h = 720 - (2*40)
        im = im[y:y+h, x:x+w]
        im = cv2.resize(im, (320,160))

        # Both the BGR and HSV. Not sure why Thomas had BGR -> RGB here?
        #self.bgr = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
        self.bgr = im.copy()
        self.hsv = cv2.cvtColor(im, cv2.COLOR_BGR2HSV)

    def spin(self):
        """Continually display the image.

        If self.bgr is in BGR mode (the default) then just play it and it will
        look correctly to our eyes.
        """
        while not rospy.is_shutdown():
            if self.bgr is not None:
                self.update()
            if self.dst is None:
                rospy.sleep(0.1)
            else:
                cv2.imshow(self.title_window, self.dst)
                #cv2.imshow(self.title_window,
                #           cv2.cvtColor(self.dst, cv2.COLOR_BGR2RGB))
                cv2.waitKey(30)


if __name__ == '__main__':
    # NOTE(daniel): both '/k4a/rgb/image_raw' and 'k4a/rgb/image_raw' seem to work.
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_topic', help='ROS topic to subscribe to for RGB image',
        default='/k4a/rgb/image_raw')
    parser.add_argument('--bgr', type=int, help='1 if using BGR, else HSV', default=0)
    args, _ = parser.parse_known_args()

    s = Segmenter(args)
    s.spin()
