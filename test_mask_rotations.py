"""Use this script to test rotations.

Basically when we download images we need to make sure that we are
using appropriate rotations, conditioned on pick place and fitting a
line to the cable portions.

So far it seems promising. I need to then put the simplified code
into our utilities script so we can just query it on demand.
"""
import os
import cv2
import numpy as np
import daniel_utils as DU
RED = (255,0,0)
GREEN = (0,255,0)
BLUE = (0,0,255)


def get_pix_and_rotation(mask):
    """Get rotation from image and picking point.

    First let's concern ourselves with just getting the right rotation
    on the image. Then we worry about incorporating our z-offset.

    A lot of images / tests here ... we'll simplify for the 'real' code.

    Parameters
    ----------
    mask, image with type uint8, shape (160,320,3), max val should be 255.
    """
    mask_prob = mask[:, :, 0]

    # Test with a lot of samples by re-running this script, but keep 1 sample.
    pix = DU.sample_distribution(prob=mask_prob, n_samples=1)
    wname = f'pix: {pix}'
    mask_copy = np.copy(mask)

    # For bounding box stuff.
    p0 = pix[0]
    p1 = pix[1]
    ss = 10

    # ------------------------- visualize pick+box --------------------------- #
    # Circle the picking point.
    cv2.circle(mask_copy, center=(pix[1],pix[0]), radius=5, color=RED, thickness=2)
    cv2.putText(
        img=mask_copy,
        text="{}".format(pix),
        org=(10, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=GREEN,
        thickness=1
    )

    # Put a bounding box around the picking point. Blue with BGR imgs. If it's
    # close to the boundary, then some of the bounding box might not be visible.
    cv2.rectangle(mask_copy, (p1-ss,p0-ss), (p1+ss, p0+ss), BLUE, 2)

    # Make it consistent with viewing and saving.
    mask_copy = DU.rgb_to_bgr(mask_copy)

    # If we just want to view the picking point.
    cv2.imshow(wname, mask_copy)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('mask_pick_and_crop.jpg', mask_copy)
    cv2.destroyAllWindows()
    # ------------------------------------------------------------------------ #

    # ---------------------- Contours full image ----------------------------- #
    # Detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
    contours, hierarchy = cv2.findContours(
        image=mask_prob, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    # Draw contours.
    image_copy = mask.copy()
    cv2.drawContours(
        image=image_copy,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )

    # Show contours on the full image. This is detected quite nicely, BTW.
    cv2.imshow('mask_contours_full', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('mask_contours_full.jpg', image_copy)
    cv2.destroyAllWindows()
    # ------------------------------------------------------------------------ #

    # ---------------------- Contours crop image ----------------------------- #
    ## Maybe try and find contours on a smaller crop? First, we should pad a copy
    ## of these images so that we are guaranteed a crop of the right size.
    #mask_prob_pad = np.pad(mask_prob.copy(), ((ss,ss), (ss,ss)), constant_values=0)
    #mask_pad      = np.pad(mask.copy(), ((ss,ss), (ss,ss), (0,0)), constant_values=0)
    # Or do we? Maybe this is over-complicating things?

    # Now crop and get exactly desired image sizes (with black as padded region).
    mask_prob_crop = mask_prob[p0-ss:p0+ss, p1-ss:p1+ss]  # grayscale
    mask_crop      =      mask[p0-ss:p0+ss, p1-ss:p1+ss]  # 'rgb'

    # It could be smaller due to cropping near the boundary of images.
    # assert mask_prob_crop.shape == (ss*2,ss*2), mask_prob_crop.shape
    # assert mask_crop.shape == (ss*2,ss*2,3), mask_crop.shape

    # Draw contours.
    contours, hierarchy = cv2.findContours(
        image=mask_prob_crop, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE
    )

    # Show contours on the cropped image.
    image_copy = mask_crop.copy()
    cv2.drawContours(
        image=image_copy,
        contours=contours,
        contourIdx=-1,
        color=(0, 255, 0),
        thickness=1,
        lineType=cv2.LINE_AA
    )
    cv2.imshow('mask_contours_crop', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('mask_contours_crop.jpg', image_copy)
    cv2.destroyAllWindows()
    # ------------------------------------------------------------------------ #

    # ----------------------------- Fit line? -------------------------------- #
    # This is the first contour, I think biggest?
    cnt = contours[0]

    # https://docs.opencv.org/4.x/dd/d49/tutorial_py_contour_features.html
    # Wow this actually works nicely. :) This is the tangent line.
    # Well we might need something here in case of vx or vy as 0?
    # vx and vy returns a normalized vector, so just have to make sure that
    # they aren't all 0?
    image_copy = mask_crop.copy()
    rows, cols = mask_prob_crop.shape
    [vx,vy,x,y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
    print(f'After cv2.fitLine, vx, vy: {vx} {vy}, x, y: {x} {y}')
    if vx == 0 or vy == 0:
        print('Warning! vx or vy are 0, need to re-normalize.')
        # Not sure how principled this is but might be OK?
        if vx == 0:
            vx += 1e-4
        if vy == 0:
            vy += 1e-4
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image_copy, (cols-1,righty), (0,lefty), GREEN, 2)

    # I think we want to get the _perpendicular_ line segment.
    tmp = vx
    vx = vy
    vy = -tmp
    lefty = int((-x*vy/vx) + y)
    righty = int(((cols-x)*vy/vx)+y)
    cv2.line(image_copy, (cols-1,righty), (0,lefty), BLUE, 2)

    cv2.imshow('mask_contours_crop_fitline', image_copy)
    cv2.waitKey(0)
    cv2.imwrite('mask_contours_crop_fitline.jpg', image_copy)
    cv2.destroyAllWindows()
    # ------------------------------------------------------------------------ #

    # -------------------- Perp line, put on orig img ------------------------ #
    # Let's get the perpendicular line, and put all the info on the original image.
    # This is mask_copy, BTW, but need to be back to rgb now.
    mask_copy = DU.bgr_to_rgb(mask_copy)

    # Simple: just override the cropped region? YES! Nice.
    #assert image_copy.shape == (int(ss*2), int(ss*2), 3), f'{image_copy.shape} {ss}'
    if image_copy.shape != (int(ss*2), int(ss*2), 3):
        print(f'Warning: {image_copy.shape}, means a crop smaller than usual.')
    mask_copy[p0-ss:p0+ss, p1-ss:p1+ss] = image_copy

    # Let's pad the image. Don't pad the 3rd channel! Pad with 255 (i.e., white).
    # This is STRICTLY FOR READABILITY later when we insert the angles (in deg and
    # radian). This is NOT the same as padding for handling crops/grasps close to
    # the image boundary (which would need black values, i.e., 0, for padding).
    mask_copy = np.pad(mask_copy, ((40,40), (40,40), (0,0)), constant_values=255)
    print(f'After padding, mask copy: {mask_copy.shape}')

    # Also before we do this let's annotate the angle. Same color as perpendicular
    # line segment, for consistency (with all the potential BGR vs RGB confusions).
    # https://stackoverflow.com/questions/66839398/python-opencv-get-angle-direction-of-fitline
    # Note that we overrode vx and vy to be the perpendicular case.
    x_axis      = np.array([1, 0])    # unit vector in the same direction as the x axis
    your_line   = np.array([vx, vy])  # unit vector in the same direction as your line
    dot_product = np.dot(x_axis, your_line)
    angle_2_x   = np.arccos(dot_product)
    angle_deg   = np.rad2deg(angle_2_x)
    cv2.putText(
        img=mask_copy,
        text="{:.2f} (rad)".format(angle_2_x[0]),
        org=(20, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=BLUE,
        thickness=1
    )
    cv2.putText(
        img=mask_copy,
        text="{:.2f} (deg)".format(angle_deg[0]),
        org=(120, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=BLUE,
        thickness=1
    )

    # Take into account how we start from a '90 deg' rotation. I think this is
    # simpler than I thought, because the default degree is just angle_deg=0.
    # Also we only get positive values from the rad2deg it seems.
    assert angle_deg[0] > 0, angle_deg
    if angle_deg[0] >= 95:
        angle_deg_revised = angle_deg[0] - 180.
        angle_deg_revised = max(angle_deg_revised, -30)  # not too much
    else:
        angle_deg_revised = angle_deg[0]  # no need for changes :)
    print(f'Our angle revised: {angle_deg_revised:0.2f}')

    cv2.putText(
        img=mask_copy,
        text="{:.2f} (revised)".format(angle_deg_revised),
        org=(240, 20),
        fontFace=cv2.FONT_HERSHEY_SIMPLEX,
        fontScale=0.5,
        color=BLUE,
        thickness=1
    )

    # The usual.
    mask_copy = DU.rgb_to_bgr(mask_copy)
    cv2.imshow('mask_contours_all_info_orig', mask_copy)
    cv2.waitKey(0)
    cv2.imwrite('mask_contours_all_info_orig.jpg', mask_copy)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    np.random.seed(4)

    # Just get these from `data_collector.py`.
    #img_path = 'scripts/mask_example_01.png'
    #img_path = 'scripts/mask_example_02.png'
    img_path = 'scripts/mask_example_03.png'

    assert os.path.exists(img_path), img_path
    mask = cv2.imread(img_path)
    print(f'Loaded mask: {mask.shape}')

    get_pix_and_rotation(mask)
