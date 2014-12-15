# Note: This file is a modified version of https://github.com/cbuntain/stitcher

# TODO: Reduce the number of imports
import os
import sys
import cv2
import math
import numpy as np
from numpy import linalg

def filter_matches(matches, ratio = 0.75):
    filtered_matches = []
    for m in matches:
        if len(m) == 2 and m[0].distance < m[1].distance * ratio:
            filtered_matches.append(m[0])
    return filtered_matches
    
def imageDistance(matches):
    sumDistance = 0.0
    for match in matches:
        sumDistance += match.distance
    return sumDistance

def findDimensions(image, homography):

    base_p1 = np.ones(3, np.float32)
    base_p2 = np.ones(3, np.float32)
    base_p3 = np.ones(3, np.float32)
    base_p4 = np.ones(3, np.float32)

    (y, x) = image.shape[:2]

    base_p1[:2] = [0,0]
    base_p2[:2] = [x,0]
    base_p3[:2] = [0,y]
    base_p4[:2] = [x,y]

    max_x = None
    max_y = None
    min_x = None
    min_y = None

    for pt in [base_p1, base_p2, base_p3, base_p4]:

        hp = np.matrix(homography, np.float32) * np.matrix(pt, np.float32).T
        hp_arr = np.array(hp, np.float32)
        normal_pt = np.array([hp_arr[0]/hp_arr[2], hp_arr[1]/hp_arr[2]], np.float32)

        if ( max_x == None or normal_pt[0,0] > max_x ):
            max_x = normal_pt[0,0]
        if ( max_y == None or normal_pt[1,0] > max_y ):
            max_y = normal_pt[1,0]
        if ( min_x == None or normal_pt[0,0] < min_x ):
            min_x = normal_pt[0,0]
        if ( min_y == None or normal_pt[1,0] < min_y ):
            min_y = normal_pt[1,0]

    min_x = min(0, min_x)
    min_y = min(0, min_y)

    return (min_x, min_y, max_x, max_y)

def stitch(base_img, add_img):

    base_img_blur = cv2.GaussianBlur(cv2.cvtColor(base_img.copy(), cv2.COLOR_BGR2GRAY), (5,5), 0)

    # Use the SIFT feature detector
    detector = cv2.SIFT()

    # Find key points in base image for motion estimation
    base_features, base_descs = detector.detectAndCompute(base_img_blur, None)

    # Parameters for nearest-neighbor matching
    FLANN_INDEX_KDTREE = 1
    flann_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
    matcher = cv2.FlannBasedMatcher(flann_params, {})

    # Info
    next_img = cv2.GaussianBlur(cv2.cvtColor(add_img.copy(), cv2.COLOR_BGR2GRAY), (5,5), 0)
    next_features, next_descs = detector.detectAndCompute(next_img, None)
    matches = matcher.knnMatch(next_descs, trainDescriptors=base_descs, k=2)
    matches_subset = filter_matches(matches)
    distance = imageDistance(matches_subset)
    averagePointDistance = distance/float(len(matches_subset))

    # TODO: list comprehension
    kp1 = []
    kp2 = []
    for match in matches_subset:
        kp1.append(base_features[match.trainIdx])
        kp2.append(next_features[match.queryIdx])

    p1 = np.array([k.pt for k in kp1])
    p2 = np.array([k.pt for k in kp2])

    H, status = cv2.findHomography(p1, p2, cv2.RANSAC, 5.0)

    inlierRatio = float(np.sum(status)) / float(len(status))
    
    closestImage = {}
    closestImage['h'] = H
    closestImage['inliers'] = inlierRatio
    closestImage['rgb'] = add_img
    closestImage['img'] = next_img
    closestImage['match'] = matches_subset


    # Does it work from here??? -----------------------------------------


    H = closestImage['h']
    H = H / H[2,2]
    H_inv = linalg.inv(H)

    (min_x, min_y, max_x, max_y) = findDimensions(closestImage['img'], H_inv)

    # Adjust max_x and max_y by base img size
    max_x = max(max_x, base_img_blur.shape[1])
    max_y = max(max_y, base_img_blur.shape[0])

    move_h = np.matrix(np.identity(3), np.float32)

    if ( min_x < 0 ):
        move_h[0,2] += -min_x
        max_x += -min_x

    if ( min_y < 0 ):
        move_h[1,2] += -min_y
        max_y += -min_y

    mod_inv_h = move_h * H_inv

    img_w = int(math.ceil(max_x))
    img_h = int(math.ceil(max_y))

    # Warp the new image given the homography from the old image
    base_img_warp = cv2.warpPerspective(base_img, move_h, (img_w, img_h))
    next_img_warp = cv2.warpPerspective(closestImage['rgb'], mod_inv_h, (img_w, img_h))

    # Put the base image on an enlarged palette
    enlarged_base_img = np.zeros((img_h, img_w, 3), np.uint8)

    # Create a mask from the warped image for constructing masked composite
    (ret,data_map) = cv2.threshold(cv2.cvtColor(next_img_warp, cv2.COLOR_BGR2GRAY), 
        0, 255, cv2.THRESH_BINARY)

    enlarged_base_img = cv2.add(enlarged_base_img, base_img_warp, 
        mask=np.bitwise_not(data_map), 
        dtype=cv2.CV_8U)

    # Now add the warped image
    final_img = cv2.add(enlarged_base_img, next_img_warp, 
        dtype=cv2.CV_8U)

    # Crop off the black edges
    final_gray = cv2.cvtColor(final_img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(final_gray, 1, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    print "Found %d contours..." % (len(contours))

    max_area = 0
    best_rect = (0,0,0,0)

    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        # print "Bounding Rectangle: ", (x,y,w,h)

        deltaHeight = h-y
        deltaWidth = w-x

        area = deltaHeight * deltaWidth

        if ( area > max_area and deltaHeight > 0 and deltaWidth > 0):
            max_area = area
            best_rect = (x,y,w,h)

    if ( max_area > 0 ):
        print "Maximum Contour: ", max_area
        print "Best Rectangle: ", best_rect

        final_img_crop = final_img[best_rect[1]:best_rect[1]+best_rect[3],
                best_rect[0]:best_rect[0]+best_rect[2]]

        # utils.showImage(final_img_crop, scale=(0.2, 0.2), timeout=0)
        # cv2.destroyAllWindows()

        final_img = final_img_crop

    # Write out the current round
    #final_filename = "%s/%d.JPG" % (output_dir, round)
    #cv2.imwrite(final_filename, final_img)
    cv2.imwrite('foop.jpg', final_img)


if __name__ == '__main__':
    base_img = cv2.imread('../data/m7/IMG_0290.JPG')
    add_img = cv2.imread('../data/m7/IMG_0291.JPG')
    stitch(base_img, add_img)
