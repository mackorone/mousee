import cv2
import math
import numpy as np
import matplotlib.pylab as pl

# Import our function for skeletonizing red regions
from getLines import getLines

# Draws lines on an image
def draw_lines_on_img(lines, img, color):
    for (rho, theta) in lines:
        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        cv2.line(img,(x1,y1),(x2,y2), color,2)

# Specify the image path
image_path = '../data/m7/IMG_0289.JPG'

# Input image (should be the skeletonized image)
img = getLines(image_path)
img_num_rows = np.shape(img)[1]
img_num_cols = np.shape(img)[0]

# Get the color image
color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)

# This code was taken from [http://opencv-python-tutroals.readthedocs.org/
# en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html]
lines = [(rho, theta) for rho, theta in cv2.HoughLines(img, 1, np.pi/180, 60)[0]]

# Fraction deviation that is tolerated
rho_tolerance = 50 # pixel distance
theta_tolerance = np.pi / 4.0 # radians

# Create the groups of lines
groups = []
def putLineInGroup(line):
    for g in groups:
        good_rho = abs(line[0] - np.mean([x[0] for x in g])) <= rho_tolerance
        good_theta = abs(line[1] - np.mean([x[1] for x in g])) <= theta_tolerance
        if good_rho and good_theta:
            g.append(line)
            return
    groups.append([line])
for line in lines:
    putLineInGroup(line)

# Show each of the groupings
for g in groups:
    copy = color_img.copy()
    draw_lines_on_img(g, copy, (0, 0, 255))
    cv2.imshow('houghlines.jpg', copy)
    cv2.waitKey()

'''
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]
'''

ave_lines = []
for g in groups:
    if len(g) < 2: # Check to ensure we don't get single lines
        continue
    # TODO: get rid of outliers
    ave_rho = np.mean([x[0] for x in g])
    ave_theta = np.mean([x[1] for x in g])
    ave_lines.append((ave_rho, ave_theta))
    '''
    ave_slope = np.mean(reject_outliers(np.array([x[0] for x in g])))
    ave_intercept = np.mean(reject_outliers(np.array([x[1] for x in g])))
    #if not np.isnan(ave_slope) and not np.isnan(ave_intercept):
    ave_lines.append((ave_slope, ave_intercept))
    '''

draw_lines_on_img(ave_lines, color_img, (0, 255, 0))
cv2.imshow('houghlines.jpg',color_img)
cv2.waitKey()
