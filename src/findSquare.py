import cv2
import math
import numpy as np
import matplotlib.pylab as pl

# Import our function for skeletonizing red regions
from getLines import getLines

# Specify the image path
image_path = '../data/m7/IMG_0289.JPG'

# Input image (should be the skeletonized image)
img = getLines(image_path)
img_num_rows = np.shape(img)[1]
img_num_cols = np.shape(img)[0]

# Get the color image
color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)

# Create a list to hold (slope, intercept) tuples
lines = []

# This code was taken from [http://opencv-python-tutroals.readthedocs.org/
# en/latest/py_tutorials/py_imgproc/py_houghlines/py_houghlines.html]
for rho, theta in cv2.HoughLines(img, 1, np.pi/180, 50)[0]:

    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a*rho
    y0 = b*rho

    x1 = int(x0 + 1000*(-b))
    y1 = int(y0 + 1000*(a))
    x2 = int(x0 - 1000*(-b))
    y2 = int(y0 - 1000*(a))

    # Make sure we're not dividing by zero
    if (x2 != x1):
        slope = float(y2 - y1)/float(x2-x1)
        intercept = y1 - slope*x1
        lines.append((slope, intercept))

    cv2.line(color_img,(x1,y1),(x2,y2),(0,0,255),2)

# The idea is that lines that have a similar slope and intersect within the window
# should be grouped together (we assume we won't have truly parallel lines)
slope_tolerance = 0.2 # Fraction deviation that is tolerated

def intertsect_in_window(line1, line2):
    if line1[0] == line2[0]:
        return abs(line1[1] - line2[1])/float(line1[1]) < slope_tolerance # TODO
    x = float(line2[1] - line1[1])/float(line2[0]-line1[0])
    y = line1[0]*x + line1[1]
    return x < img_num_rows and y < img_num_cols

groups = [[lines[0]]]
def putLineInGroup(myLine):
    for g in groups:
        for line in g:
            good_slope = abs((myLine[0] - line[0])/float(myLine[0])) <= slope_tolerance
            if good_slope and intertsect_in_window(myLine, line):
                g.append(myLine)
                return
    groups.append([myLine])

for myLine in lines[1:]:
    putLineInGroup(myLine)

# TODO
def reject_outliers(data, m=2):
    return data[abs(data - np.mean(data)) < m * np.std(data)]

ave_lines = []
for g in groups:
    if len(g) < 1: # Check to ensure we don't get single lines
        continue
    # TODO: get rid of outliers
    ave_slope = np.mean([x[0] for x in g])
    ave_intercept = np.mean([x[1] for x in g])
    ave_lines.append((ave_slope, ave_intercept))
    '''
    ave_slope = np.mean(reject_outliers(np.array([x[0] for x in g])))
    ave_intercept = np.mean(reject_outliers(np.array([x[1] for x in g])))
    #if not np.isnan(ave_slope) and not np.isnan(ave_intercept):
    ave_lines.append((ave_slope, ave_intercept))
    '''

for ln in ave_lines:
    x1 = 0
    y1 = int(ln[1])
    x2 = 1000
    y2 = int(ln[1] + ln[0]*1000)
    cv2.line(color_img,(x1,y1),(x2,y2),(0,255,0),2)

cv2.imshow('houghlines.jpg',color_img)
cv2.waitKey()
