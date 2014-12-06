# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 13:28:03 2014

@author: Scott Will, SUNY Buffalo, Department of Electrical Engineering
"""

import cv2
import numpy as np

cv2.destroyAllWindows()

# Read in the target image
filename = '../data/m7/IMG_0290.JPG'
img = cv2.imread(filename)

# Convert BGR to HSV
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# define range of blue color in HSV

# This one approximately outlines the red regions
#lower_red = np.array([0, 50, 0])
#upper_red = np.array([10, 200, 200])

# This one gets all of the red region
lower_red = np.array([0, 100, 30])
upper_red = np.array([10, 255, 255])

dim = 10
kernel = np.ones((dim, dim), np.uint8)

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)
mask = cv2.dilate(mask, kernel, iterations=1)

#############################
size = np.size(mask)
skel = np.zeros(mask.shape, np.uint8)

ret, mask = cv2.threshold(mask, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
done = False

while(not done):
    eroded = cv2.erode(mask, element)
    temp = cv2.dilate(eroded, element)
    temp = cv2.subtract(mask, temp)
    skel = cv2.bitwise_or(skel, temp)
    img = eroded.copy()

    zeros = size - cv2.countNonZero(mask)
    if zeros == size:
        done = True

cv2.imshow("skel", skel)
##############################

# Bitwise-AND mask and original image
#res = cv2.bitwise_and(red, red, mask=mask)
#
#cv2.imshow('img', img)
#cv2.imshow('hsv', hsv)
#cv2.imshow('mask', mask)
#cv2.imshow('res', res)
