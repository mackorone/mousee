# -*- coding: utf-8 -*-
"""
Created on Sun Oct 26 13:14:30 2014

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

# define range of red color in HSV
lower_red = np.array([0, 0, 0])
upper_red = np.array([5, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

# Convert to grayscale
gray = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

# Run Harris corner detection
# Arguments: (image, blocksize, sobel aperture size, free parameter)
dst = cv2.cornerHarris(gray, 2, 3, 0.04)

# Result is dilated for marking the corners, not important
dst = cv2.dilate(dst, None)

# Threshold for an optimal value, it may vary depending on the image.
img[dst > 0.01*dst.max()] = [0, 0, 255]

cv2.imshow('dst', img)
