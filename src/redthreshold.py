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
lower_red = np.array([0, 0, 0])
upper_red = np.array([5, 255, 255])

# Threshold the HSV image to get only red colors
mask = cv2.inRange(hsv, lower_red, upper_red)

# Bitwise-AND mask and original image
res = cv2.bitwise_and(img, img, mask=mask)

size = np.size(img)
skel = np.zeros(img.shape, np.uint8)

th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,\
            cv2.THRESH_BINARY, 11, 2)
#element = cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3))
#done = False
#
#while(not done):
#    eroded = cv2.erode(img,element)
#    temp = cv2.dilate(eroded,element)
#    temp = cv2.subtract(img,temp)
#    skel = cv2.bitwise_or(skel,temp)
#    img = eroded.copy()
#
#    zeros = size - cv2.countNonZero(img)
#    if zeros==size:
#        done = True
#
#cv2.imshow("skel",skel)


#median = cv2.medianBlur(res, 3)

cv2.imshow('img', img)
cv2.imshow('hsv', hsv)
cv2.imshow('mask', mask)
cv2.imshow('res', res)
#cv2.imshow('med', median)

#k = cv2.waitKey(27) & 0xFF
#cv2.destroyAllWindows()
