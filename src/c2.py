import cv2
import numpy as np

# Get the image and convert to HSV
img = cv2.imread('../data/m7/IMG_0290.JPG',cv2.CV_LOAD_IMAGE_COLOR)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

# Specify the red range for the color mask
lower_red = np.array([0,0,0])
upper_red = np.array([10,255,255])
red_mask = cv2.inRange(hsv, lower_red, upper_red)

res = cv2.bitwise_and(hsv, hsv, mask = red_mask)

#dialation_size = 
dialation_kernal = np.ones((5,5),np.float32)/25
mack = cv2.dilate(res,dialation_kernal,iterations = 1)

res2 = cv2.bitwise_and(mack, mack, mask = red_mask)
mack2 = cv2.dilate(res2,kern,iterations = 1)

#mack = cv2.morphologyEx(res, cv2.MORPH_CLOSE, kern)
rgb = cv2.cvtColor(mack2, cv2.COLOR_HSV2BGR)

size = 1
kernel = np.ones((size,size),np.float32)/size**2
blurred = cv2.filter2D(rgb,-1,kernel)

cv2.imshow('frame',blurred)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)
cv2.waitKey()
