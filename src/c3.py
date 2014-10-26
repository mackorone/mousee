import cv2
import numpy as np

# Get the image and convert to HSV
img = cv2.imread('../data/m7/IMG_0289.JPG',cv2.CV_LOAD_IMAGE_COLOR)

# First create the red mask
lower_red = np.array([0,10,155])
upper_red = np.array([10,255,255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# Now dilate the red mask
d_k_s = 5 # dilation_kernel_size
dilation_kernel = np.ones((d_k_s, d_k_s), np.float32)/d_k_s**2
dilated = cv2.dilate(red_mask, dilation_kernel, iterations = 3)

# Now filter the image using the dilated red mask
red_filtered = cv2.bitwise_and(img, img, mask = dilated)
# cv2.imshow("f", red_filtered)
# cv2.waitKey()

#cv2.imshow("f", blur)
#cv2.waitKey()

# Opening filter...

# Now filter out dark pixels
# lower_dark = np.array([0, 0, 150])
# upper_dark = np.array([255, 255, 255])
# hsv_dilated = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2HSV)
# dark_mask = cv2.inRange(hsv_dilated, lower_dark, upper_dark)
# dark_filtered = cv2.bitwise_and(red_filtered, red_filtered, mask = dark_mask)
# cv2.imshow('frame', dark_filtered)
# cv2.waitKey()

# b_k_s = 5 # blurring_kernel_size
# kernel = np.ones((b_k_s, b_k_s), np.float32)/b_k_s**2
# blurred = cv2.filter2D(dark_filtered,-1,kernel)
# cv2.imshow('frame',blurred)
# cv2.waitKey()

# Image after red filter
# red = img.copy()
# red[:,:,0] = 0
# red[:,:,1] = 0
# red[:,:,2] = 255
# only_red = cv2.bitwise_and(red, red, mask = red_mask)
# #cv2.imshow('frame', red)
# cv2.imshow('frame', only_red)
# cv2.waitKey()

