import cv2
import numpy as np

# Get the image and convert to HSV
img = cv2.imread('../data/m7/IMG_0293.JPG',cv2.CV_LOAD_IMAGE_COLOR)

# Filter out the non-red pixels
lower_red = np.array([0,50,50])
upper_red = np.array([10,255,255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
red_mask = cv2.inRange(hsv, lower_red, upper_red)
red_filtered = cv2.bitwise_and(img, img, mask = red_mask)

# Now dilate
d_k_s = 5 # dilation_kernel_size
dilation_kernel = np.ones((d_k_s, d_k_s), np.float32)/d_k_s**2
dilated = cv2.dilate(red_filtered, dilation_kernel, iterations = 5)

# Now filter out dark pixels
lower_dark = np.array([0, 0, 150])
upper_dark = np.array([255, 255, 255])
hsv_dilated = cv2.cvtColor(dilated, cv2.COLOR_BGR2HSV)
dark_mask = cv2.inRange(hsv_dilated, lower_dark, upper_dark)
dark_filtered = cv2.bitwise_and(dilated, dilated, mask = dark_mask)

cv2.imshow('frame', dark_filtered)
cv2.waitKey()


#mack = cv2.morphologyEx(red, cv2.MORPH_CLOSE, kern)

b_k_s = 5 # blurring_kernel_size
kernel = np.ones((b_k_s, b_k_s), np.float32)/b_k_s**2
blurred = cv2.filter2D(dark_filtered,-1,kernel)
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

