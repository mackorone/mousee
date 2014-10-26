import cv2
import numpy as np

# Get the image and convert to HSV
img = cv2.imread('../data/m7/IMG_0290.JPG',cv2.CV_LOAD_IMAGE_COLOR)

# Create the red mask
lower_red = np.array([0,10,155])
upper_red = np.array([10,255,255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
red_mask = cv2.inRange(hsv, lower_red, upper_red)

# Dilate the red mask
d_k_s = 5 # dilation_kernel_size
dilation_kernel = np.ones((d_k_s, d_k_s), np.float32)/d_k_s**2
dilated = cv2.dilate(red_mask, dilation_kernel, iterations = 3)

# Filter the image using the dilated red mask
red_filtered = cv2.bitwise_and(img, img, mask = dilated)

# Convert to binary and remove any small contours
gray = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2GRAY)
ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
contours, hier = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) < 500:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(binary,(x,y),(x+w,y+h),0,-1)

# Skeletonize
size = np.size(binary)
skel = np.zeros(binary.shape,np.uint8)
ret, im = cv2.threshold(binary, 127, 255, 0)
element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
while True:
    eroded = cv2.erode(im,element)
    temp = cv2.dilate(eroded,element)
    temp = cv2.subtract(im,temp)
    skel = cv2.bitwise_or(skel,temp)
    im = eroded.copy()
    zeros = size - cv2.countNonZero(im)
    if zeros == size:
        break

# Now remove the noise from the skeleton
contours, hier = cv2.findContours(skel.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
for c in contours:
    if cv2.contourArea(c) < 10:
        (x,y,w,h) = cv2.boundingRect(c)
        cv2.rectangle(skel,(x,y),(x+w,y+h),0,-1)

cv2.imshow("f", skel)
cv2.waitKey()

# Opening filter...

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

