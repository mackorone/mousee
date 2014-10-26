import cv2
import numpy as np

img = cv2.imread('../data/m7/IMG_0290.JPG',cv2.CV_LOAD_IMAGE_COLOR)

lower_red = np.array([0,0,0])
upper_red = np.array([10,255,255])

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower_red, upper_red)
res = cv2.bitwise_and(hsv, hsv, mask = mask)

cop = res.copy()
for row in range(len(res)):
    for col in range(len(res[0])):
        if all([res[row][col][i] != 0 for i in [1,2]]):
            r = 5
            for i in range(-r,r):
                for j in range(-r,r):
                    if (0 <= row + i and row + i < len(res)
                    and 0 <= col + j and col + j < len(res[0])):
                        cop[row+i][col+j][0] = 0
                        cop[row+i][col+j][1] = 255
                        cop[row+i][col+j][2] = 255

rgb = cv2.cvtColor(cop, cv2.COLOR_HSV2BGR)

size = 5
kernel = np.ones((size,size),np.float32)/size**2
blurred = cv2.filter2D(rgb,-1,kernel)

cv2.imshow('frame',blurred)
#cv2.imshow('mask',mask)
#cv2.imshow('res',res)
cv2.waitKey()
