import numpy as np
import cv2
import matplotlib.pylab as plt

# Load one of the test images
img = cv2.imread('../data/m7/IMG_0290.jpg', cv2.CV_LOAD_IMAGE_COLOR)

# Coordinates of the square from the perspective image
topleft = [416, 207]
topright = [710, 172]
bottomright = [884, 291]
bottomleft = [515, 348]
src = np.array([topleft, topright, bottomright, bottomleft], np.float32)

# Coordinates of the square from an orthographic point of view
stopleft = [200, 600]
stopright = [400, 600]
sbottomright = [400, 800]
sbottomleft = [200, 800]
dst = np.array([stopleft, stopright, sbottomright, sbottomleft], np.float32)

transform = cv2.getPerspectiveTransform(src, dst)
print transform
print img.shape

# ortho = np.zeros(img.shape)

ortho = cv2.warpPerspective(img, transform, (720, 960))

cv2.imshow('img', img)
cv2.imshow('ortho', ortho)
cv2.waitKey()