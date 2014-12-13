import cv2
import numpy as np
from hough import hough_lines
from hough import draw_rho_theta_lines
from hough import rho_theta_to_x1y1_x2y2

# The code for this function was taken from:
# http://stackoverflow.com/questions/20677795/find-the-point-of-intersecting-lines
def line_intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1]) #Typo was here
    def det(a, b):
        return a[0] * b[1] - a[1] * b[0]
    div = det(xdiff, ydiff)
    if div == 0:
       raise Exception('lines do not intersect')
    d = (det(*line1), det(*line2))
    x = det(d, xdiff) / div
    y = det(d, ydiff) / div
    return x, y

# Specify the image path and get the color image
image_path = '../data/m7/IMG_0289.JPG'
color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)

# Get the hough lines for the image
ave_lines = hough_lines(image_path)
img_with_lines = draw_rho_theta_lines(ave_lines, color_img, (0, 255, 0))
x_y_lines = rho_theta_to_x1y1_x2y2(ave_lines, np.shape(color_img))
#just_ave_lines = draw_rho_theta_lines(ave_lines, np.zeros(np.shape(color_img)), (255, 255, 255))

for i in range(len(x_y_lines)):
    for j in range(i+1, len(x_y_lines)):
        A,B = x_y_lines[i]
        C,D = x_y_lines[j]
        X = line_intersection((A, B), (C, D))
        cv2.circle(img_with_lines, X, 10, (255, 0, 0), -1)

cv2.imshow('foo', img_with_lines)
cv2.waitKey()
