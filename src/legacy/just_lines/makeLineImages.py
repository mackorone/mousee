import cv2
import numpy as np
import os
from hough import hough_lines
from hough import draw_rho_theta_lines
from hough import draw_x1y1_x2y2_lines
from hough import rho_theta_to_x1y1_x2y2

# Specify the image path and get the color image
for image in os.listdir('../data/m7/'):
    image_path = '../data/m7/' + image
    color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
    ave_lines = hough_lines(image_path)
    just_ave_lines = draw_rho_theta_lines(ave_lines, np.zeros(np.shape(color_img)), (255, 255, 255))
    cv2.imwrite('./out/' + image, just_ave_lines)
