import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from get_square import *
from skeleton import *

# Sorts points in a polygon in circular order
def circ_sort(points):
    center = (np.mean([z[0] for z in points]), np.mean([z[1] for z in points]))
    xy_and_theta = [(p,np.arctan2(p[1]-center[1], p[0]-center[0])) for p in points]
    return [z[0] for z in sorted(xy_and_theta, key = lambda x: x[1])]

def persp_to_ortho(img):

    # Clockwise sort the corners
    corners = circ_sort(get_square(img))

    ortho_size = 2500 # Length/width of the ortho image
    square_size = 200 # Length/width of the side of a tile in the ortho image

    # Construct the pixel locations of the square in the middle of the transformed image
    central_square = [(ortho_size/2 - square_size/2, ortho_size/2 - square_size/2),
                      (ortho_size/2 + square_size/2, ortho_size/2 - square_size/2),
                      (ortho_size/2 + square_size/2, ortho_size/2 + square_size/2),
                      (ortho_size/2 - square_size/2, ortho_size/2 + square_size/2)]

    # Numpy-ize the lists of corner locations to use them in the homography function
    src = np.array(corners, np.float32)
    dst = np.array(central_square, np.float32)

    # Get the transformation and apply it
    transform = cv2.getPerspectiveTransform(src, dst)
    ortho = cv2.warpPerspective(img, transform, (ortho_size, ortho_size))

    # Crop to the red region
    border = 50
    nonzeros = np.nonzero(get_red_mask(ortho))
    upper = np.min(nonzeros[0][:]) - border
    lower = np.max(nonzeros[0][:]) + border
    left = np.min(nonzeros[1][:]) - border
    right = np.max(nonzeros[1][:]) + border

    ortho = ortho[upper:lower, left:right]
    return ortho

# Demo
if __name__ == "__main__":
    path = '../data/m7/IMG_0290.JPG'
    img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
    ortho = persp_to_ortho(img)
    cv2.imshow('Ortho', ortho)
    cv2.waitKey()
