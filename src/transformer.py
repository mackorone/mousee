import cv2
import numpy as np
import matplotlib.pyplot as plt
from getSquare import *
from hough import *  # Might not need to import this directly here

if __name__ == "__main__":

	cv_or_mpl = 'cv'

	# Load the test image
	path = '../data/m7/IMG_0289.JPG'
	img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

	# Get the smallest square in the image
	polys, group1, group2 = find_squares_in_image(img)

	# The corners are ordered: top right, bottom right, top left, bottom left
	corners = find_smallest_square(polys)

	# Construct a square by assuming that the top right and bottom left corners
	# will be the starting points
	top_right = corners[0]
	bottom_left = corners[3]
	side_length = max([np.abs(top_right[i] - bottom_left[i]) for i in range(2)])

	# Offset the starting corner by some amount in the x direction to avoid
	# clipping
	offset = 50
	trr = tuple([top_right[0] + offset, top_right[1]])
	rectified_corners = [trr, (trr[0], trr[1] + side_length), 
						(trr[0] - side_length, trr[1]),
						(trr[0] - side_length, trr[1] + side_length)]


	# Define some colors to use in plotting
	blue = (255, 0, 0)
	green = (0, 255, 0)

	# Draw
	draw_corners(img, corners, blue)
	draw_corners(img, rectified_corners, green)
	if cv_or_mpl == 'cv':
		cv2.imshow('img', img)
		cv2.waitKey()
	else:
		plt.figure('img')
		plt.imshow(img)

	src = np.array(corners, np.float32)
	dst = np.array(rectified_corners, np.float32)
	transform = cv2.getPerspectiveTransform(src, dst)
	ortho = cv2.warpPerspective(img, transform, (960, 720))
	cv2.imshow('ortho', ortho)
	cv2.imwrite('ortho.jpg', ortho)
	cv2.waitKey()