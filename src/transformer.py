import cv2
import numpy as np
import matplotlib.pyplot as plt
from getSquare import *
from hough import *  # Might not need to import this directly here

if __name__ == "__main__":

	draw = False 		# Draw results at the end?
	cv_or_mpl = 'mpl'	# Use openCV or matplotlib to render the results?

	for image_number in range(89, 96):
		# Load the test image
		path = '../data/m7/IMG_02' + str(image_number) + '.jpg'
		# path = '../data/m7/IMG_0291.JPG'
		img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)

		# Get the smallest square in the image
		polys, group1, group2 = find_squares_in_image(img)

		# The corners are ordered: top right, bottom right, top left, bottom left
		corners = find_smallest_square(polys)
		print corners

		corners = sorted(corners)
		print corners

		'''
		Note: the corner order is top left, bottom left, top right, bottom right
		'''
		# Define some colors to use when we draw vertices
		blue = (255, 0, 0)
		green = (0, 255, 0)

		ortho_side = 2500	# Side length of square rectified image 
		ortho_dim = tuple([ortho_side]*2)  # Get explicit dimensions as tuple
		sq_side = 200  # Side length of the rectified version of smallest square

		# Construct the pixel locations of the square in the middle of the
		# transformed image
		central_square = [(ortho_side/2 - sq_side/2, ortho_side/2 - sq_side/2), 
						  (ortho_side/2 - sq_side/2, ortho_side/2 + sq_side/2),
						  (ortho_side/2 + sq_side/2, ortho_side/2 - sq_side/2), 
						  (ortho_side/2 + sq_side/2, ortho_side/2 + sq_side/2)]

		# Numpy-ize the lists of corner locations to use them in the homography 
		# function
		src = np.array(corners, np.float32)
		dst = np.array(central_square, np.float32)

		# Get the transformation and apply it
		transform = cv2.getPerspectiveTransform(src, dst)
		ortho = cv2.warpPerspective(img, transform, ortho_dim)

		# Now we have a huge image filled mostly with nonzeros, so crop up to the
		# square hull of the nonzero region
		nonzeros = np.nonzero(ortho)
		upper = np.min(nonzeros[0][:])
		lower = np.max(nonzeros[0][:])
		left = np.min(nonzeros[1][:])
		right = np.max(nonzeros[1][:])

		ortho = ortho[upper:lower, left:right]
		print upper, lower, left, right

		# Draw everything
		if draw:
			# Draw the original and orthographic projections
			draw_corners(img, corners, blue)
			draw_corners(ortho, central_square, green)
			if cv_or_mpl == 'cv':
				cv2.imshow('img', img)
				cv2.waitKey()
				cv2.imshow('ortho', ortho)
				cv2.waitKey()
			else:
				plt.figure('img')
				plt.imshow(img)
				plt.figure('ortho')
				plt.imshow(ortho)
				plt.show()
		
		cv2.imwrite('orthographic/' + str(image_number) + '.jpg', ortho)