import cv2
import numpy as np
import matplotlib.pyplot as plt
from getSquare import *
from hough import *  # Might not need to import this directly here

if __name__ == "__main__":
	path = '../data/m7/IMG_0289.JPG'
	img = cv2.imread(path, cv2.CV_LOAD_IMAGE_COLOR)
	polys, group1, group2 = find_squares_in_image(img)

	plt.figure('img')
	plt.imshow(img)
	plt.show()