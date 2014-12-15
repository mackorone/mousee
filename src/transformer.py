import cv2
import numpy as np
import matplotlib.pyplot as plt
from get_square import *
import os

if __name__ == "__main__":

    draw = False         # Draw results at the end?
    cv_or_mpl = 'mpl'    # Use openCV or matplotlib to render the results?

    for path in os.listdir('imgs/'):

        # Load the test image
        #path = '../data/m7/IMG_02' + str(image_number) + '.JPG'
        # path = '../data/m7/IMG_0291.JPG'
        img = cv2.imread('imgs/' + path, cv2.CV_LOAD_IMAGE_COLOR)

        # Do a clockwise sort on the points
        def cw_sort(points):
            center = (np.mean([z[0] for z in points]), np.mean([z[1] for z in points]))
            cv2.circle(img, (int(center[0]), int(center[1])), 10, (255, 255, 255), -1)
            xy_and_theta = [(p,np.arctan2(p[1]-center[1], p[0]-center[0])) for p in points]
            return [z[0] for z in sorted(xy_and_theta, key = lambda x: x[1])]

        # Clockwise sort the corners
        corners = cw_sort(get_square(img))

        ortho_side = 2500    # Side length of square rectified image 
        ortho_dim = tuple([ortho_side]*2)  # Get explicit dimensions as tuple
        sq_side = 200  # Side length of the rectified version of smallest square

        # Construct the pixel locations of the square in the middle of the
        # transformed image
        central_square = [
                          (ortho_side/2 - sq_side/2, ortho_side/2 - sq_side/2),
                          (ortho_side/2 + sq_side/2, ortho_side/2 - sq_side/2),
                          (ortho_side/2 + sq_side/2, ortho_side/2 + sq_side/2),
                          (ortho_side/2 - sq_side/2, ortho_side/2 + sq_side/2)
                         ]

        # Numpy-ize the lists of corner locations to use them in the homography function
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

            # Define some colors to use when we draw vertices
            blue = (255, 0, 0)
            green = (0, 255, 0)

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
        
        cv2.imwrite('FOO-' + path, ortho)
