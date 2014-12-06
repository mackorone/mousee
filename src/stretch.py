import cv2
import math
import numpy as np
import pylab as pl

'''
                   Camera
                 .-. ^
              . - .  |
           .  -  .   |
        .   -   .    | camera_height
     .    -    .     |
  .     -     .      V
####################
             <-----> unseen_length
<------------------> view_distance

'''

camera_height = 5.0  # Height of the camera
view_distance = 15.0 # The length of the line connecting the point below the camera and the leftmost point captured
unseen_length = 3.0  # The length of the line connecting the point below the camera and the rightmost point captured
big_theta = math.atan(view_distance/camera_height) # Angle between the vertical and line connecting the camera to the leftmost point
lil_theta = math.atan(unseen_length/camera_height) # Angle between the vertical and line connecting the camera to the rightmost point
                  
# Input image
img = cv2.imread('../data/m7/IMG_0290.JPG', cv2.CV_LOAD_IMAGE_COLOR)
img_num_rows = np.shape(img)[0]
img_num_cols = np.shape(img)[1]

# Get's the physical position for a given row, based on the parameters listed above
def getPhysicalPosition(row):
    row_percentage = row/float(img_num_rows)
    theta = row_percentage*(big_theta-lil_theta)+lil_theta
    position = math.tan(theta)*camera_height
    return position

def getStretchFactor(row):
    # Normalized derivative of getPhysicalPosition
    one = camera_height*(big_theta-lil_theta)
    two = row*(big_theta-lil_theta)/img_num_rows + lil_theta
    three = one*(1/math.cos(two))**2/img_num_rows
    if (row == 0):
        return three
    else:
        return three / getStretchFactor(0)

xs = np.arange(1, 720)
ys = [getPhysicalPosition(x) for x in xs]
sf = [getStretchFactor(x) for x in xs]
pl.plot(xs, ys, 'b')
pl.plot(xs, sf, 'r')
pl.show()
exit(0)

# TODO: should each row have its own scale value???
stretch_factor = 5

# Input image
img = cv2.imread('../data/m7/IMG_0290.JPG', cv2.CV_LOAD_IMAGE_COLOR)

print scale_values
exit(0)

# Interpolation
large = cv2.resize(img, (np.shape(img)[1]*stretch_factor, np.shape(img)[0]*stretch_factor))

print np.shape(img)
print np.shape(large)
#cv2.imshow("Image", large)
#cv2.waitKey()
#cv2.imwrite('img.jpg', large)
