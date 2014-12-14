import cv2
import math
import numpy as np
import matplotlib.pylab as pl

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
img = cv2.imread('../data/m7/IMG_0290.JPG', cv2.CV_LOAD_IMAGE_COLOR) # TODO: Modularize this
img_num_rows = np.shape(img)[0]
img_num_cols = np.shape(img)[1]

# Gets the physical position for a given row, based on the parameters listed above
def getPhysicalPosition(row):
    row_percentage = row/float(img_num_rows)
    theta = row_percentage*(big_theta-lil_theta)+lil_theta
    position = math.tan(theta)*camera_height
    return position

def getStretchFactor(row):
    # Normalized derivative of getPhysicalPosition
    one = camera_height*(big_theta - lil_theta)
    two = row*(big_theta - lil_theta)/img_num_rows + lil_theta
    three = one*(1/math.cos(two))**2/img_num_rows
    if (row == 0):
        return three
    else:
        return three / getStretchFactor(0)

# This will look good in our report
'''
xs = np.arange(1, img_num_rows)
ys = [getPhysicalPosition(x) for x in xs]
sf = [getStretchFactor(x) for x in xs]
pl.plot(xs, ys, 'b')
pl.plot(xs, sf, 'r')
pl.show()
'''

# Get the output for every image in the data directory
import os
for string in os.listdir('../data/m7')[1:]:

    img = cv2.imread('../data/m7/' + string, cv2.CV_LOAD_IMAGE_COLOR)
    print string

    # Interpolate to a large image, perform the discretization, and subsample
    output = []
    for i in range(img_num_rows):
        for j in range(int(3*getStretchFactor(img_num_rows-i))):
            output.append(img[i,:,:])
    output = np.array(output)

    # Write the output image
    cv2.imwrite('out/' + string, cv2.resize(output, (img_num_cols, img_num_rows)))
