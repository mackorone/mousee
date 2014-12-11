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


def quantize(val, r):
    # Quantize a list to the nearest increment r
    return np.around(np.float(val)/r)*r

inc = 0.1   # The interval used to quantize the stretch factor values

# This will look good in our report
xs = np.arange(1, img_num_rows)
ys = [getPhysicalPosition(x) for x in xs]
sf = [getStretchFactor(x) for x in xs]
pl.plot(xs, ys, 'b')
pl.plot(xs, sf, 'r')
pl.show()

#sfq = [quantize(x, inc) for x in sf]     # The quantized stretch factor values
# Plot the stretch factors alongside the quantized stretch factors
#pl.plot(xs, sf, 'r-', xs, sfq, 'b-', linewidth=2)
#pl.show()
# exit(0)

'''
# Assuming that we round each stretch factor to the nearest 0.1: obtain a list of integer values representing the exact
# number of rows in the interpolated image that each row of the original image occupies.
#   Ex: If sfq = 1.1, a row from the original image occupies 11 rows in the interpolated image
sfq_int = [int(10*x) for x in sfq]

# Get the total number of rows that the interpolated image needs to have
interp_rows = np.sum(sfq_int)
print "Row count of interpolated image: ", interp_rows

# Interpolation
# large = cv2.resize(img, (np.shape(img)[1]*sf, np.shape(img)[0]*sf))
# print np.shape(img)
# print np.shape(large)
'''

# Get the output for every image in the data directory
import os
for string in os.listdir('../data/m7')[1:]:

    img = cv2.imread('../data/m7/' + string, cv2.CV_LOAD_IMAGE_COLOR)

    # Interpolate to a large image, perform the discretization, and subsample
    output = []
    for i in range(img_num_rows):
        for j in range(int(3*getStretchFactor(img_num_rows-i))):
            output.append(img[i,:,:])
    output = np.array(output)

    # Write the output image
    cv2.imwrite('out/' + string + '.jpg', cv2.resize(output, (img_num_cols, img_num_rows)))
