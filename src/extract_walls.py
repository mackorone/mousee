import cv2
import numpy as np
from get_square import get_square
from hough import hough_lines
from hough import draw_rho_theta_lines
from hough import draw_x1y1_x2y2_lines
from hough import rho_theta_to_x1y1_x2y2
from get_square import get_groups_of_lines
from get_square import line_intersection
from get_square import intersection_within_image

'''
Code for getting the square

image_path = '../data/m7/IMG_0292.JPG'
square = get_square(image_path)
p1,p2,p3,p4 = square
img = cv2.imread(image_path)

cv2.circle(img, p1, 10, (255, 0, 0), -1)
cv2.circle(img, p2, 10, (255, 0, 0), -1)
cv2.circle(img, p3, 10, (255, 0, 0), -1)
cv2.circle(img, p4, 10, (255, 0, 0), -1)
cv2.imshow('foo', img)
cv2.waitKey()
'''

image_path = 'topdown_cropped.jpg'
color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
draw_img = color_img.copy()
ave_lines = hough_lines(image_path)
group1, group2 = get_groups_of_lines(rho_theta_to_x1y1_x2y2(ave_lines, np.shape(color_img)), np.shape(color_img))

# Get all of the intersection points
points = []
for i in range(len(group1)):
    for j in range(len(group2)):
        if (intersection_within_image(group1[i], group2[j], np.shape(color_img))):
            points.append(line_intersection(group1[i], group2[j]))

'''
color_img = draw_x1y1_x2y2_lines(group1, color_img, (0, 255, 0))
color_img = draw_x1y1_x2y2_lines(group2, color_img, (255, 0, 0))
for p in points:
    cv2.circle(color_img, p, 10, (0, 0, 255), -1)
cv2.imshow('foo', color_img)
cv2.waitKey()
'''

# First, sort and split the points by y-value
points = sorted(points, key = lambda z: z[1])
rows = [points[5*i:5*i + 5] for i in range(7)]

# Then sort by x-value
rows = [sorted(r, key = lambda x: x[0]) for r in rows]

# Create the red mask for determining whether or not walls exist
lower_red = np.array([0,10,155])
upper_red = np.array([10,255,255])

def horizontal_wall_exists(img, left, right):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    y = int((left[1] + right[1])/2)
    window = hsv[y-25:y+25, left[0]+15:right[0]-15, :]
    red_mask = cv2.inRange(window, lower_red, upper_red)
    return sum(sum(red_mask)) > 500

def vertical_wall_exists(img, top, bottom):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    x = int((top[0] + bottom[0])/2)
    window = hsv[top[1]+15:bottom[1]-15, x-25:x+25, :]
    red_mask = cv2.inRange(window, lower_red, upper_red)
    return sum(sum(red_mask)) > 500

# The tiles are numbered as follows:
# |  0  1  2  3 |
# |  4  5  6  7 |
# |  8  9 10 11 |
# | 12 13 14 15 |
# | 16 17 18 19 |
# | 20 21 22 23 |
walls = [[0,0,0,0] for i in range(24)]

# Check all pairs of points along the horizonal
for i in range(len(rows)):
    for j in range(len(rows[i])-1):
        if horizontal_wall_exists(color_img, rows[i][j], rows[i][j+1]):
            center = ((rows[i][j][0] + rows[i][j+1][0])/2, (rows[i][j][1] + rows[i][j+1][1])/2)
            # TODO: Add to walls
            # walls[
            cv2.circle(draw_img, center, 10, (0, 0, 255), -1)

# Check all pairs of points along the vertical
for i in range(len(rows[0])):
    for j in range(len(rows)-1):
        if vertical_wall_exists(color_img, rows[j][i], rows[j+1][i]):
            center = ((rows[j][i][0] + rows[j+1][i][0])/2, (rows[j][i][1] + rows[j+1][i][1])/2)
            # TODO: Add to walls
            cv2.circle(draw_img, center, 10, (0, 0, 255), -1)

cv2.imshow('foo', draw_img)
cv2.waitKey()
