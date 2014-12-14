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

# Now, we just need to check the pixels within pairs of points

