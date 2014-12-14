import cv2
import numpy as np
from skeleton import *

# Converts rho-theta lines to x1y1-x2y2 lines. We use the size of the image so
# as to ensure the endpoints of the seqments do not exceed the image edges 
def rho_theta_to_x1y1_x2y2(lines, img_shape):

    num_rows, num_cols = img_shape[0], img_shape[1]
    output = []

    for (rho, theta) in lines:

        a = np.cos(theta)
        b = np.sin(theta)
        x0 = a*rho
        y0 = b*rho

        # Theta is oriented as follows:
        #
        #  (0,0)--------->
        #    | \ theta
        #    |  \
        #    |   \
        #    V    \
        #
        # And the "quadrants" can be labeled as follows:
        #
        #      3 | 4
        #     ___|___
        #        |
        #      2 | 1
        #
        # Using this knowledge, we can determine the appropriate x and y limits, and
        # from these limits we can determine how much to extend the line for (x0, y0)

        # Quadrant 1
        if (0 < theta < np.pi/2):
            r1 = (num_cols - x0)/b
            r2 = (       0 - y0)/-a
            s1 = (       0 - x0)/-b
            s2 = (num_rows - y0)/a

        # Quadrant 2
        elif (np.pi/2 < theta < np.pi):
            r1 = (num_cols - x0)/b
            r2 = (num_rows - y0)/-a
            s1 = (       0 - x0)/-b
            s2 = (       0 - y0)/a

        # Quadrant 3
        elif (np.pi < theta < 3*np.pi/2):
            r1 = (       0 - x0)/b
            r2 = (num_rows - y0)/-a
            s1 = (num_cols - x0)/-b
            s2 = (       0 - y0)/a

        # Quadrant 4
        else:
            r1 = (       0 - x0)/b
            r2 = (       0 - y0)/-a
            s1 = (num_cols - x0)/-b
            s2 = (num_rows - y0)/a

        m1 = min(r1, r2)
        m2 = min(s1, s2)

        x1 = int(x0 + m1*b)
        y1 = int(y0 + m1*-a)
        x2 = int(x0 + m2*-b)
        y2 = int(y0 + m2*a)

        # Sanity check
        for a in [x1, x2]:
            if not (0 <= a and a <= num_cols):
                print 'ERROR - x value of segment out of range'
                exit(0)
        for b in [y1, y2]:
            if not (0 <= b and b <= num_rows):
                print 'ERROR - y value of segment out of range'
                exit(0)

        output.append(((x1,y1),(x2,y2)))

    return output

# Draws rho-theta lines on an image, where color is (b, g, r)
def draw_rho_theta_lines(lines, img, color):
    new_img = img.copy()
    for ((x1,y1),(x2,y2)) in rho_theta_to_x1y1_x2y2(lines, np.shape(new_img)):
        cv2.line(new_img, (x1,y1), (x2,y2), color, 2)
    return new_img

# Draws ((x1,y1),(x2,y2)) lines on an image, where color is (b, g, r)
def draw_x1y1_x2y2_lines(lines, img, color):
    new_img = img.copy()
    for ((x1,y1),(x2,y2)) in lines:
        cv2.line(new_img, (x1,y1), (x2,y2), color, 2)
    return new_img

# Gets the rho-theta hough lines for a maze image
def hough_lines(img):

    # Skeletonize the input image
    skel = skeletonize(img)

    # Use a Hough Tranform to extra all lines in rho-theta form
    lines = [(rho, theta) if rho >= 0 else (-rho, theta - np.pi)
            for rho, theta in cv2.HoughLines(skel, 1, np.pi/180, 65)[0]]

    # Note: Lines on opposite sides of the origin aren't being recognized as part
    # of the same group. The quoted out portions are my attempt to fix those lines.
    '''
    lines = [(rho, theta) if theta >= 0 else (-rho, theta - np.pi)
    '''

    # Grouping tolerance for each of the parameters
    rho_tolerance = 30 # pixel distance
    theta_tolerance = np.pi / 30.0 # radians

    # Create the groups of lines
    line_groups = []
    def putLineInGroup(line):
        for group in line_groups:
            '''
            good_rho = abs(abs(line[0]) - abs(np.mean([x[0] for x in group]))) <= rho_tolerance
            theta_dist = min((line[1] - np.mean([x[1] for x in group])) % np.pi,
                             (np.mean([x[1] for x in group]) - line[1]) % np.pi)
            print theta_dist
            good_theta = theta_dist <= theta_tolerance
            '''
            good_rho = abs(line[0] - np.mean([x[0] for x in group])) <= rho_tolerance
            good_theta = abs(line[1] - np.mean([x[1] for x in group])) <= theta_tolerance
            if good_rho and good_theta:
                group.append(line)
                return
        line_groups.append([line])
    for line in lines:
        putLineInGroup(line)

    # Show each of the groupings, sorted by theta (not necessary)
    show_groups = False
    if show_groups:
        for group in sorted(line_groups, key = lambda x: x[0][1]):
            print group
            copy = color_img.copy()
            copy = draw_rho_theta_lines(group, copy, (0, 0, 255))
            cv2.imshow('houghlines.jpg', copy)
            cv2.waitKey()

    # TODO: Can we re-center the lines? Sometimes they're a little off

    # Get the average for each group
    ave_lines = [(np.mean([x[0] for x in group]), np.mean([x[1] for x in group]))
                for group in line_groups if len(group) > 1]

    return ave_lines

# Demo
if __name__ == '__main__':
    image_path = '../data/m7/IMG_0290.JPG'
    color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
    lines = hough_lines(color_img)
    color_img = draw_rho_theta_lines(lines, color_img, (0,255,0))
    cv2.imshow('Hough', color_img)
    cv2.waitKey()
