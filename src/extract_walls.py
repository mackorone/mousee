import cv2
import numpy as np
from hough import *
from get_square import *

def get_intersection_points(img):

    ave_lines = hough_lines(img)
    xy_lines = rho_theta_to_x1y1_x2y2(ave_lines, np.shape(img))
    group1, group2 = get_groups_of_lines(xy_lines, np.shape(img))

    points = []
    for i in range(len(group1)):
        for j in range(len(group2)):
            if (intersection_within_image(group1[i], group2[j], np.shape(img))):
                points.append(line_intersection(group1[i], group2[j]))

    return points

def extract_walls(img):

    maze_width = 4
    maze_height = 6

    # Get all of the intersection points
    points = get_intersection_points(img)

    # First, sort and split the points by y-value
    points = sorted(points, key = lambda z: z[1])
    rows = [points[(maze_width+1)*i:(maze_width+1)*(i+1)] for i in range(7)]

    # Then sort by x-value
    rows = [sorted(r, key = lambda x: x[0]) for r in rows]

    # Create the red mask for determining whether or not walls exist
    lower_red = np.array([0,10,155])
    upper_red = np.array([10,255,255])

    # Window parameters
    win_offset = 15 # Offset from the vertex enpoints (so as to not include other walls)
    win_size = 25 # Height in the horizontal, width in the vertical

    def horizontal_wall_exists(img, left, right):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        y = int((left[1] + right[1])/2)
        window = hsv[y-win_size:y+win_size, left[0]+win_offset:right[0]-win_offset, :]
        red_mask = cv2.inRange(window, lower_red, upper_red)
        return sum(sum(red_mask)) > 500

    def vertical_wall_exists(img, top, bottom):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        x = int((top[0] + bottom[0])/2)
        window = hsv[top[1]+win_offset:bottom[1]-win_offset, x-win_size:x+win_size, :]
        red_mask = cv2.inRange(window, lower_red, upper_red)
        return sum(sum(red_mask)) > 500

    # 2D array of the wall values
    walls = [[[0,0,0,0] for i in range(maze_width)] for j in range(maze_height)]

    # Check all pairs of points along the horizonal
    for i in range(len(rows)): # TODO: The indices can be changed
        for j in range(len(rows[i])-1): # TODO: The indices can be changed
            if horizontal_wall_exists(img, rows[i][j], rows[i][j+1]):
                # TODO: Delete this
                '''
                center = ((rows[i][j][0] + rows[i][j+1][0])/2, (rows[i][j][1] + rows[i][j+1][1])/2)
                cv2.circle(img, center, 8, (0, 0, 255), -1)
                '''
                pass
                # TODO: Add to walls

    # Check all pairs of points along the vertical
    for i in range(len(rows[0])): # TODO: The indices can be changed
        for j in range(len(rows)-1): # TODO: The indices can be changed
            if vertical_wall_exists(img, rows[j][i], rows[j+1][i]):
                # TODO: Delete this
                '''
                center = ((rows[j][i][0] + rows[j+1][i][0])/2, (rows[j][i][1] + rows[j+1][i][1])/2)
                cv2.circle(img, center, 8, (0, 0, 255), -1)
                '''
                pass
                # TODO: Add to walls
    # TODO: Delete this
    '''
    cv2.imshow('Walls', img)
    cv2.waitKey()
    '''

    return walls

# Demo
if __name__ == '__main__':

    #image_path = 'topdown_cropped.jpg'
    image_path = '3.JPG'
    img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)

    # We need a sparate image to draw on
    draw_img = img.copy()

    # Draw the corners
    points = get_intersection_points(img)
    for p in points:
        cv2.circle(draw_img, p, 8, (0, 255, 0), -1)

    # Draw the walls
    walls = extract_walls(img)
    #center = ((rows[i][j][0] + rows[i][j+1][0])/2, (rows[i][j][1] + rows[i][j+1][1])/2)
    print walls

    cv2.imshow('Walls', draw_img)
    cv2.waitKey()
