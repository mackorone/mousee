import cv2
import numpy as np
from hough import *
from get_square import *

def get_intersection_points(img):

    # Group here again to ensure we the correct amount of lines
    ave_lines = hough_lines(img)
    xy_lines = rho_theta_to_x1y1_x2y2(ave_lines, np.shape(img))
    group1, group2 = get_groups_of_lines(xy_lines, np.shape(img), True) # Filter incorrect lines

    # Note: We expect lines to be perfect here (i.e. no extra lines), thus we filter

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
    win_offset = 25 # Offset from the vertex enpoints (so as to not include other walls)
    win_size = 25 # Height in the horizontal, width in the vertical
    thresh = 750

    def horizontal_wall_exists(img, left, right):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        y = int((left[1] + right[1])/2)
        window = hsv[max(y-win_size, 0):min(y+win_size, np.shape(img)[0]), left[0]+win_offset:right[0]-win_offset, :]
        red_mask = cv2.inRange(window, lower_red, upper_red)
        return sum(sum(red_mask)) > thresh

    def vertical_wall_exists(img, top, bottom):
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        x = int((top[0] + bottom[0])/2)
        window = hsv[top[1]+win_offset:bottom[1]-win_offset, max(x-win_size,0):min(x+win_size, np.shape(img)[1]), :]
        red_mask = cv2.inRange(window, lower_red, upper_red)
        return sum(sum(red_mask)) > thresh

    # 2D array of the wall values, rows and cols indexed like a matrix, NESW order for the walls
    walls = [[[0,0,0,0] for i in range(maze_width)] for j in range(maze_height)]

    # Initialize the outer walls
    for r in range(len(walls)):
        for c in range(len(walls[r])):
            if r == 0:
                walls[r][c][0] = 1
            if c == 0:
                walls[r][c][3] = 1
            if r == maze_height-1:
                walls[r][c][2] = 1
            if c == maze_width-1:
                walls[r][c][1] = 1

    # Check all pairs of points along the horizonal
    for i in range(1, len(rows)-1):
        for j in range(len(rows[i])-1):
            if horizontal_wall_exists(img, rows[i][j], rows[i][j+1]):
                walls[i-1][j][2] = 1
                walls[i][j][0] = 1
                center = ((rows[i][j][0] + rows[i][j+1][0])/2, (rows[i][j][1] + rows[i][j+1][1])/2)
                cv2.circle(img, center, 8, (0, 0, 255), -1)

    # Check all pairs of points along the vertical
    for j in range(1, len(rows[0])-1):
        for i in range(len(rows)-1):
            if vertical_wall_exists(img, rows[i][j], rows[i+1][j]):
                walls[i][j-1][1] = 1
                walls[i][j][3] = 1
                center = ((rows[i][j][0] + rows[i+1][j][0])/2, (rows[i][j][1] + rows[i+1][j][1])/2)
                cv2.circle(img, center, 8, (0, 0, 255), -1)

    cv2.imwrite('asdf.jpg', img)

    # Return the rows as well, for drawing purposes
    return walls, rows

# Demo
if __name__ == '__main__':

    # This function should be used on the final, large image
    #image_path = 'imgs/large.jpg'
    image_path = 'final.jpg'
    img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)

    # We need a sparate image to draw on
    draw_img = img.copy()

    # Draw the corners
    points = get_intersection_points(img)
    for p in points:
        cv2.circle(draw_img, p, 8, (255, 0, 0), -1)

    # Draw the walls
    walls, rows = extract_walls(img)

    # TODO
    '''
    # Check all pairs of points along the horizontal
    for i in range(len(rows)):
        for j in range(len(rows[i])-1):
            if walls[i-1][j][2] == 1:
                center = ((rows[i][j][0] + rows[i][j+1][0])/2, (rows[i][j][1] + rows[i][j+1][1])/2)
                cv2.circle(draw_img, center, 8, (0, 0, 255), -1)

    # Check all pairs of points along the vertical
    for j in range(len(rows[0])):
        for i in range(len(rows)-1):
            if walls[i][j-1][3] == 1:
                center = ((rows[i][j][0] + rows[i+1][j][0])/2, (rows[i][j][1] + rows[i+1][j][1])/2)
                cv2.circle(draw_img, center, 8, (0, 0, 255), -1)
    '''

    cv2.imwrite('Walls.jpg', draw_img)
    #cv2.imshow('Walls', draw_img)
    #cv2.waitKey()
