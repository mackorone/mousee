import os
import sys
import numpy as np
from hough import *
from ortho import *
from stitch import *
from extract_walls import *

# Instructions on how to use this script
if len(sys.argv) != 3:
    print 'Usage: python driver.py <input_directory> <output_file.txt>'
    exit(0)

# Get the args, perform sanity check
input_directory, output_file = sys.argv[1:3]
if not os.path.isdir(input_directory):
    print '"' + input_directory + '" does not exist'
    exit(1)

# Get the imgs
min_num_imgs = 4
max_num_imgs = 6
imgs = [f for f in os.listdir(input_directory) if not os.path.isdir(input_directory + '/' + f)]
if not min_num_imgs <= len(imgs) <= max_num_imgs :
    print ('The input directory, ' + input_directory + ', needs to contain ['
           + str(min_num_imgs) + ',' + str(max_num_imgs) + '] images')
    exit(1)

# Get confirmation
print ('Preparing the use following images to generate maze file,'
       " assuming they're in left to right order:")
for img in imgs:
    print img
print 'Continue? [Y/n]'
while True:
    response = raw_input()
    if response.lower() in ['y', 'yes']:
        break
    elif response.lower() in ['n', 'no']:
        exit(0)

# Pair off the images
img_groups = [list(t) for t in zip(imgs[0::2], imgs[1::2])]
if 2*len(img_groups) < len(imgs):
    img_groups[-1].append(imgs[-1])

# First, stitch the img_groups together
print 'Performing preliminary stitching...'
stitched_imgs = []
for g in img_groups:
    i1 = cv2.imread(input_directory + '/' + g[0])
    i2 = cv2.imread(input_directory + '/' + g[1])
    result = stitch(i1, i2)
    if len(g) == 3:
        result = stitch(result, cv2.imread(input_directory + '/' + g[2]))
    stitched_imgs.append(result)

# Perform the orthographic transforms
print 'Performing homographies...'
ortho_imgs = [persp_to_ortho(img) for img in stitched_imgs]

# Stitch the ortho images to get the final image
print 'Performing final stitching...'
final_img = stitch(ortho_imgs[0], ortho_imgs[1])
for ortho in ortho_imgs[2:]:
    final_img = stitch(final_img, ortho)
    
# Do the wall detection
print 'Detecting walls...'
cv2.imwrite('final.jpg', final_img)
walls = extract_walls(final_img)
print walls
print 'DONE'
