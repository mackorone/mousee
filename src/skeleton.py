import cv2
import numpy as np

def get_red_mask(img):
    lower_red = np.array([0,10,155])
    upper_red = np.array([10,255,255])
    hsv = cv2.cvtColor(img.copy(), cv2.COLOR_BGR2HSV)
    return cv2.inRange(hsv, lower_red, upper_red)

def skeletonize(img):

    # Ensure we don't change the original image
    img = img.copy()

    # Create the red mask
    red_mask = get_red_mask(img)

    # Dilate the red mask
    d_k_s = 5 # dilation_kernel_size
    dilation_kernel = np.ones((d_k_s, d_k_s), np.float32)/d_k_s**2
    dilated = cv2.dilate(red_mask, dilation_kernel, iterations = 3)

    # Filter the image using the dilated red mask
    red_filtered = cv2.bitwise_and(img, img, mask = dilated)

    # Convert to binary and remove any small contours
    gray = cv2.cvtColor(red_filtered, cv2.COLOR_BGR2GRAY)
    ret, binary = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)
    contours, hier = cv2.findContours(gray, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 500:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(binary,(x,y),(x+w,y+h),0,-1)

    # Skeletonize
    size = np.size(binary)
    skel = np.zeros(binary.shape,np.uint8)
    ret, im = cv2.threshold(binary, 127, 255, 0)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while True:
        eroded = cv2.erode(im,element)
        temp = cv2.dilate(eroded,element)
        temp = cv2.subtract(im,temp)
        skel = cv2.bitwise_or(skel,temp)
        im = eroded.copy()
        zeros = size - cv2.countNonZero(im)
        if zeros == size:
            break

    # Now remove the noise from the skeleton
    contours, hier = cv2.findContours(skel.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        if cv2.contourArea(c) < 1:
            (x,y,w,h) = cv2.boundingRect(c)
            cv2.rectangle(skel,(x,y),(x+w,y+h),0,-1)

    return skel

# Demo
if __name__ == '__main__':
    image_path = '../data/m7/IMG_0290.JPG'
    color_img = cv2.imread(image_path, cv2.CV_LOAD_IMAGE_COLOR)
    cv2.imshow('Skeleton', skeletonize(color_img))
    cv2.waitKey()
