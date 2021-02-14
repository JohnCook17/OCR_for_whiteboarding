#!/usr/bin/python3
import os
try:
    import cv2
except ImportError:
    os.system("pip3 install opencv-python")
    import cv2


def line_finder():
    """ Finds a new line in an image of characters """
    # step 1.1 init an empty list of lines
    list_of_lines = []
    # step 1.2 use cv2 to read the image in as a greyscale image
    img = cv2.imread("image_to_ocr/img.jpg", cv2.IMREAD_GRAYSCALE)
    # invert the colors
    img = (255 - img)
    # threshold the image
    img = (thresh, img) = cv2.threshold(img, 228, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("test/t1.jpg", img)
    cv2.imwrite("test/t2.jpg", img)
    # blur the image
    img = cv2.GaussianBlur(img, (31, 31), 0)
    cv2.imwrite("test/t3.jpg", img)
    # get the height and width of the image
    height, width = img.shape
    # init start and end
    start = 0
    end = 0
    # scans the image horizontally one pixel at a time
    for h in range(height):
        # sets the size and location of the slice
        horizontal_slice = img[h: h + 1, 0: width]
        # if the average pixel value is low and start has not been found yet
        # set end to off and start to current location
        if horizontal_slice.mean() < 5 and start == 0:
            end = 0
            start = h
        # if the average is 0 and start has been found but end has not
        # been found
        # set end to current location and sets start to not found
        if horizontal_slice.mean() == 0 and start != 0 and end == 0:
            end = h
            # this line is for making sure that the line is always read from
            # start to
            # end and that there are errors
            if start < end:
                # defines where a line is making sure that start and
                # end are not outside the bounds of the image
                line = img[start - 1: end + 1, 0: width]
                # if line contains letters append it to list of lines
                if line.mean() > 1:
                    list_of_lines.append(line)
            start = 0
    # j = 0
    # for i in list_of_lines:
    # cv2.imwrite("test/" + str(j) + "line.jpg", i)
    # j += 1
    return list_of_lines
