#!/usr/bin/python3
import os
try:
    import cv2
except ImportError:
    os.system("pip3 install opencv-python")
    import cv2
try:
    import numpy
except ImportError:
    os.system("pip3 install numpy")

def line_finder():
    list_of_lines = []
    img = cv2.imread("image_to_ocr/alphabet_blue.jpg", cv2.IMREAD_GRAYSCALE)
    img = (255 - img)
    img = (thresh, img) = cv2.threshold(img, 224, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("line_of_letters/invert.jpg", img)
    height, width = img.shape
    start = 0
    end = 0
    for h in range(height):
        horizontal_slice = img[h: h + 1, 0: width]
        #print(horizontal_slice.mean())
        horizontal_slice.mean()
        if horizontal_slice.mean() < 5 and start == 0:
            end = 0
            start = h
        if horizontal_slice.mean() == 0 and start != 0 and end == 0:
            end = h
            if start < end:
                line = img[start - 1: end + 1, 0: width]
                cv2.imwrite("line_of_letters/test" + str(h) + ".jpg", line)
                list_of_lines.append(line)
            start = 0
    return list_of_lines
