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
    import numpy

def char_segmentation():
    img = cv2.imread("line_of_letters/A_G.jpg", cv2.IMREAD_GRAYSCALE)
    img = (255 - img)
    img = (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    img2 = cv2.GaussianBlur(img, (7, 7), 0)
    height, width = img2.shape
    letter_number = 0
    start = 0
    end = 0
    for w in range(width):
        vertical_slice = img2[0:0 + height, w:w + 1]
        print(vertical_slice.mean())
        if vertical_slice.mean() < 1 and start == 0:
            end = 0
            start = w
        if vertical_slice.mean() == 0 and start != 0 and end == 0:
            end = w
            letter = img[0:0 + height, start:end]
            if letter.mean() > 5:
                cv2.imwrite("line_of_letters/" + str(letter_number) + ".jpg", letter)
                letter_number += 1
            start = 0
        #cv2.imwrite("line_of_letters/slice" + str(i) + ".jpg", vertical_slice)
    cv2.imwrite("line_of_letters/inverted_and_blured.jpg", img2)

char_segmentation()