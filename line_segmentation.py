#!/usr/bin/python3
import os
try:
    import cv2
except ImportError:
    os.system("pip3 install opencv-python")
    import cv2


def line_finder():
    list_of_lines = []
    img = cv2.imread("image_to_ocr/Juno.jpg", cv2.IMREAD_GRAYSCALE)
    img = (255 - img)
    img = (thresh, img) = cv2.threshold(img, 228, 255,
                                        cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    cv2.imwrite("test/t1.jpg", img)
    cv2.imwrite("test/t2.jpg", img)
    img = cv2.GaussianBlur(img, (51, 51), 0)
    cv2.imwrite("test/t3.jpg", img)
    height, width = img.shape
    start = 0
    end = 0
    for h in range(height):
        horizontal_slice = img[h: h + 1, 0: width]
        if horizontal_slice.mean() < 5 and start == 0:
            end = 0
            start = h
        if horizontal_slice.mean() == 0 and start != 0 and end == 0:
            end = h
            if start < end:
                line = img[start - 1: end + 1, 0: width]
                if line.mean() > 1:
                    list_of_lines.append(line)
            start = 0
    j = 0
    for i in list_of_lines:
        print(j)
        cv2.imwrite("test/" + str(j) + "line.jpg", i)
        j += 1
    return list_of_lines
