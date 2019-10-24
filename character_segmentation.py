#!/usr/bin/python3
import os
import math
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


def combine_two_images(img1, img2, anchor_y, anchor_x):

    foreground, background = img1.copy(), img2.copy()

    foreground = numpy.float32(foreground)
    background = numpy.float32(background)

    background_height = background.shape[1]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    if foreground_height+anchor_y > background_height or
    foreground_width+anchor_x > background_width:
        raise ValueError("The foreground image exceeds the background" +
                         "boundaries at this location")

    alpha = 1

    # do composite at specified location
    start_y = anchor_y
    start_x = anchor_x
    end_y = anchor_y+foreground_height
    end_x = anchor_x+foreground_width
    blended_portion = cv2.addWeighted(foreground,
                                      alpha,
                                      background[start_y:end_y, start_x:end_x],
                                      1 - alpha,
                                      0,
                                      background)
    background[start_y:end_y, start_x:end_x] = blended_portion
    return background


def append_space(start, end, letter_width):
    square = numpy.zeros((28, 28))
    print(start)
    print(end)
    space_width = end - start
    print("space vs letter below")
    print(space_width)
    print(letter_width)
    if math.isclose(space_width, letter_width, abs_tol=1):
        print("adding space")
        return square


def char_segmentation(img):
    img_list = []
    img2 = cv2.GaussianBlur(img, (7, 7), 0)
    height, width = img2.shape
    start = 0
    end = 0
    for w in range(width):
        vertical_slice = img2[0: height, w:w + 1]
        if vertical_slice.mean() < 1 and start == 0:
            end_prev = end
            end = 0
            start = w
        if vertical_slice.mean() == 0 and start != 0 and end == 0:
            end = w
            letter = img[0: height, start:end]
            letter_height, letter_width = letter.shape
            # make this if dynamic by using the letter mean at some point
            if letter.mean() > .25:
                M = cv2.moments(letter)
                letter_mid = int(M["m01"] / M["m00"])
                letter_up = letter_mid
                letter_down = letter_mid
                top = letter[letter_up - 1: letter_up, 0: letter_width].mean()
                bottom = letter[letter_down: letter_down + 1,
                                0: letter_width].mean()
                while top > 0:
                    if letter_up - 1 > 0:
                        top = letter[letter_up - 1: letter_up,
                                     0: letter_width].mean()
                        letter_up -= 1
                    if letter_up - 1 == 0 or letter_up - 1 < 0:
                        break
                while bottom > 0:
                    if letter_down + 1 < letter_height:
                        bottom = letter[letter_down: letter_down + 1,
                                        0: letter_width].mean()
                        letter_down += 1
                    if letter_down + 1 == letter_height or
                    letter_down + 1 > letter_height:
                        break
                letter = letter[letter_up: letter_down, 0: letter_width]
                if letter_down - letter_up > letter_width:
                    square_size = 2 * (letter_down - letter_up)
                else:
                    square_size = 2 * letter_width
                square = numpy.zeros((square_size, square_size))
                letter = combine_two_images(letter, square, letter_width // 2,
                                            (letter_down - letter_up) // 2)
                img_list.append(letter)
            start = 0
    cv2.imwrite("line_of_letters/inverted_and_blured.jpg", img2)
    i = 0
    for img in img_list:
        i += 1
        cv2.imwrite("test/" + str(i) + ".jpg", img)
    return img_list
