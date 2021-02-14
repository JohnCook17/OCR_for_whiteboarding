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
    """ A function to combine two images"""
    # make a copy of the forground and background
    foreground, background = img1.copy(), img2.copy()
    # set the data type of the arrays
    foreground = numpy.float32(foreground)
    background = numpy.float32(background)
    # get the height and width of the background and forground
    background_height = background.shape[1]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    # make sure anchors are not outside the bounds of the image
    if ((foreground_height+anchor_y > background_height or
         foreground_width+anchor_x > background_width)):
        raise ValueError("The foreground image exceeds the background" +
                         "boundaries at this location")
    # set the alpha level of the images to be combined in this case the
    # forground is supposed to cover the background so it is set to 1
    alpha = 1

    # do composite of the two images at specified location
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
    # returns the composite of the two images
    return background


# Unused code
# def append_space(start, end, letter_width):
    # square = numpy.zeros((28, 28))
    # print(start)
    # print(end)
    # space_width = end - start
    # print("space vs letter below")
    # print(space_width)
    # print(letter_width)
    # if math.isclose(space_width, letter_width, abs_tol=1):
    # print("adding space")
    # return square
# End unused code

def char_segmentation(img):
    """ Gets each individual letter and makes each its own image"""
    # Init img_list to empty
    img_list = []
    # make a copy of the img and blur it
    img2 = cv2.GaussianBlur(img, (7, 7), 0)
    # get the demensions of the image
    height, width = img2.shape
    # set start and end to 0 to indicate not found
    start = 0
    end = 0
    # scan vertically one pixel at a time
    for w in range(width):
        # controlls the size of the vertical slice and its location
        vertical_slice = img2[0: height, w:w + 1]
        # if the mean is low and start has not been found
        # start equals current location and end is set to off
        if vertical_slice.mean() < 1 and start == 0:
            end_prev = end
            end = 0
            start = w
        # if start is on and end is not and the slice contains a letter
        # then end is set to the current location and the letter image is set
        if vertical_slice.mean() == 0 and start != 0 and end == 0:
            end = w
            letter = img[0: height, start:end]
            letter_height, letter_width = letter.shape
            # checks if the image is valid and if the mean is not low
            # then center the image some
            if numpy.any(letter) and letter.mean() > .25:
                # finds the letter on the background
                M = cv2.moments(letter)
                # finds the midpoint of the letter, not the image
                letter_mid = int(M["m01"] / M["m00"])
                # the location to travel up from
                letter_up = letter_mid
                # the location to travel down from
                letter_down = letter_mid
                # the mean of the slice of the image from the middle
                # working up
                top = letter[letter_up - 1: letter_up, 0: letter_width].mean()
                # the mean of the slice of the image from the middle
                # working down
                bottom = letter[letter_down: letter_down + 1,
                                0: letter_width].mean()
                # go until the top of the letter, not the image
                while top > 0:
                    if letter_up - 1 > 0:
                        top = letter[letter_up - 1: letter_up,
                                     0: letter_width].mean()
                        letter_up -= 1
                    if letter_up - 1 == 0 or letter_up - 1 < 0:
                        break
                # go until the bottom of the letter, not the image
                while bottom > 0:
                    if letter_down + 1 < letter_height:
                        bottom = letter[letter_down: letter_down + 1,
                                        0: letter_width].mean()
                        letter_down += 1
                    if ((letter_down + 1 == letter_height or
                         letter_down + 1 > letter_height)):
                        break
                # crops the image of the letter to have less background
                letter = letter[letter_up: letter_down, 0: letter_width]
                # sets the square size
                if letter_down - letter_up > letter_width:
                    square_size = 2 * (letter_down - letter_up)
                else:
                    square_size = 2 * letter_width
                # sets the square background size
                square = numpy.zeros((square_size, square_size))
                # combines the new background with the image and kinda
                # centers it
                letter = combine_two_images(letter, square, letter_width // 2,
                                            (letter_down - letter_up) // 2)
                # appends the letter to the image list to process by the mlp
                img_list.append(letter)
            # turns start off
            start = 0
    cv2.imwrite("line_of_letters/inverted_and_blured.jpg", img2)
    # i = 0
    # for img in img_list:
    # i += 1
    # cv2.imwrite("test/" + str(i) + ".jpg", img)
    return img_list
