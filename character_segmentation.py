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

def combine_two_images(img1, img2, anchor_y, anchor_x):

    foreground, background = img1.copy(), img2.copy()

    foreground = numpy.float32(foreground)
    background = numpy.float32(background)

    background_height = background.shape[1]
    background_width = background.shape[1]
    foreground_height = foreground.shape[0]
    foreground_width = foreground.shape[1]
    if foreground_height+anchor_y > background_height or foreground_width+anchor_x > background_width:
        raise ValueError("The foreground image exceeds the background boundaries at this location")

    alpha =1

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

def char_segmentation(img):
    img_list = []
    """
    img = cv2.imread("line_of_letters/alphabet_2.jpg", cv2.IMREAD_GRAYSCALE)
    img = (255 - img)
    img = (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    """
    img2 = cv2.GaussianBlur(img, (7, 7), 0)
    height, width = img2.shape
    letter_number = 0
    start = 0
    end = 0
    for w in range(width):
        vertical_slice = img2[0: height, w:w + 1]
        #print(vertical_slice.mean())
        if vertical_slice.mean() < 1 and start == 0:
            end = 0
            start = w
        if vertical_slice.mean() == 0 and start != 0 and end == 0:
            end = w
            letter = img[0: height, start:end]
            #make this if statment dynaic by using the letter mean at some point
            if numpy.any(letter):
                if letter.mean() > .25:
                    #new code
                    M = cv2.moments(letter)
                    letter_mid = int(M["m01"] / M["m00"])
                    letter_up = letter_mid
                    letter_down = letter_mid
                    #print("++++++++++++++++++")
                    #print(letter_mid)
                    #print("++++++++++++++++++")
                    #end new code
                    letter_height, letter_width = letter.shape
                    top = letter[letter_up - 1: letter_up, 0: letter_width].mean()
                    bottom = letter[letter_down: letter_down + 1, 0: letter_width].mean()
                    while top > 0:
                        #print("top++++++++++++++++++++++++++++")
                        #print(top)
                        if letter_up - 1 > 0:
                            top = letter[letter_up - 1: letter_up, 0: letter_width].mean()
                            letter_up -= 1
                        if letter_up - 1 == 0 or letter_up - 1 < 0:
                            break
                    while bottom > 0:
                        #print("letter_height")
                        #print(letter_height)
                        #print("letter_down")
                        #print(letter_down)
                        #print("bottom+++++++++++++++++++++++++")
                        #print(bottom)
                        if letter_down + 1 < letter_height:
                            bottom = letter[letter_down: letter_down + 1, 0: letter_width].mean()
                            letter_down += 1
                        if letter_down + 1 == letter_height or letter_down + 1 > letter_height:
                            break
                    letter = letter[letter_up: letter_down, 0: letter_width]
                    if letter_down - letter_up > letter_width:
                        square_size = 2 * (letter_down - letter_up)
                    else:
                        square_size = 2 * letter_width
                    square = numpy.zeros((square_size, square_size))
                    letter = combine_two_images(letter, square, letter_width // 2, (letter_down - letter_up) // 2)
                    cv2.imwrite("single_letters/" + str(letter_number) + ".jpg", letter)
                    letter_number += 1
                    img_list.append(letter)
            start = 0
        #cv2.imwrite("line_of_letters/slice" + str(i) + ".jpg", vertical_slice)
    cv2.imwrite("line_of_letters/inverted_and_blured.jpg", img2)
    return img_list
