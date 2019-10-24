#!/usr/bin/python3
import os
from character_segmentation import char_segmentation
from line_segmentation import line_finder
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

def processing(mlp2):
    # STEP 1.1
    processed_story = []
    lines = line_finder()
    white_square = cv2.imread("special_char/white_square.jpg", cv2.IMREAD_GRAYSCALE)
    for line in lines:
        files = char_segmentation(line)
        handwritten_story = []
        for img in files:
            handwritten_story.append(img)
        print("Importing the scanned images.")
        # STEP 1.2
        iterator = 0
        for img in handwritten_story:
            points = cv2.findNonZero(img)
            x, y, w, h = cv2.boundingRect(points)
            if (w > 0 and h > 0):
                if w > h:
                    y = y - (w-h)//2
                    img = img[y:y+w, x:x+w]
                else:
                    x = x - (h-w)//2
                    img = img[y:y+h, x:x+h]
            #saving images to disk for easier debugging
            cv2.imwrite("images/center/img" + str(iterator) + ".jpg", img)
            img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
            processed_story.append(img)
            iterator += 1
        processed_story.append(white_square)
    print("Processed the scanned images.")
    #saving images to disk for easier debugging
    iterator_1 = 0
    for letter_image in processed_story:
        cv2.imwrite("images/final_image/img" + str(iterator_1) + ".jpg", letter_image)
        iterator_1 += 1
    # STEP 1.3
    iterator = 0
    typed_story = ""
    for letter in processed_story:
        # this bit of code checks to see if the image is just a blank space
        # by looking at the color of all the pixels summed
        total_pixel_value = 0
        for j in range(28):
            for k in range(28):
                total_pixel_value += letter[j, k]
        if total_pixel_value < 20:
            typed_story = typed_story + " "
        elif letter.mean() == 255:
            typed_story = typed_story + "\n"
        # if it NOT a blank, it actually runs the prediction algorithm on it
        else:
            single_item_array = (numpy.array(letter)).reshape(1, 784)
            prediction = mlp2.predict(single_item_array)
            typed_story = typed_story + str(chr(prediction[0]+96))
            cv2.imwrite("images/letter_" +
                        str(iterator) +
                        ".jpg", processed_story[iterator])
        iterator += 1
    print("Conversion to typed story complete!")
    print(typed_story)
