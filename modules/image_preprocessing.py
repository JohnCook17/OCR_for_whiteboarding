#!/usr/bin/python3

import os
from character_segmentation import char_segmentation
from line_segmentation import line_finder
# This is a library we need to do some math on the image to be able to give it
# to the MLP in the right format
try:
    import cv2
except ImportError:
    os.system("pip3 install opencv-python")
    import cv2
# This library allows us to do math and use complex arrays
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
        # STEP 1.2
        # This code processes all the scanned images and adds them to the
        # handwritten_story
        handwritten_story = []
        for i in files:
            img = i
            #img = (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY |
            #                                    cv2.THRESH_OTSU)
            cv2.imwrite("images/greyscale/gs_img" + str(i) + ".jpg", img)
            handwritten_story.append(img)
        print("Imported the scanned images.")

        # STEP 1.3
        """
        iterator = 0
        typed_story = ""
        for letter in handwritten_story:
            if numpy.any(letter):
                letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("images/resize/letter" + str(iterator) + ".jpg", letter)
                single_item_array = (numpy.array(letter)).reshape(1, 784)
                prediction = mlp2.predict(single_item_array)
                typed_story = typed_story + str(chr(prediction[0]+96))
            iterator += 1
        print("Conversion to typed story complete!")
        print("V1")
        print(typed_story)
        """
        # STEP 1.4
        """
        typed_story = ""
        for letter in handwritten_story:
            if numpy.any(letter):
                letter = cv2.resize(letter, (28, 28), interpolation=cv2.INTER_CUBIC)
                # this bit of code checks to see if the image is just a blank space by
                # looking at the color of all the pixels summed
                total_pixel_value = 0
                for j in range(28):
                    for k in range(28):
                        total_pixel_value += letter[j, k]
                if total_pixel_value < 20:
                    typed_story = typed_story + " "
                # if it NOT a blank, it actually runs the prediction algorithm on it
                else:
                    single_item_array = (numpy.array(letter)).reshape(1, 784)
                    prediction = mlp2.predict(single_item_array)
                    typed_story = typed_story + str(chr(prediction[0] + 96))
        print("Conversion to typed story complete!")
        print("V2")
        print(typed_story)
        """
        # STEP 1.5

        # These steps process the scanned images to be in the same format and have the
        # same properties as the EMNIST images
        # They are described by the EMNIST authors in detail here:
        # https://arxiv.org/abs/1702.05373v1
        iterator = 0
        for img in handwritten_story:
            if numpy.any(img):
                # step 1: Apply Gaussian blur filter
                img = cv2.GaussianBlur(img, (7, 7), 0)
                cv2.imwrite("images/post_gaussian_blur/img" + str(iterator) + ".jpg", img)
                # steps 2 and 3: Extract the Region of Interest in
                # the image and center in square
                points = cv2.findNonZero(img)
                x, y, w, h = cv2.boundingRect(points)
                if (w > 0 and h > 0):
                    if w > h:
                        y = y - (w-h)//2
                        img = img[y:y+w, x:x+w]
                    else:
                        x = x - (h-w)//2
                        img = img[y:y+h, x:x+h]

                cv2.imwrite("images/center/img" + str(iterator) + ".jpg", img)

                # step 4: Resize and resample to be 28 x 28 pixels
                img = cv2.resize(img, (28, 28), interpolation=cv2.INTER_CUBIC)
                cv2.imwrite("images/resize_and_resample/img" + str(iterator) + ".jpg", img)
                """
                #step 5: Normalize pixels and reshape before adding to the new story array
                img = img/255
                img = img.reshape((28,28))
                cv2.imwrite("images/normalize/img" + str(iterator) + ".jpg", img)
                """
                #img = skeleton(img)
                #img = cv2.GaussianBlur(img, (7, 7), 0)
                #img = (thresh, img) = cv2.threshold(img, 224, 255, cv2.THRESH_BINARY |
                #                                    cv2.THRESH_OTSU)
                cv2.imwrite("images/skel/img" + str(iterator) + ".jpg", img)

                processed_story.append(img)
            iterator += 1
        processed_story.append(white_square)

    print("Processed the scanned images.")
    #saving images to disk for easier debugging
    iterator_1 = 0
    for letter_image in processed_story:
        cv2.imwrite("images/final_image/img" + str(iterator_1) + ".jpg", letter_image)
        iterator_1 += 1
    # STEP 1.6

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
    print("V3")
    print(typed_story)
