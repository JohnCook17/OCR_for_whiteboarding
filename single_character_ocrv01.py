#!/usr/bin/python3
print("Importing libraries.")
import os
from modules.demo import demo
from modules.main_training_module import training
from modules.image_preprocessing import processing
from character_segmentation import char_segmentation
try:
    from emnist import extract_training_samples
except ImportError:
    os.system("git clone https://github.com/sorki/python-mnist")
    os.system("./python-mnist/get_data.sh")
    os.system("pip3 install emnist")
    from emnist import extract_training_samples

# STEP 1.1
# Grab the data from the OpenML website
# X will be our images and y will be the labels
X, y = extract_training_samples("letters")

# Make sure that every pixel in all of the images is a value between 0 and 1
X = X / 255.

# Use the first 60000 instances as training and the next 10000 as testing
X_train, X_test = X[:60000], X[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# There is one other thing we need to do, we need to
# record the number of samples in each dataset and the number of pixels in
# each image
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)

my_input = "waiting"
while(my_input != "0"):
    print("Enter a coresponding number for that option")
    print("1. Demo")
    print("2. Run Main training algorithm")
    print("3. Delete Demo training data")
    print("4. Delete Main traning data")
    print("5. Help")
    print("0. exit")
    my_input = input()
    if my_input == "1":
        demo(X_train, y_train, X_test, y_test)
    elif my_input == "2":
        char_segmentation()
        mlp2 = training(X_train, y_train, X_test, y_test)
        processing(mlp2)
    elif my_input == "3":
        print("Are you sure? type yes to confirm")
        if input() == "yes":
            print("deleting")
            os.replace("mlp1.joblib", "backups/mlp1.joblib")
        else:
            print("changed your mind?")
    elif my_input == "4":
        print("Are you sure? type yes to confirm")
        if input() == "yes":
            print("deleting")
            os.replace("mlp2.joblib", "backups/mlp2.joblib")
        else:
            print("changed your mind?")
    elif my_input == "5":
        print("please put images in single_letters to have them recognized.")
    