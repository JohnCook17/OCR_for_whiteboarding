#!/usr/bin/python3
print("Importing libraries.")
import os
from modules.demo import demo
from modules.main_training_module import training
from modules.image_preprocessing import processing
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

print("Extracted our samples and divided our training and testing data sets")


demo(X_train, y_train, X_test, y_test)
mlp2 = training(X_train, y_train, X_test, y_test)
processing(mlp2)

