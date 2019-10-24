#!/usr/bin/python3
import os
# These two lines import the ML libraries we need
try:
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix
except ImportError:
    os.system("pip3 install -U scikit-learn")
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix
# This is a library we need to do some math on the image to be able to give it
# to the MLP in the right format and dislay the confusion matrix
try:
    import cv2
except ImportError:
    os.system("pip3 install opencv-python")
    import cv2
# This is the library we need to save the state of the Neural Network
try:
    from joblib import dump, load
except ImportError:
    os.system("pip3 install joblib")
    from joblib import dump, load
try:
    from emnist import extract_training_samples
except ImportError:
    os.system("git clone https://github.com/sorki/python-mnist")
    os.system("./python-mnist/get_data.sh")
    os.system("pip3 install emnist")
    from emnist import extract_training_samples

# STEP 1.1
# Grab the data from the OpenML website
# x will be our images and y will be the labels
x, y = extract_training_samples("letters")

# Make sure that every pixel in all of the images is a value between 0 and 1
x = x / 255.

# Use the first 60000 instances as training and the next 10000 as testing
x_train, x_test = x[:60000], x[60000:70000]
y_train, y_test = y[:60000], y[60000:70000]

# There is one other thing we need to do, we need to
# record the number of samples in each dataset and the number of pixels in
# each image
x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

def training():
    if not os.path.exists("mlp2.joblib"):
        mlp2 = MLPClassifier(hidden_layer_sizes=(250,
                                                250,
                                                250,
                                                250,
                                                250,
                                                250,),
                            max_iter=1000, alpha=1e-4,
                            solver="sgd", verbose=1, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
        mlp2.fit(x_train, y_train)
        print("Training set score: %f" % mlp2.score(x_train, y_train))
        print("Test set score: %f" % mlp2.score(x_test, y_test))
        
        dump(mlp2, "mlp2.joblib")
    else:
        mlp2 = load("mlp2.joblib")
        y_pred = mlp2.predict(x_test)
        cm = confusion_matrix(y_test, y_pred)
        cv2.imwrite("confusion matrix2.jpg", cm)
    return mlp2