#!/usr/bin/python3
import os
try:
    from sklearn.neural_network import MLPClassifier
except ImportError:
    os.system("pip3 install -U scikit-learn")
    from sklearn.neural_network import MLPClassifier
try:
    import cv2
except ImportError:
    os.system("pip3 install opencv-python")
    import cv2
try:
    from joblib import dump, load
except ImportError:
    os.system("pip3 install joblib")
    from joblib import dump, load
try:
    import numpy
except ImportError:
    os.system("pip3 install numpy")
    import numpy
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

def demo():
    """
    A demo of my machine learning program
    """
    if not os.path.exists("mlp1.joblib"):
        # This creates our first MLP with 1 hidden layer with 50 neurons and sets
        # it to run through the data 20 times
        mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                            solver="sgd", verbose=10, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
        print("Created our first MLP network")

        # STEP 1.1
        mlp1.fit(x_train, y_train)
        print("Training set score: %f" % mlp1.score(x_train, y_train))
        print("Test set score: %f" % mlp1.score(x_test, y_test))

        # STEP 1.2
        # First let"s initialize a list with all the predicted values from the
        # training set
        y_pred = mlp1.predict(x_test)

        # STEP 1.3
        # You can change this to any letters that you think the neural network may
        # have confused...
        predicted_letter = "l"
        actual_letter = "i"
        # This code counts all mistakes for the letters above
        mistake_list = []
        for i in range(len(y_test)):
            if (y_test[i] == (ord(actual_letter) - 96) and
            y_pred[i] == (ord(predicted_letter) - 96)):
                mistake_list.append(i)
        print("There were " +
            str(len(mistake_list)) +
            " times that the letter " +
            actual_letter +
            " was predicted to be the letter " +
            predicted_letter +
            ".")
        dump(mlp1, "mlp1.joblib")
    else:
        mlp1 = load("mlp1.joblib")
    # STEP 2.1
    if not os.path.exists("lab1-neural-networks/letters_mod"):
        print("demo data not found getting data")
        # Pulls the scanned data set from GitHub
        os.system("git clone https://github.com/crash-course-ai/lab1-neural-networks.git")
        os.system("git pull")
        os.system("ls lab1-neural-networks/letters_mod")
        os.system("pwd")
        os.system("cd lab1-neural-networks/letters_mod")
        os.system("pwd")
    # Puts all the data in the "files" variable
    path, dirs, files = next(os.walk("lab1-neural-networks/letters_mod/"))
    files.sort()

    # STEP 2.2
    # This code processes all the scanned images and adds them to the handwritten_story
    handwritten_story = []
    for i in range(len(files)):
        img = cv2.imread("lab1-neural-networks/letters_mod/"+files[i],cv2.IMREAD_GRAYSCALE)
        handwritten_story.append(img)

    print("Imported the scanned images.")
    # STEP 2.3

    typed_story = ""
    for letter in handwritten_story:
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
            prediction = mlp1.predict(single_item_array)
            typed_story = typed_story + str(chr(prediction[0] + 96))
    print("Conversion to typed story complete!")
    print(typed_story)