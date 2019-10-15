#!/usr/bin/python3
print("Importing libraries.")
# STEP 1.1
import os
try:
    from emnist import extract_training_samples
except ImportError:
    os.system("git clone https://github.com/sorki/python-mnist")
    os.system("./python-mnist/get_data.sh")
    os.system("pip3 install emnist")
    from emnist import extract_training_samples
try:
    import matplotlib.pyplot as plt
except ImportError:
    os.system("pip3 install matplotlib")
    import matplotlib.pyplot as plt
# These three lines import the ML libraries we need
try:
    from sklearn.datasets import fetch_openml
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix
except ImportError:
    os.system("pip3 install -U scikit-learn")
    from sklearn.datasets import fetch_openml
    from sklearn.neural_network import MLPClassifier
    from sklearn.metrics import confusion_matrix
# This is a library we need to do some math on the image to be able to give it to the MLP in the right format
# These libraries let us import the letters, resize them, and print them out
# These are libraries we need to do some math on the image
# to be able to give it to the MLP in the right format and to resize it to 28x28 pixels
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
import numpy
from modules.skel import skel

print("Imported the libraries we need!")

# STEP 1.2
if not os.path.exists("mlp1.joblib"):
    # Grab the data from the OpenML website
    # X will be our images and y will be the labels
    X, y = extract_training_samples("letters")

    # Make sure that every pixel in all of the images is a value between 0 and 1
    X = X / 255.

    # Use the first 60000 instances as training and the next 10000 as testing
    X_train, X_test = X[:60000], X[60000:70000]
    y_train, y_test = y[:60000], y[60000:70000]

    # There is one other thing we need to do, we need to
    # record the number of samples in each dataset and the number of pixels in each image
    X_train = X_train.reshape(60000,784)
    X_test = X_test.reshape(10000,784)

    print("Extracted our samples and divided our training and testing data sets")

    # STEP 1.3


    img_index = 14000 # <<<<<  You can update this value to look at other images
    img = X_train[img_index]
    print("Image Label: " + str(chr(y_train[img_index]+96)))
    plt.imshow(img.reshape((28,28)))

    # STEP 2.1



    # This creates our first MLP with 1 hidden layer with 50 neurons and sets it to run through the data 20 times
    mlp1 = MLPClassifier(hidden_layer_sizes=(50,), max_iter=20, alpha=1e-4,
                        solver="sgd", verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)

    print("Created our first MLP network")

    # STEP 3.1

    mlp1.fit(X_train, y_train)
    print("Training set score: %f" % mlp1.score(X_train, y_train))
    print("Test set score: %f" % mlp1.score(X_test, y_test))

    # STEP 3.2

    # First let"s initialize a list with all the predicted values from the training set
    y_pred = mlp1.predict(X_test)

    # Now let"s visualize the errors between the predictions and the actual labels using a confusion matrix

    cm = confusion_matrix(y_test, y_pred)
    plt.matshow(cm)
    cv2.imwrite("confusion matrix.jpg", cm)


    # STEP 3.3

    # You can change this to any letters that you think the neural network may have confused...
    predicted_letter = "l"
    actual_letter = "i"


    # This code counts all mistakes for the letters above
    mistake_list = []
    for i in range(len(y_test)):
      if (y_test[i] == (ord(actual_letter) - 96) and y_pred[i] == (ord(predicted_letter) - 96)):
        mistake_list.append(i)
    print("There were " + str(len(mistake_list)) + " times that the letter " + actual_letter + " was predicted to be the letter " + predicted_letter + ".")

    # Once we know how many mistakes were made, we can change this to see an image of a particular one
    mistake_to_show = 4 # <<< e.g., change this to 3 if you want to see the 4th mistake

    # This code checks that the number mistake you asked for can be shown and if so, displays an image of it
    if (len(mistake_list)> mistake_to_show):
        img = X_test[mistake_list[mistake_to_show]]
        plt.imshow(img.reshape((28,28)))
    else:
          print("Couldn't show mistake number " + str(mistake_to_show + 1) + " because there were only " + str(len(mistake_list)) + " mistakes to show!")
    dump(mlp1, "mlp1.joblib")
else:
    mlp1 = load("mlp1.joblib")

# STEP 3.4
if not os.path.exists("mlp2.joblib"):

    # Change some of the values in the below statement and re-run to see how they 
    # affect performance!
    mlp2 = MLPClassifier(hidden_layer_sizes=(250,250,250,250,250,250,), max_iter=1000, alpha=1e-4,
                        solver="sgd", verbose=10, tol=1e-4, random_state=1,
                        learning_rate_init=.1)
    mlp2.fit(X_train, y_train)
    print("Training set score: %f" % mlp2.score(X_train, y_train))
    print("Test set score: %f" % mlp2.score(X_test, y_test))
    dump(mlp2, "mlp2.joblib")
else:
    mlp2 = load("mlp2.joblib")

# STEP 4.1
# Puts all the data in the "files" variable
path, dirs, files = next(os.walk("single_letters/"))
files.sort()

# STEP 4.2
# This code processes all the scanned images and adds them to the handwritten_story
handwritten_story = []

for i in range(len(files)):
  img = cv2.imread("single_letters/"+files[i],cv2.IMREAD_GRAYSCALE)
  img = (255 - img)
  img = (thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  cv2.imwrite("greyscale/gs_img" + str(i) + ".jpg", img)
  handwritten_story.append(img)
  

print("Imported the scanned images.")

#plt.imshow(handwritten_story[4])  #<--- Change this index to see different letters


# STEP 4.3
iterator = 0
typed_story = ""
for letter in handwritten_story:
    letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_CUBIC)
    cv2.imwrite("resize/letter" + str(iterator) + ".jpg", letter)
    single_item_array = (numpy.array(letter)).reshape(1,784)
    prediction = mlp2.predict(single_item_array)
    typed_story = typed_story + str(chr(prediction[0]+96))
    iterator += 1
    
print("Conversion to typed story complete!")
print("V1")
print(typed_story)

# STEP 4.4

typed_story = ""
for letter in handwritten_story:
  letter = cv2.resize(letter, (28,28), interpolation = cv2.INTER_CUBIC)
    
  #this bit of code checks to see if the image is just a blank space by looking at the color of all the pixels summed
  total_pixel_value = 0
  for j in range(28):
    for k in range(28):
      total_pixel_value += letter[j,k]
  if total_pixel_value < 20:
    typed_story = typed_story + " "
  else:         #if it NOT a blank, it actually runs the prediction algorithm on it
    single_item_array = (numpy.array(letter)).reshape(1,784)
    prediction = mlp2.predict(single_item_array)
    typed_story = typed_story + str(chr(prediction[0]+96))
    
print("Conversion to typed story complete!")
print("V2")
print(typed_story)

# STEP 4.5

# These steps process the scanned images to be in the same format and have the same properties as the EMNIST images
# They are described by the EMNIST authors in detail here: https://arxiv.org/abs/1702.05373v1
processed_story = []

iterator = 0
for img in handwritten_story:
  #step 1: Apply Gaussian blur filter
  img = cv2.GaussianBlur(img, (7,7), 0)
  cv2.imwrite("post_gaussian_blur/img" + str(iterator) + ".jpg", img)
  iterator += 1
  """
  #steps 2 and 3: Extract the Region of Interest in the image and center in square
  points = cv2.findNonZero(img)
  x, y, w, h = cv2.boundingRect(points)
  if (w > 0 and h > 0):
    if w > h:
      y = y - (w-h)//2
      img = img[y:y+w, x:x+w]
    else:
      x = x - (h-w)//2
      img = img[y:y+h, x:x+h]

  cv2.imwrite("center/img" + str(iterator) + ".jpg", img)

  """
  #step 4: Resize and resample to be 28 x 28 pixels
  img = cv2.resize(img, (28,28), interpolation = cv2.INTER_CUBIC)
  cv2.imwrite("resize_and_resample/img" + str(iterator) + ".jpg", img)
  """
  #step 5: Normalize pixels and reshape before adding to the new story array
  img = img/255
  img = img.reshape((28,28))
  cv2.imwrite("normalize/img" + str(iterator) + ".jpg", img)
  """
  img = skel(img)
  img = cv2.GaussianBlur(img, (7,7), 0)
  img = (thresh, img) = cv2.threshold(img, 224, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
  cv2.imwrite("skel/img" + str(iterator) + ".jpg", img)

  processed_story.append(img)

print("Processed the scanned images.")


# STEP 4.6

iterator = 0
typed_story = ""
for letter in processed_story:
  #this bit of code checks to see if the image is just a blank space by looking at the color of all the pixels summed
  total_pixel_value = 0
  for j in range(28):
    for k in range(28):
      total_pixel_value += letter[j,k]
  if total_pixel_value < 20:
    typed_story = typed_story + " "
  else:         #if it NOT a blank, it actually runs the prediction algorithm on it
    single_item_array = (numpy.array(letter)).reshape(1,784)
    prediction = mlp2.predict(single_item_array)
    typed_story = typed_story + str(chr(prediction[0]+96))
    cv2.imwrite("letter_" + str(iterator) + ".jpg", processed_story[iterator])
  iterator += 1
    
print("Conversion to typed story complete!")
print("V3")
print(typed_story)