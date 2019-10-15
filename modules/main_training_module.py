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


def training(X_train, y_train, X_test, y_test):
    if not os.path.exists("mlp2.joblib"):

        # Change some of the values in the below statement and re-run to see how
        # they affect performance!
        mlp2 = MLPClassifier(hidden_layer_sizes=(250,
                                                250,
                                                250,
                                                250,
                                                250,
                                                250,),
                            max_iter=1000, alpha=1e-4,
                            solver="sgd", verbose=10, tol=1e-4, random_state=1,
                            learning_rate_init=.1)
        mlp2.fit(X_train, y_train)
        print("Training set score: %f" % mlp2.score(X_train, y_train))
        print("Test set score: %f" % mlp2.score(X_test, y_test))
        
        dump(mlp2, "mlp2.joblib")
    else:
        mlp2 = load("mlp2.joblib")
        y_pred = mlp2.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        cv2.imwrite("confusion matrix2.jpg", cm)
    return mlp2