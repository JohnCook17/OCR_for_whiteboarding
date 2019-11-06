#!/usr/bin/python3
print("Importing libraries.")
# Importing libraries
import os
from modules.demo import demo
from modules.main_training_module import training
from modules.image_preprocessing import processing

# Initialize my_input
my_input = "waiting"
# While not exit condition print the menu below
while my_input != "0":
    # Menu to print
    print("Enter a coresponding number for that option")
    print("1. Demo")
    print("2. Run Main Program")
    print("3. Delete Demo training data")
    print("4. Delete Main traning data")
    print("5. Help")
    print("0. exit")
    # Get user input
    my_input = input()
    # compair user input to options provided
    # option 1 runs the demo
    if my_input == "1":
        demo()
    # option 2 runs the main program
    elif my_input == "2":
        mlp2 = training()
        processing(mlp2)
    # option 3 deletes the demo training data
    elif my_input == "3":
        # Prints a conformation message
        print("Are you sure? type yes to confirm")
        # If yes moves the traning data to backups does not delete it.
        if input() == "yes":
            print("deleting")
            os.replace("mlp1.joblib", "backups/mlp1.joblib")
        # If not yes exits to main menu
        else:
            print("changed your mind?")
    # option 4 deletes the main traning data
    elif my_input == "4":
        # Prints a conformation message
        print("Are you sure? type yes to confirm")
        # If yes moves the traning data to backups does not delete it.
        if input() == "yes":
            print("deleting")
            os.replace("mlp2.joblib", "backups/mlp2.joblib")
        # If not yes exits to main menu
        else:
            print("changed your mind?")
    # option 5 offers help
    elif my_input == "5":
        print("please put image in the folder 'image_to_ocr' and name it img.jpg to perform ocr on it.")
