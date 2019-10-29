#!/usr/bin/python3
print("Importing libraries.")
import os
from modules.demo import demo
from modules.main_training_module import training
from modules.image_preprocessing import processing

my_input = "waiting"
while my_input != "0":
    print("Enter a coresponding number for that option")
    print("1. Demo")
    print("2. Run Main Program")
    print("3. Delete Demo training data")
    print("4. Delete Main traning data")
    print("5. Help")
    print("0. exit")
    my_input = input()
    if my_input == "1":
        demo()
    elif my_input == "2":
        mlp2 = training()
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
        print("please put image in the folder 'image_to_ocr' and name it img.jpg to perform ocr on it.")
