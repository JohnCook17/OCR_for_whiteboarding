# Optical Character Recognition For White Boards




This project's goal was to make a program to read handwriting from a whiteboard and translate it into digital text. While it only works when the input image is excelent, I am proud of my work. Here is the [GitHub][git], the [landing page][page], my [LinkedIn], and a [blog] post about the project.



# Installation

  This program is designed to run on linux (Ubuntu 16) and was tested with python 3.6.7, it will not run on Mac or Windows as of this writing. The installation is simple enough, just run the program and it should fetch all the required data and libraries for installation, provided you are properly connected to the internet and have pip3 installed properly. This does take a good while however due to importing the 7000 images or so for the training data, for the machine learning. In the future I will optimize this. If you have a library already installed I can not ensure that it is the correct version and you may have to update, or revert versions. The demo is dependent on files from PBS's [GitHub][pbs] and I can not ensure that they will always be there.


# Usage

You must put the image you wish to run the program on in the "image_to_ocr" directory and label it "img.jpg". Then run "./single_character_ocrv01.py", it will take a moment to load all the libraries. It will then present you with the following option menu:
![menu](https://i.imgur.com/5whBRMr.jpg)
enter the number to the option you wish to run. The demo runs a short demo of the machine learning process. The main program takes the image in "image_to_ocr" that is "img.jpg" and will do its best to extract letters and translate them to text. Note that deleting does not truly delete the file but moves it to the "backups" directory. For best results it is best to supply an image with little to no glare, and low lighting seems to work better then bright lighting. It is best to use capital letters, and put bars on your I's. Currently I have not trained the machine learning algorithm on numbers or punctuation, but plan to eventually. It is also important to be able to draw a vertical line between each letter as this is how the character segmentation works.

# How it works

This program works by taking a given image, turning it black and white, and then blurring the image. We then scan one pixel at a time horizontally until there is no letter detected. This is done by taking horizontal slices and taking their mean pixel values. Since the image is black and white the pixel values can range from 0 - 255. If the pixel value of the horizontal slice is close to 0 then it is determined to be a new line. A similar process is done for each character except for this time the slice is done vertically. We then attempt to center the character, we then append the centered character to a list of characters to process by the machine learning algorithm. We then reshape the image to 28X28 pixels, this method allows us to process characters of varying sizes contained within an image, it is also prone to noise in the image. The image is then changed to a single dimensional array of 784 pixels and feeds it to the machine learning algorithm. Once the machine learning algorithm is fed the array, it makes a prediction and appends this letter to a list to be printed out at the end.

# Contribution

John Cook is the only contributor at this time

# Related Projects

Here is a [repo] that represents my python skills.
# Licensing

Free to use but please give me credit.


   [git]: <https://github.com/JohnCook17/OCR_for_whiteboarding>
   [linkedin]: <https://www.linkedin.com/in/john-cook-17a13b17a/>
   [blog]: <https://www.linkedin.com/pulse/optical-character-recognition-white-boards-john-cook/?published=t>
   [page]: <https://johncook17.github.io/ocr_for_whiteboards.github.io/>
   [pbs]: <https://github.com/crash-course-ai/lab1-neural-networks>
   [repo]: <https://github.com/JohnCook17/holbertonschool-higher_level_programming>


# Version 2.0 coming sometime in the future

Needless to say this program does not work, but it was my first try at anything related to machine learning. With that said I would make some major changes knowing what I know now. I would still go character by character but now I would use a CNN instead of a basic neral network. This is due to the fact that in programming variables can be whatever the programmer wants them to be. This makes word prediction difficult but not impossible. Next I would use some Natural Language Processing to help predict the structure of the code. I would also like to use GANS to train it on characters in mnist that are not common such as the curly bracket. As far as preprocessing goes I would use what I would call an image sticher, to take several images and combine them together to form one composite image, similar to how a smartphone can take a panoramic shot. I would do this because often there is glare, or fuzz or other flaws in the captured image. This would resolve some of those flaws.
