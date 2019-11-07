# Optical Character Recognition For White Boards




This project's goal was to make a program to read handwriting from a whiteboard and translate it into digital text. While it only works when the input image is excelent, I am proud of my work. Here is the [GitHub][git], the [landing page][page], my [LinkedIn], and a [blog] post about the project.



# Installation

  This program is designed to run on linux (Ubuntu 16) and was tested with python 3.6.7, it will not run on Mac or Windows as of this writing. The installation is simple enough, just run the program and it should fetch all the required data and libraries for installation. However if you have a library already installed I can not ensure that it is the correct version and you may have to update, or revert versions. The demo is dependent on files from PBS's [GitHub][pbs] and I can not ensure that they will always be there.


# Usage

You must put the image you wish to run the program on in the "image_to_ocr" directory and label it "img.jpg". Then run "./single_character_ocrv01.py", it will take a moment to load all the libraries. It will then present you with the following option menu:
[![menu](https://i.imgur.com/5whBRMr.jpg)]
enter the number to the option you wish to run. The demo runs a short demo of the machine learning process. The main program takes the image in "image_to_ocr" that is "img.jpg" and will do its best to extract letters and translate them to text. Note that deleting does not truly delete the file but moves it to the "backups" directory. For best results it is best to supply an image with little to no glare, and low lighting seems to work better then bright lighting.


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
