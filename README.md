# Applied Machine Learning Systems (2023-24) Code
## Project Description
This project contains the code for the ELEC0134 assignment. The code develops CNNs for a binary classification and a multiclass classification tasks. Two types of CNNs -- one following the "pyramid" structure, and the other following the "reverse-pyramid" structure -- will be developed for each task.

The binary classification task involves classifying grayscale images of anterior posterior chest X-ray scans from the PneumoniaMNIST dataset into two classes: normal and pneumonia.

The multiclass classification task involves classifying RGB images of colorectal tissues from the PathMNIST dataset into nine classes: Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM).

Both datasets can be found [here](https://medmnist.com/).

This code will also print out each develped CNN's confusion matrix, accuracy score, and learning curves (losses vs. epoch, and losses vs. training set size).

## Project Organisation
The project folder is arranged as follows:
- a
  - a
  - 

Folder arrangement.

All files can be run from main.py

Task A tasks are run in Task_A.py, Task B tasks are run from Task_B.py.

## Running the Project
User will be directed to a menu screen where they will be prompted to choose if they want to run Task A or Task B. Must answer A, B, or X.

User will also be asked if they want to see the learning curvse. Learning curves take lots of computational power to generate, and some users' machines may not be able to handle it, so they are given the option to skip this stage.

Program will prompt the user if they want to see misclassified images for Task A, but not Task B. This is because there are more images in Task B than A, making it messy to print them all out.

## Required Packages
