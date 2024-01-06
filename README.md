# AMLS_23-24_SN20086546
## Project Description
This project contains the code for the ELEC0134 assignment. The code develops CNNs for a binary classification and a multiclass classification tasks. Two types of CNNs -- one following the "pyramid" structure, and the other following the "reverse-pyramid" structure -- will be developed for each task.

The binary classification task involves classifying grayscale images of anterior posterior chest X-ray scans from the PneumoniaMNIST dataset into two classes: normal and pneumonia.

The multiclass classification task involves classifying RGB images of colorectal tissues from the PathMNIST dataset into nine classes: Adipose (ADI), background (BACK), debris (DEB), lymphocytes (LYM), mucus (MUC), smooth muscle (MUS), normal colon mucosa (NORM), cancer-associated stroma (STR), colorectal adenocarcinoma epithelium (TUM).

This code will also print out each develped CNN's confusion matrix, accuracy score, and learning curves (losses vs. epoch, and losses vs. training set size).

## Project Organisation
The project folder is arranged as follows:
- AMLS_23-24_SN20086546
  - A
    - Task_A.py
  - B
    - Task_B.py
  - Datasets
    - pathmnist.npz
    - pneumoniamnist.npz
  - .gitignore
  - main.py
  - README.md

All code related to the assignment must be run from main.py.

Task_A.py and Task_B.py contain code required for building, training, testing the CNNs, and generating graphs related to Tasks A and B, respectively.

pathmnist.npz and pneumoniamnist.npz are not included in this project repository. The user is expected to download the files themselves, and paste them into the _Datasets_ folder in their own copy of this code on their own machine. Instructions on how to download the files can be found in the *Setting Up the Project* Section.

## Setting Up the Project
1. Download the project onto the user's local machine.
2. Open the project. The project folder should look like the one shown in the *Project Organisation* section, except the *Dataset* folder should be empty.
3. Download the datasets. Click [here](https://zenodo.org/records/6496656) to open the website which contains all the MedMNIST datasets. Download the pathmnist.npz and pneumonia.npz files from the _Files_ section.
4. Paste the datasets from Step 3 into the _Datasets_ folder, so the user's project folder should look like the one shown in the _Project Organisation_ section.
5. Download the required libraries and packages. Refer to the _Required Packages_ section for further details.

## Required Packages


## Running the Project
User will be directed to a menu screen where they will be prompted to choose if they want to run Task A or Task B. Must answer A, B, or X.

User will also be asked if they want to see the learning curvse. Learning curves take lots of computational power to generate, and some users' machines may not be able to handle it, so they are given the option to skip this stage.

Program will prompt the user if they want to see misclassified images for Task A, but not Task B. This is because there are more images in Task B than A, making it messy to print them all out.

## Required Packages
| Package Name | Version |
| -------- | -------- |
| tensorflow | 2.15.0 |

