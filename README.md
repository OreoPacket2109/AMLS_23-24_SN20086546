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
  - requirement.txt

All code related to the assignment must be run from main.py.

Task_A.py and Task_B.py contain code required for building, training, testing the CNNs, and generating graphs related to Tasks A and B, respectively.

pathmnist.npz and pneumoniamnist.npz are not included in this project repository. The user is expected to download the files themselves, and paste them into the _Datasets_ folder in their own copy of this code on their own machine. Instructions on how to download the files can be found in the *Setting Up the Project* Section.

requirement.txt contains the list of libraries required to run the code.

## Setting Up the Project
1. Download the project onto the user's local machine.
2. Open the project. The project folder should look like the one shown in the *Project Organisation* section, except the *Dataset* folder should be empty.
3. Download the datasets. Click [here](https://zenodo.org/records/6496656) to open the website which contains all the MedMNIST datasets. Download the pathmnist.npz and pneumonia.npz files from the _Files_ section.
4. Paste the datasets from Step 3 into the _Datasets_ folder, so the user's project folder should look like the one shown in the _Project Organisation_ section.
5. Download the required libraries and packages. Refer to the _Required Packages_ section for further details.

## Required Packages
| Package Name | Version |
| -------- | -------- |
| matplotlib | 3.7.2 |
| numpy | 1.24.3 |
| random | 1.2.4 |
| scikit-learn | 1.3.0 |
| tensorflow | 2.15.0 |

For more information, refer to _requirements.txt_ in the repository.

## Running the Project
1. Run the project from main.py
2. The program will ask the user if they want to run Task A, B, or exit the program:
    - The user must enter either A, B, or X. The program will keep asking the user until they enter one of the three input options.
    - If the user entered A, the program will run Task_A.py
    - If the user entered B, the program will run Task_B.py
    - If the user entered X, the program will terminate itself.

3. The program will then ask the user if they want to view the learning curves.
    - The user must enter either Y or N. The program will keep asking the user until they enter one of the two input options.
    - This option is included because generating the learning curves takes time and computation resources that some users may not have access to. So they can choose not to view them.

4. After the model has finished training, the program will ask the user if they want to view the misclassified images.
    - This option is included because some users may not be interested in the misclassified images, so they may skip this step.
    - This function is only included for Task A because there are significantly more misclassified images (in the hundreds range) for Task B, making it a messy affair to print them all out.
    - The user is free to improve this code by adding functions for printing out misclassified images for Task B (refer to _Contributing Guidelines_ on how to contact the project owner regarding additional contributions to the code).

5. The code returns to step 2. The code will keep looping back to the menu function from step 2 after each Task is completed, until the user enters X when the option to enter A, B, or X is given. In which case, the program terminates.

## Contributing Guidelines
Contact zceegdu@ucl.ac.uk for permission to edit code.
