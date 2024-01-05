#General stuff
import cv2
import numpy as np
import pandas as pd
import random

from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from sklearn.svm import SVC

from A.Task_A import Task_A_Tasks
from B.Task_B import Task_B_Tasks

#Function for loading the dataset from the directory dataset_dir
def load_data(dataset_dir):
    #Saves laoded data as the variable dataset
    dataset = np.load(dataset_dir)

    #Prints out the headers in the dataset
    dataset_pd = pd.DataFrame(dataset)

    #Prints out number of images in each set
    n_train = dataset['train_images'].shape[0]
    n_val = dataset['val_images'].shape[0]
    n_test = dataset['test_images'].shape[0]
    print("Number of train images: ", n_train)
    print("Number of validation images: ", n_val)
    print("Number of test images: ", n_test)

    #Returns the laoded dataset
    return dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Program asks the user if they want to run Task A, B, or exit the program.
    user_choice = input("Enter 'A' to run Task A, 'B' to run Task B, and 'X' to exit the program.")

    #User input must be A, B, or X, otherwise the program will keep prompting the user to re-enter their choice.
    while(user_choice != 'X'):
        while(user_choice != 'A' and user_choice != 'B' and user_choice != 'X'):
            user_choice = input("You must enter either 'A', 'B', or 'X'.")

        #If the user entered A, the program will run Task A
        if(user_choice == 'A'):
            #Loading dataset from PneumoniaMNIST
            data_dir_A = 'Datasets/pneumoniamnist.npz'
            print("Details for PneumoniaMNIST are printed below:")
            dataset_A = load_data(data_dir_A)

            #Passing the dataset into the function for running TaskA-related tasks
            Task_A_Tasks(dataset_A)

            #Informing the user that the task has finished.
            print("Finished running Task A.")

        #If the user entered B, the program will run Task B
        elif(user_choice == 'B'):
            #Loading dataset from PathMNIST
            data_dir_B = 'Datasets/pathmnist.npz'
            dataset_B = load_data(data_dir_B)

            #Passing the dataset into the function for running TaskB-related tasks
            Task_B_Tasks(dataset_B)

            #Informing the user that the task has finished.
            print("Finished running Task B.")

        #If the user entered X, the program will terminate itself.
        if(user_choice == 'X'):
            print("Exiting program...")

        #If the user did not enter 'X' -- but entered 'A' or 'B' previously -- the program will loop itself, and ask for the user's input again, until the user enters 'X' to terminate the program.
        else:
            user_choice = input("Enter 'A' to run Task A, 'B' to run Task B, and 'X' to exit the program.")

            if(user_choice == 'X'):
                print('Exiting program...')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
