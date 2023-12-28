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

def load_data(dataset_dir):
    #Saves laoded data as the variable dataset
    dataset = np.load(dataset_dir)

    #Prints out the headers in the dataset
    dataset_pd = pd.DataFrame(dataset)
    print(dataset_pd.head())

    #Prints out number of images in each set
    n_train = dataset['train_images'].shape[0]
    n_val = dataset['val_images'].shape[0]
    n_test = dataset['test_images'].shape[0]
    print("Number of train images: ", n_train)
    print("Number of validation images: ", n_val)
    print("Number of test images: ", n_test)

    return dataset

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    #Loading dataset from PneumoniaMNIST
    data_dir_A = 'Datasets/pneumoniamnist.npz'
    dataset_A = load_data(data_dir_A)

    #Passing the dataset into the function for running TaskA-related tasks
    Task_A_Tasks(dataset_A)

    #Loading dataset from PathMNIST
    #data_dir_B = 'Datasets/pathmnist.npz'
    #dataset_B = load_data(data_dir_B)

    #Passing the dataset into the function for running TaskB-related tasks
    #Task_B_Tasks(dataset_B)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
