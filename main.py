#General stuff
import cv2
import numpy as np
import pandas as pd
import random

from matplotlib import pyplot as plt

from A.Task_A import Task_A_Tasks

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
    data_dir = 'Datasets/pneumoniamnist.npz'
    dataset = load_data(data_dir)

    Task_A_Tasks(dataset)
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
