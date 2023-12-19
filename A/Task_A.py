#importing libraries
#General stuff
import cv2
import numpy as np
import pandas as pd
import random

#Tensorflow
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
from PIL import Image
from keras.src.optimizers import Adam
from keras.src.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Flatten, AveragePooling2D, Dropout
from tensorflow.keras.metrics import Precision, Recall, BinaryAccuracy
from tensorflow.keras.regularizers import l2
from tensorflow.keras.models import Model
from tensorflow.python.keras import regularizers

#Sklearn
import seaborn as sns
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import roc_curve, auc

#Function for returning random noisy image from a set of images
def add_noise_to_image(images):
    #Gets number of images from the image set
    NUMBER_OF_IMAGES = len(images)

    #Chooses a random image from the image set
    random_image_index = random.randint(0, NUMBER_OF_IMAGES-1)
    image = images[random_image_index]

    #Describing the Gaussian noise function with mean = 0, sd = randomised
    mean = 0
    standard_deviation = random.uniform(0.25,0.5)
    noise = np.random.normal(mean, standard_deviation, image.shape).astype(np.uint8)

    #Creating noisy image
    noisy_image = Image.fromarray(np.clip(image + noise, 0, 255))
    noisy_image = np.array(noisy_image)

    #Return noisy image
    return noisy_image

#Function for increasing the population of all classes in the dataset to new_population.
def increase_class_population(images, labels, new_population):
    #List for storing images in classes 0 and 1
    class0 = []
    class1 = []

    #Constant for storing the initial number of images in the dataset
    NUMBER_OF_IMAGES = images.shape[0]

    #Sorts the images in the initial dataset into either class 0 or 1, based on their labels
    for i in range(NUMBER_OF_IMAGES):
        #E.g., if image i's label = 0, it will be added to the list class0
        if(labels[i] == 0):
            class0.append(images[i])
        else:
            class1.append(images[i])

    #Array for storing the dataset's initial distribution, current_distribution.
    current_distribution = [len(class0), len(class1)]

    #Array for storing the dataset's new distribution after new images have been added to the dataset.
    new_distribution = current_distribution

    #Variables for storing the number of class 0 and class 1 images added to the dataset.
    number_of_class0_images_added = 0
    number_of_class1_images_added = 0

    #Variable storing the total number of images that have to be added to the dataset to ensure that all classes in the new dataset have a population of new_population
    total_number_of_images_to_be_added = (2*new_population) - sum(current_distribution)

    #Array storing the images and labels that will be appended to the initial dataset, to ensure that all classes in the new dataset have a population of new_population
    images_added = np.zeros((total_number_of_images_to_be_added, 28, 28))
    labels_added = np.zeros((total_number_of_images_to_be_added, 1))

    #Keeps adding new noisy class0 images to images_added until the total number of class0 images (from the initial dataset, and the newly added noisy class0 images) is equal to new_population.
    while(new_distribution[0] < new_population):
        #Creates a noisy class0 image with the function add_noise_to_image
        new_image = add_noise_to_image(class0)

        #Stores the noisy image in the array images_added
        images_added[number_of_class0_images_added+number_of_class1_images_added] = new_image

        #Stores the label (=0, because we have just added a noisy class0 image) in the array labels_added
        labels_added[number_of_class0_images_added+number_of_class1_images_added] = 0

        #number_of_class0_images_added stores the number of new noisy class0 images generated using add_noise_to_image. (number_of_class0_images_added + number_of_class1_images_added) is used as a counter to store the position the newly generated image will be stored in the imaged_added array
        number_of_class0_images_added = number_of_class0_images_added + 1

        #Updating the population of class0 images in the dataset. This is compared with new_population at every iteration of the loop to check if more images need to be added (more images need to be added if the calss population is still less than new_population)
        new_distribution[0] = new_distribution[0] + 1

    #Keeps adding new noisy class1 images to images_added until the total number of class1 images (from the initial dataset, and the newly added noisy class1 images) is equal to new_population
    while(new_distribution[1] < new_population):
        #Creates noisy class1 image with the function add_noise_to_image
        new_image = add_noise_to_image(class1)

        #Storing the generated image in images_added, and its corresponding label (class = 1) in labels_added
        images_added[number_of_class0_images_added+number_of_class1_images_added] = new_image
        labels_added[number_of_class0_images_added+number_of_class1_images_added] = 1

        #Updating the counter for counting the number of class1 images that have been generated
        number_of_class1_images_added = number_of_class1_images_added + 1

        #Updating class1's population.
        new_distribution[1] = new_distribution[1] + 1

    #Appends the noisy images and their corresponding labels to the initial dataset and its labels
    return_images = np.concatenate((images, images_added), axis = 0)
    return_labels = np.concatenate((labels, labels_added), axis = 0)

    #Returns the new dataset and its corresponding labels
    return return_images, return_labels

#Function for building cnn. NUMBER_OF_FILTERS stores the number of filters the 1st, 2nd, 3rd, and 4th convolution layers in the CNN has. INPUT_SHAPE stores the shape of the input image.
def build_CNN(NUMBER_OF_FILTERS, INPUT_SHAPE):

    #Constants
    KERNEL_SIZE = 3
    STRIDES = 1


    #NUMBER_OF_FILTERS is an array containing the number of kernels per convolution layer, 5
    cnn = Sequential()

    cnn.add(Conv2D(NUMBER_OF_FILTERS[0], kernel_size = (KERNEL_SIZE,KERNEL_SIZE), strides = STRIDES, padding = 'same', activation = 'relu', input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    for i in range(len(NUMBER_OF_FILTERS)-1):
        cnn.add(Conv2D(NUMBER_OF_FILTERS[i+1], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same', activation='relu'))


        cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Flatten())

    cnn.add(Dense(84, activation = 'relu'))
    cnn.add(Dense(1, activation='sigmoid'))

    cnn.compile(loss='binary_crossentropy', metrics=['accuracy'])

    cnn.summary()

    return cnn

#Function for scaling down images
def scale_down(dataset):
    scaled_dataset = dataset/255
    return scaled_dataset

#Function for rounding up y_pred
def round_up_y_pred(y_pred, threshold):
    y_pred_rounded = np.zeros(len(y_pred))

    for i in range(len(y_pred)):
        if(y_pred[i] > threshold):
            y_pred_rounded[i] = 1
        else:
            y_pred_rounded[i] = 0

    return y_pred_rounded

#Function for getting performance metrics
def find_performance_metric(y_test, y_pred):
    recall = recall_score(y_test, y_pred, average = 'weighted')
    precision = precision_score(y_test, y_pred, average = 'weighted')
    f1 = f1_score(y_test, y_pred, average = 'weighted')
    accuracy = accuracy_score(y_test, y_pred)
    print("Recall score: " + str(recall))
    print("Precision score: " + str(precision))
    print("F1 score: " + str(f1))
    print("Accuracy score: " +str(accuracy))

#Function for printing out confusion matrix
def find_confusion_matrix(y_test, y_pred, TITLE):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot = True, cmap = 'Blues', fmt = 'g')
    plt.xlabel('Predicted Value')
    plt.ylabel('True Value')
    plt.suptitle(TITLE)
    plt.show()

def train_test_cnn(number_of_filters_per_layer, X_train, y_train, X_val, y_val, X_test, y_test):

    INPUT_SHAPE = (28,28,1)
    EPOCHS = 5
    BATCH_SIZE = 32
    weights = {0: 1, 1: 8}

    cnn = build_CNN(number_of_filters_per_layer, INPUT_SHAPE)

    history = cnn.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_val, y_val), class_weight = weights)

    training_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    epochs = range(1, len(training_accuracy) + 1)
    plt.plot(epochs, training_accuracy, label = 'Training')
    plt.plot(epochs, val_accuracy, label = 'Validation')
    plt.title('Training and Validation Accuracy')
    plt.show()

    y_pred = cnn.predict(X_val)

    fpr, tpr, thresholds = roc_curve(y_val, y_pred)

    youden_j = tpr - fpr

    maximised_youden_j = 0
    maximised_threshold = 0
    maximised_index = 0

    for i in range(len(youden_j)):
        if (youden_j[i] > maximised_youden_j):
            maximised_youden_j = youden_j[i]
            maximised_threshold = thresholds[i]

    y_pred = cnn.predict(X_test)

    y_pred = round_up_y_pred(y_pred, maximised_threshold)

    find_confusion_matrix(y_test, y_pred, 'Confusion Matrix')
    find_performance_metric(y_test, y_pred)

def Task_A_Tasks(dataset):
    X_train = dataset['train_images']
    X_val = dataset['val_images']
    X_test = dataset['test_images']

    y_train = dataset['train_labels']
    y_val = dataset['val_labels']
    y_test = dataset['test_labels']

    X_train_balanced, y_train_balanced = increase_class_population(X_train, y_train, 10000)
    X_val_balanced, y_val_balanced = increase_class_population(X_val, y_val, 500)

    X_train_scaled = scale_down(X_train_balanced)
    X_val_scaled = scale_down(X_val)

    pyramid = [40, 35, 30, 25]
    reverse_pyramid = [25, 30, 35, 40]
    constant_filter = [30, 30, 30, 30]

    for number_of_filters_per_layer in [pyramid, reverse_pyramid, constant_filter]:
        train_test_cnn(number_of_filters_per_layer, X_train_scaled, y_train_balanced, X_val_scaled, y_val, X_test, y_test)

