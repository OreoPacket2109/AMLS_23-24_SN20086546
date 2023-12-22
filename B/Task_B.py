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

#Importing functions from Task A
from A.Task_A import scale_down
from A.Task_A import find_confusion_matrix

#Font sizes for plotting to ensure figures have consistent font sizes
VERY_BIG_FONT = 18
BIG_FONT = 16
MEDIUM_FONT = 14
SMALL_FONT = 12

#Function for finding class distribution
def find_class_distribution(labels):
    #Array for storing each class's population. Initiated to be 0s.
    class_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 0]

    #Constant for storing the number of images in the inputted dataset
    NUMBER_OF_IMAGES = len(labels)

    #Converts labels into a 1D array
    labels = np.array(labels)
    labels = labels.flatten()

    #Adjusting class_distribution based on the value of labels[i]. E.g., if labels[i] = 3, class_distribution[3] (which represents the number of images recorded in class3) will be incremented by 1
    for i in range(NUMBER_OF_IMAGES):
        #Increments the element in class_distribution which corresponds to the value stored in labels[i] (i.e., the image's class) by 1
        temp_label = labels[i]
        class_distribution[temp_label] = class_distribution[temp_label] + 1

    #Returns the number of images in each class
    return class_distribution

#Function for adding noise to a random RGB image from a group of images, images
def add_noise_to_image(images):
    #Constant for storing the number of images
    NUMBER_OF_IMAGES = len(images)

    #Randomly generated integer used to randomly select an RGB image from images
    random_image_index = random.randint(0, NUMBER_OF_IMAGES - 1)

    #Variale used to store the randomly chosen image
    image = images[random_image_index]

    #Setting the mean and standard deviations for the gaussian noise that will be superimposed onto the image
    mean = 0

    #Standard deviation is a random number between 0.25 and 2. This allows the model to adapt to images with varying levels of noises.
    standard_deviation = random.uniform(0.25, 2)

    #Creates the noise
    noise = np.random.normal(mean, standard_deviation, image.shape).astype('uint8')

    #Superimposing the noise onto the image
    noisy_image = cv2.add(image, noise)

    #Returning the noisy image
    return noisy_image

#Function for increasing the population of all classes in the dataset to new_population
def increase_class_population(images, labels, new_population):
    #List for storing images in classes 0 and 1
    class0 = []
    class1 = []
    class2 = []
    class3 = []
    class4 = []
    class5 = []
    class6 = []
    class7 = []
    class8 = []

    #Constant for storing the number of images in the dataset
    NUMBER_OF_IMAGES = images.shape[0]

    #Constant for storing the number of classes in the dataset
    NUMBER_OF_CLASSES = 9

    #labels is initially inputted as a list. The code below converts it into an array, so that it is easier to work with.
    labels = np.array(labels)
    labels = labels.flatten()

    #Dictionary for referencing the lists for storing the images (e.g., class0, class1, etc.) to their respective class numbers (e.g., 0, 1, respectively).
    class_dict = {0: class0, 1: class1, 2: class2, 3: class3, 4: class4, 5: class5, 6: class6, 7: class7, 8: class8}

    #Appends an image to its respective class list. E.g., an image whose label = 3 will be added to class3
    for i in range(NUMBER_OF_IMAGES):
        temp_label = labels[i]

        #Appends the image to the list corresponding to image[i]'s class. E.g., if labels[i] = 1, image[i] belongs to class 1, and will be appended to the list class1
        class_dict[temp_label].append(images[i])


    #Finds the dataset's current distribution by using the function find_class_distribution()
    current_distribution = find_class_distribution(labels)

    #The new distribution is initially set to be equal to the current_distribution
    new_distribution = current_distribution

    #Lists for storing the new images and labels that will be appended to the existing ones, such that the returned dataset is balanced (i.e., all classes have the same population: new_population)
    new_images = []
    new_labels = []

    #Checks if the current class has a population of new_population. If not, a noisy image (generated based on a random image from that class) will be added to new_images
    for i in range(NUMBER_OF_CLASSES):

        #The current class that is being investigated is class i. E.g., if i = 0, current_class = class_dict[0] = class0
        current_class = class_dict[i]

        #Keeps adding new images to the list containing images for each class (e.g., class0, class1, ...) until the class has population = new_population
        while(new_distribution[i] < new_population):
            #Generating a noisy image from class current_class
            new_image = add_noise_to_image(current_class)

            #Appending the image to new_images
            new_images.append(new_image)

            #Also appends the label corresponding to new_image (e.g., new_label = 1 if new_image is from class1) to new_labels
            new_labels.append(i)

            #Increments the element in new_distribution which corresponds to the current_class's population by 1
            new_distribution[i] = new_distribution[i] + 1

    #Converts new_images back to array
    new_images = np.array(new_images)
    return_images = np.concatenate((images, new_images), axis = 0)

    #Converts new_labels back to array
    new_labels = np.array(new_labels)
    new_labels = new_labels.flatten()
    return_labels = np.concatenate((labels, new_labels), axis = 0)

    #Returns the new images and corresponding labels that are balanced
    return return_images, return_labels

#Function for pulling a specific number of images from the main pool of images
def sample_from_main_pool(X_main, y_main, sample_size):
    return_images = np.zeros((sample_size, 28, 28, 3))
    return_labels = np.zeros((sample_size, 1))

    for i in range(sample_size):
        return_images[i] = X_main[i]
        return_labels[i] = y_main[i]

    return_labels = np.array(return_labels)
    return_labels = return_labels.flatten()
    return_labels = return_labels.astype(int)

    print(return_labels)

    return return_images, return_labels

#Function for building the CNN with NUMBER_OF_FILTERS filters in each layer. This CNN can take RGB images as an input.
def build_CNN(NUMBER_OF_FILTERS, INPUT_SHAPE):

    #Constants
    KERNEL_SIZE = 3
    STRIDES = 1

    cnn = Sequential()

    #Adding the first convolutional (Conv) layer
    cnn.add(Conv2D(NUMBER_OF_FILTERS[0], kernel_size = (KERNEL_SIZE,KERNEL_SIZE), strides = STRIDES, padding = 'same', activation = 'relu', input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])))
    cnn.add(Conv2D(NUMBER_OF_FILTERS[1], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same',
                   activation='relu'))
    cnn.add(Conv2D(NUMBER_OF_FILTERS[2], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same',
                   activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(NUMBER_OF_FILTERS[3], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same',
                   activation='relu'))
    cnn.add(Conv2D(NUMBER_OF_FILTERS[4], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same',
                   activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    cnn.add(Conv2D(NUMBER_OF_FILTERS[5], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same',
                   activation='relu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Adding the first MaxPooling layer
    #cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Adding new convolutional layer. Number of filters in the i^th convolutional layer depends on the i^th element of the array NUMBER_OF_FILTERS
    #for i in range(len(NUMBER_OF_FILTERS)-1):
    #    cnn.add(Conv2D(NUMBER_OF_FILTERS[i+1], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same', activation='relu'))
    #    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Adding the flatten layer
    cnn.add(Flatten())

    #Adding the fully-connected (FC) layer
    cnn.add(Dense(256, activation = 'relu'))
    cnn.add(Dense(128, activation='relu'))
    cnn.add(Dense(64, activation='relu'))
    #9 nodes, each representing the probability of the image being in one of the 9 classes. The CNN outputs a 9x1 array, wherein the i^th element represents the probability of the image belonging to the i^th class
    cnn.add(Dense(9, activation = 'softmax'))

    #Using the Adam optimiser, with learning rate = 0.001 (i.e., the default value)
    my_optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001)

    #Compiling the model
    cnn.compile(optimizer = my_optimizer, loss = 'sparse_categorical_crossentropy', metrics=['accuracy'])

    cnn.summary()

    #Returning the model
    return cnn

#Function for rounding up y_pred. y_pred[i] is a 9x1 array, wherein each element containts the probability of the image belonging to a class. This function finds the largest probability in each of these arrays, and outputs the corresponding class number.
#E.g., if y_pred[i] = [0.4 0 0 0 0 0 0 0 0.6], 0.6 is the largest probability, so y_pred_rounded[i] = 8
def round_up_y_pred(y_pred):
    #Constants for storing the numbers of predictions and classes
    NUMBER_OF_PREDICTIONS = y_pred.shape[0]
    NUMBER_OF_CLASSES = y_pred.shape[1]

    #Array for storing the rounded up predictions
    y_pred_rounded = np.zeros(len(y_pred))

    #Loops through all the predictions' 9x1 arrays, and outputs the class with the highest probability
    for i in range(NUMBER_OF_PREDICTIONS):
        #Highest probability stores the highest probability in any 9x1 array for each y_pred[i]. Initiated to be 0.
        highest_probability = 0

        #Most likely class corresponds to the class that has the highest probability in the 9x1 array for each y_pred[i]. Also initiated to be 0.
        most_likely_class = 0

        #Finds the class with the highest probability for each prediction, and sets it as the most likely class.
        for j in range(NUMBER_OF_CLASSES):
            #If the i^th image's probability of belonging to the jth class is higher than the highest probability recorded in the i^th image's 9x1 array so far, the j^th class's probability is set to be the highest_probability, and the most_likely_class is also set to j
            if(y_pred[i, j] > highest_probability):
                highest_probability = y_pred[i,j]
                most_likely_class = j

        #After checking all of the i^th image's probabilities of belonging in each class, the class with the highest probability is set to be equal to y_pred_rounded[i]
        y_pred_rounded[i] = most_likely_class

    #Returns the prediction, wherein the predicted label for each image is a scalar, instead of a 9x1 vector
    return y_pred_rounded

#Function for getting performance metrics
def find_performance_metric(y_test, y_pred):
    recall = recall_score(y_test, y_pred, average='micro')
    precision = precision_score(y_test, y_pred, average='micro')
    f1 = f1_score(y_test, y_pred, average='micro')
    accuracy = accuracy_score(y_test, y_pred)
    print("Recall score: " + str(recall))
    print("Precision score: " + str(precision))
    print("F1 score: " + str(f1))
    print("Accuracy score: " +str(accuracy))

#Function for training cnn
def train_test_cnn(number_of_filters_per_layer, X_train, y_train, X_val, y_val, X_test, y_test):

    #==============================|3.1. Training the Model|==============================
    #Constants
    INPUT_SHAPE = (28, 28, 3)
    EPOCHS = 5
    BATCH_SIZE = 64

    #Building the CNN with the number of filters in each layer specified by number_of_filters_per_layer
    cnn = build_CNN(number_of_filters_per_layer, INPUT_SHAPE)

    #Training the cnn. Training history is stored in history.
    history = cnn.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_val, y_val))

    #Storing the training and validation losses at each epoch for plotting later.
    training_loss = history.history['loss']
    val_loss = history.history['val_loss']

    #Creating an array to store the number of epochs for plotting later.
    epochs = range(1, len(training_loss) + 1)

    #Plotting the training and validation losses vs. epoch curve
    plt.figure(figsize=(8, 6))
    plt.plot(epochs, training_loss, label = 'Training')
    plt.plot(epochs, val_loss, label = 'Validation')
    plt.title('Training and Validation Loss', fontsize = BIG_FONT)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize = MEDIUM_FONT)
    plt.xlabel('Epochs', fontsize = MEDIUM_FONT)
    plt.ylabel('Loss', fontsize = MEDIUM_FONT)
    plt.xticks(fontsize=SMALL_FONT)
    plt.yticks(fontsize=SMALL_FONT)
    plt.grid()
    plt.show()

    #Finding y_pred based on the trained model.
    y_pred = cnn.predict(X_test)

    #Reformat the output of y_pred from a 2D matrix (wherein each prediction is a 9x1 vector) to a 1D array (wherein each prediction is a scalar number)
    y_pred = round_up_y_pred(y_pred)

    #Finding the confusion matrix and performance metrics (i.e., accuracy, f1 score, etc.)
    find_confusion_matrix(y_test, y_pred, "Confusion Matrix")
    find_performance_metric(y_test, y_pred)

#Function for finding the model loss when different training set sizes are used
def find_loss_vs_sample_size(X_train, y_train, X_val, y_val, INPUT_SHAPE, NUMBER_OF_FILTERS_PER_LAYER):
    #Constants
    EPOCHS = 3
    BATCH_SIZE = 64

    #Building and training the model
    cnn = build_CNN(NUMBER_OF_FILTERS_PER_LAYER, INPUT_SHAPE)
    history = cnn.fit(X_train, y_train, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_val, y_val))

    #Finding the training and validation loss after the model has been fully trained
    train_loss = cnn.evaluate(X_train, y_train)
    val_loss = cnn.evaluate(X_val, y_val)

    #Returns the training and validation loss
    return train_loss[0], val_loss[0]

#Function for running tasks associated with Task B
def Task_B_Tasks(dataset):
    #============================================|1. Analysing Given Dataset|============================================
    #Spitting the provided dataset into train, val, and test sets
    X_train = dataset['train_images']
    X_val = dataset['val_images']
    X_test = dataset['test_images']

    y_train = dataset['train_labels']
    y_val = dataset['val_labels']
    y_test = dataset['test_labels']

    #Finding the class distribution for each set
    train_distribution = find_class_distribution(y_train)
    val_distribution = find_class_distribution(y_val)
    test_distribution = find_class_distribution(y_test)

    #Plotting the class distribution for each set
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (15,15))

    axes[0].pie(train_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT})
    axes[0].set_title("Training Set", fontsize = VERY_BIG_FONT+2)
    axes[1].pie(val_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT})
    axes[1].set_title("Validation Set", fontsize = VERY_BIG_FONT+2)
    axes[2].pie(test_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT})
    axes[2].set_title("Test Set", fontsize = VERY_BIG_FONT+2)
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8'], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize = VERY_BIG_FONT)
    plt.tight_layout()
    plt.show()

    #============================================|2. Data Pre-Processing|============================================
    #Creating the main pool of images for the model to pull from during the training stage, since the model will be trained at different training set sizes to find the optimal training set size. The pool contains 12, 000 images from each class.
    X_train_balanced_pool, y_train_balanced_pool = increase_class_population(X_train, y_train, 25000)

    #Scales down image pixel values from 0-255 to 0-1
    X_train_balanced_pool = scale_down(X_train_balanced_pool)

    #Taking 10 000 images from the main pool to train on
    X_train_balanced, y_train_balanced = increase_class_population(X_train, y_train, 22000)#sample_from_main_pool(X_train_balanced_pool, y_train_balanced_pool, 22000)
    X_train_balanced = scale_down(X_train_balanced)
    balanced_train_distribution = find_class_distribution(y_train_balanced)

    #Plotting pie charts to compare the training set's distribution before and after it is balanced
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (10,10))
    axes[0].pie(train_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT})
    axes[0].set_title("Unbalanced Training Set", fontsize=VERY_BIG_FONT)
    axes[1].pie(balanced_train_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT})
    axes[1].set_title("Balanced Training Set", fontsize=VERY_BIG_FONT)
    plt.legend(['Class 0', 'Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6', 'Class 7', 'Class 8'], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=VERY_BIG_FONT)
    plt.tight_layout()
    plt.show()

    #Scaling down the validation and test images from 0-255 to 0-1
    X_val_scaled = scale_down(X_val)
    X_test_scaled = scale_down(X_test)

    #============================================|3. Model Training, Hyperparamter Tuning, and Testing|============================================
    #Arrays containing the number of filters per convolution layer for the pyramid and reverse-pyramid structures
    reverse_pyramid = [32, 40, 48, 50, 56, 64]

    #Building the cnn, training it, validating it, and testing it for both the pyramid and reverse-pyramid structures
    #train_test_cnn(pyramid, X_train_scaled_balanced, y_train_balanced, X_val_scaled, y_val, X_test_scaled, y_test)
    train_test_cnn(reverse_pyramid, X_train_balanced, y_train_balanced, X_val_scaled, y_val, X_test_scaled, y_test)

    #============================================|4. Learning Curve|============================================
    #Array containing the different training set sizes the model is trained on
    training_set_sizes = [900, 18000, 63000, 90000, 180000]

    #Arrays for storing the training and validation losses at different training set sizes
    train_loss = np.zeros((len(training_set_sizes), 1))
    val_loss = np.zeros((len(training_set_sizes), 1))

    #Finding the training and validation losses at different training set sizes
    for i in range(len(training_set_sizes)):

        #Sets the current training set size to training_set_size[i]
        current_training_set_size = training_set_sizes[i]

        #Takes training_set_size[i] images from the main dataset to use to train/validate the model
        current_X, current_y = sample_from_main_pool(X_train_balanced_pool, y_train_balanced_pool, current_training_set_size)

        #Storing the training and validation loss when current_training_set_size training points are used
        train_loss[i], val_loss[i] = find_loss_vs_sample_size(current_X, current_y, X_val_scaled, y_val, (28, 28, 3), reverse_pyramid)

    #Plotting the training and validation losses against the training set size
    plt.figure(figsize=(8, 6))
    plt.plot(training_set_sizes, train_loss, label = 'Training Loss')
    plt.plot(training_set_sizes, val_loss, label = 'Validation Loss')
    plt.xlabel('Training Set Size', fontsize = MEDIUM_FONT)
    plt.ylabel('Loss', fontsize = MEDIUM_FONT)
    plt.xticks(fontsize=SMALL_FONT)
    plt.yticks(fontsize=SMALL_FONT)
    plt.title('Learning Curve', fontsize = BIG_FONT)
    plt.legend(['Training Loss', 'Validation Loss'], fontsize = MEDIUM_FONT)
    plt.grid()
    plt.show()