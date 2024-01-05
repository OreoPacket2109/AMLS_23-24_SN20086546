#============================|Importing libraries|============================
#General stuff
import cv2
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt

#Tensorflow
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

#Font sizes for plotting to ensure figures have consistent font sizes
VERY_BIG_FONT = 18
BIG_FONT = 16
MEDIUM_FONT = 14
SMALL_FONT = 12

#Colors for pie charts to ensure figures have consistent colours
my_colors = ['cornflowerblue', 'darkorange']

#List for storing the index of misclassified images for printing out after testing
class0_misclassified_index = []
class1_misclassified_index = []
misclassified_dict = {0: class0_misclassified_index, 1: class1_misclassified_index}

#Function for returning random noisy image from a set of images
def addNoiseToImage(images):
    #Gets number of images from the image set
    NUMBER_OF_IMAGES = len(images)

    #Chooses a random image from the image set
    random_image_index = random.randint(0, NUMBER_OF_IMAGES-1)
    image = images[random_image_index]

    #Describing the Gaussian noise function with mean = 0, sd = randomised
    mean = 0
    standard_deviation = random.uniform(0.25,1)
    noise = np.random.normal(mean, standard_deviation, image.shape).astype(np.uint8)

    #Creating noisy image
    noisy_image = Image.fromarray(np.clip(image + noise, 0, 255))
    noisy_image = np.array(noisy_image)

    #Return noisy image
    return noisy_image

#Function for increasing the population of all classes in the dataset to new_population.
def increaseClassPopulation(images, labels, new_population):
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
        new_image = addNoiseToImage(class0)

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
        new_image = addNoiseToImage(class1)

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

#Function for building cnn. NUMBER_OF_FILTERS stores the number of filters in each convolution layer in the CNN. INPUT_SHAPE stores the shape of the input image.
def buildCNN(NUMBER_OF_FILTERS, INPUT_SHAPE):
    #Constants
    KERNEL_SIZE = 3
    STRIDES = 1

    cnn = Sequential()

    #Adding the first two convolution layers
    cnn.add(Conv2D(NUMBER_OF_FILTERS[0], kernel_size = (KERNEL_SIZE,KERNEL_SIZE), strides = STRIDES, padding = 'same', activation = 'relu', input_shape = (INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2])))
    cnn.add(Conv2D(NUMBER_OF_FILTERS[1], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same', activation='relu'))

    #Adding the first max pooling layer
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Adding the next convolution layer
    cnn.add(Conv2D(NUMBER_OF_FILTERS[2], kernel_size=(KERNEL_SIZE, KERNEL_SIZE), strides=STRIDES, padding='same', activation='relu'))

    #Adding the second max pooling layer
    cnn.add(MaxPooling2D(pool_size=(2, 2)))

    #Convert output of the convolutional layer into a 1-dimensional array
    cnn.add(Flatten())

    #Fully connected layer, with ReLU activation function
    cnn.add(Dense(128, activation = 'relu'))
    cnn.add(Dense(64, activation = 'relu'))

    #Fully connected layer, with sigmoid activation function.
    cnn.add(Dense(1, activation='sigmoid'))

    #Compiling model
    my_optimizer = Adam(learning_rate=0.001)
    cnn.compile(optimizer = my_optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    cnn.summary()

    #Returns model
    return cnn

#Function for scaling down pixel values in the images to 0-1, instead of 0-255.
def scaleDown(dataset):
    #pixel intensity varies from 0 to 255. This scales down the intensity so that it varies from 0 to 1
    scaled_dataset = dataset/255

    #Returns the scaled down dataset.
    return scaled_dataset

#Function for rounding up y_pred
def roundUpYPred(y_pred, threshold):
    #Array for storing the rounded up/down predictions. The CNN model outputs the probability of the image being class 1, so we have to round it up/down to either 1 or 0.
    y_pred_rounded = np.zeros(len(y_pred))

    #Compares the cnn's outputted prediction to the threshold.
    for i in range(len(y_pred)):

        #If y_pred[i] is larger than the threshold, the image should be in class 1. So, y_pred_rounded[i] = 1
        if(y_pred[i] > threshold):
            y_pred_rounded[i] = 1

        #If y_pred[i] is less than the threshold, the image should be in class 0. So, y_pred_rounded[i] = 0
        else:
            y_pred_rounded[i] = 0

    #Returns the rounded y_pred values
    return y_pred_rounded

#Function for getting performance metrics
def findPerformanceMetric(y_test, y_pred):
    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy score: " +str(accuracy))

#Function for printing out confusion matrix
def findConfusionMatrix(y_test, y_pred, TITLE):
    #Confusion matrix created based on the images' true label (y_test), and its predicted label (y_pred)
    cm = confusion_matrix(y_test, y_pred)

    #Plotting the confusion matrix
    sns.heatmap(cm, annot = True, cmap = 'Blues', fmt = 'g', annot_kws={"fontsize": SMALL_FONT})
    plt.xlabel('Predicted Value', fontsize = MEDIUM_FONT)
    plt.ylabel('True Value', fontsize = MEDIUM_FONT)
    plt.suptitle(TITLE, fontsize = BIG_FONT)
    plt.show()

#Function for training and testing the CNN which has number_of_filters_per_layer convolutional layers (number_of_filters_per_layer is an array storing the number of filters in each convolution layer).
def trainTestCnn(number_of_filters_per_layer, X_train, y_train, X_val, y_val, X_test, y_test, epoch, train_set_size):

    #==============================|1. Training the Model|==============================
    #Constants
    INPUT_SHAPE = (28,28,1)
    EPOCHS = epoch
    BATCH_SIZE = 32

    #Building the CNN
    cnn = buildCNN(number_of_filters_per_layer, INPUT_SHAPE)

    #Sampling images from the main training set pool. Number of images used = optimal training set size for this model.
    X_train_new, y_train_new = sampleFromMainPool(X_train, y_train, train_set_size)

    #Training the CNN
    cnn.fit(X_train_new, y_train_new, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data = (X_val, y_val))

    #==============================|2. Threshold Tuning|==============================
    #Finding the threshold based on the validation set.
    y_pred = cnn.predict(X_val)

    #Finding the false positive rate (fpr), true positive rate (tpr), and the threshold that yielded the aforementioned fpr and tpr
    fpr, tpr, thresholds = roc_curve(y_val, y_pred)

    #Calculating Youden J's statistics
    youden_j = tpr - fpr

    #Variable for storing the largest Youden J's stats, and the corresponding threshold that yielded the Youden J Stats
    maximised_youden_j = 0
    maximised_threshold = 0
    maximised_index = 0

    #Finding the largest youden_j by looping through all elements in the array youden_j
    for i in range(len(youden_j)):
        #If the i^th youden j is larger than the largest youden j we've found so far, the i^th youden j is set as the largest youden j, maximised_youden_j. The threshold corresponding to this is also set as maximised_threshold
        if (youden_j[i] > maximised_youden_j):
            maximised_youden_j = youden_j[i]
            maximised_threshold = thresholds[i]
            maximised_index = i

    #Plotting ROC curve
    plt.figure(figsize = (8,6))
    plt.plot(fpr, tpr)
    plt.plot([0,1], [0,1], linestyle = '--')
    plt.scatter(fpr[maximised_index], tpr[maximised_index], marker = 'x', s = 100, color = 'red')
    plt.xlabel('False Positive Rate (FPR)', fontsize = MEDIUM_FONT)
    plt.ylabel('True Positive Rate (TPR)', fontsize = MEDIUM_FONT)
    plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize = BIG_FONT)
    plt.legend(['ROC Curve', 'Random Guess', 'Maximised (FPR-TPR)'], fontsize = SMALL_FONT)
    plt.grid()
    plt.show()

    #==============================|3. Testing the Model|==============================

    #Predicting the labels for images in the test set
    y_pred = cnn.predict(X_test)

    #Rounding up the probabilities from y_pred, based on the threshold obtained using Youden J's stats.
    y_pred = roundUpYPred(y_pred, maximised_threshold)

    #Finding the confusion matrix and performance metric
    findConfusionMatrix(y_test, y_pred, 'Confusion Matrix')
    findPerformanceMetric(y_test, y_pred)

    #==============================|4. Printing Out Misclassified Images for Analysis|==============================

    #Asking the user if they want to see the misclassified images. This is only an option for Task A because there are fewer misclassified images in Task A than B. Having this option in Task B will result in too many images being printed out.
    see_misclassified_images = input("Would you like to see the misclassified images? Print 'Y' if you do, and 'N' if you do not.")

    #User can only say Y or N. If the user's answer is anything else, the program will keep prompting the user to re-enter their answer.
    while(see_misclassified_images != 'Y' and see_misclassified_images != 'N'):
        see_misclassified_images = input("Please enter either 'Y' or 'N'.")

    #The program will only print out the misclassified images if the user entered Y. Otherwise, it will skip this step.
    if(see_misclassified_images == 'Y'):
        #Scale X_test back up so that it can be shown on a plot
        X_test = X_test*255

        #Goes through all images in the test set to find misclassified images by comparing the true labels (y_test) to the predicted labels (y_pred)
        for i in range(len(y_test)):
            #If y_test[i] != y_pred[i], the image is misclassified, so it is appended to either misclassified_dict[0] (if true label = 0) or misclassiied_dict[1] (if true label = 1)
            if(y_test[i] != y_pred[i]):
                misclassified_dict[int(y_test[i])].append(int(i))

        #Prints out all the false positive images
        for i in range(len(class0_misclassified_index)):
            plot_title = "Correct Label: " + str(y_test[class0_misclassified_index[i]])
            print_image = (X_test[class0_misclassified_index[i]]).astype('uint8')
            print_image = cv2.cvtColor(print_image, cv2.COLOR_RGB2BGR)
            plt.imshow(print_image)
            plt.title(plot_title, fontsize = BIG_FONT)
            plt.show()

        #Prints out all the false negative images
        for i in range(len(class1_misclassified_index)):
            plot_title = "Correct Label: " + str(y_test[class1_misclassified_index[i]])
            print_image = (X_test[class1_misclassified_index[i]]).astype('uint8')
            print_image = cv2.cvtColor(print_image, cv2.COLOR_RGB2BGR)
            plt.imshow(print_image)
            plt.title(plot_title, fontsize = BIG_FONT)
            plt.show()

#Function for finding the class distribution in each dataset.
def findClassDistribution(labels):
    #Class distribution stores the number of images in each class. E.g., class_distribution[0] stores the number of images in class0
    class_distribution = [0, 0]

    #NUMBER_OF_IMAGES stores the number of images in the dataset
    NUMBER_OF_IMAGES = len(labels)

    #Uses the values stored in labels to determine the if an image is in class0 or class1.
    for i in range(NUMBER_OF_IMAGES):
        #If the i^th image's label = 0, it belongs to class0. So, class_distribution[0] is incremented by 1.
        if(labels[i] == 0):
            class_distribution[0] = class_distribution[0] + 1
        else:
            class_distribution[1] = class_distribution[1] + 1

    #Returns the dataset's class distribution
    return class_distribution

#Function for sampling sample_size images from the main pool X_main
def sampleFromMainPool(X_main, y_main, sample_size):
    #Arrays for storing the images to be returned, and their corresponding labels
    return_images = np.zeros((sample_size, 28, 28))
    return_labels = np.zeros((sample_size, 1))

    #Saves the i^th image into return_images until sample_size images have been copied over
    for i in range(sample_size):
        return_images[i] = X_main[i]
        return_labels[i] = y_main[i]

    #Flattens out return_labels
    return_labels = np.array(return_labels)
    return_labels = return_labels.flatten()
    return_labels = return_labels.astype(int)

    #Returns the images and their corresponding labels
    return return_images, return_labels

#Function for finding the training and validation loss for a model
def findLossVsSampleSize(X_train, y_train, X_val, y_val, TRAINING_SET_SIZE, EPOCH, INPUT_SHAPE, number_of_filters_per_layer):
    #Sampling TRAINING_SET_SIZE images from the main pool.
    current_training_set_images, current_training_set_labels = sampleFromMainPool(X_train, y_train, TRAINING_SET_SIZE)
    EPOCHS = EPOCH
    BATCH_SIZE = 32

    #Building and training the CNN
    cnn = buildCNN(number_of_filters_per_layer, INPUT_SHAPE)
    cnn.fit(current_training_set_images, current_training_set_labels, epochs = EPOCHS, batch_size = BATCH_SIZE, validation_data=(X_val, y_val))

    #Extracting the training and validation losses
    train_loss = cnn.evaluate(current_training_set_images, current_training_set_labels)
    val_loss = cnn.evaluate(X_val, y_val)

    #Returning the training and validation losses
    return train_loss[0], val_loss[0]

#Function for carrying out Task A of the assignment
def Task_A_Tasks(dataset):
    #============================================|1. Analysing Given Dataset|============================================

    #Asking the user if they want to see the learning curves (i.e., the losses vs. training set size, and losses vs. epoch curves). It takes time to compute these, so some users may not want to see them.
    user_choice = input("Would you also like to see the learning curves? Enter 'Y' if you do, and 'N' if you do not.")

    #Keeps prompting the user until they enter either 'Y' or 'N'
    while(user_choice != 'Y' and user_choice != 'N'):
        user_choice = input("Please enter either 'Y' or 'N'.")

    # ============================================|1.1. Splitting the Dataset into Train, Val, and Test Sets|============================================
    #Splitting provided dataset into train, val, test sets
    X_train = dataset['train_images']
    X_val = dataset['val_images']
    X_test = dataset['test_images']

    y_train = dataset['train_labels']
    y_val = dataset['val_labels']
    y_test = dataset['test_labels']

    #Finding the class distribution for each set
    train_distribution = findClassDistribution(y_train)
    val_distribution = findClassDistribution(y_val)
    test_distribution = findClassDistribution(y_test)

    #Plotting the class distribution for each set as a 1x3 plot
    fig, axes = plt.subplots(nrows = 1, ncols = 3, figsize = (10,10))

    axes[0].pie(train_distribution, autopct='%1.1f%%', textprops={'fontsize': BIG_FONT}, colors = my_colors)
    axes[0].set_title("Training Set", fontsize = VERY_BIG_FONT)
    axes[1].pie(val_distribution, autopct='%1.1f%%', textprops={'fontsize': BIG_FONT}, colors = my_colors)
    axes[1].set_title("Validation Set", fontsize = VERY_BIG_FONT)
    axes[2].pie(test_distribution, autopct='%1.1f%%', textprops={'fontsize': BIG_FONT}, colors = my_colors)
    axes[2].set_title("Test Set", fontsize = VERY_BIG_FONT)
    plt.legend(['Class 0', 'Class 1'], bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize = BIG_FONT)
    plt.tight_layout()
    plt.show()

    #============================================|1.2. Plotting Sample Images|============================================
    #Plotting the images as a 1x2 plot
    #The indexes 5 and 2 are chosen because they best visually represent what a "Normal" and "Pnuemonia" scan would look like. E.g., a normal scan would show a transparent chest without any white hazy areas.
    #These indexes are obtained by printing out a few images from the training set on a separate code (this code just prints out the first 20 images from the training set, along with their labels), and the images that best represent the concept of a "normal" and "pneumonia" scan are chosen to be shown here.
    #This does not affect the models' performance, these images are just chosen because they look good and can provide the reader with a good understanding of what normal and pnuemonia images look like.
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (5,5))
    class0_image = cv2.cvtColor(X_train[5], cv2.COLOR_RGB2BGR)
    axes[0].imshow(class0_image)
    axes[0].set_title("Class 0 (Normal)", fontsize = BIG_FONT)

    class1_image = cv2.cvtColor(X_train[2], cv2.COLOR_RGB2BGR)
    axes[1].imshow(class1_image)
    axes[1].set_title("Class 1 (Pneumonia)", fontsize = BIG_FONT)
    plt.tight_layout()
    plt.show()

    #============================================|2. Data Pre-Processing|============================================

    #============================================|2.1. Creating the Main Pool of Images|============================================
    #Increasing the population of the training set to 20 000 (10 000 for class0, 10 000 for class1) by adding noisy images to the original dataset. This also balances the training set, so both classes now have the same population.
    X_train_main_pool, y_train_main_pool = increaseClassPopulation(X_train, y_train, 20000)

    #============================================|2.2. Plotting a Noisy Image vs. Its Original|============================================
    #Generating an example of a noisy image. This is the same method used in addNoiseToImage, which is used by the function increaseClassPopulation to increase the training set size.
    original_image = X_train[0]
    mean = 0
    standard_deviation = random.uniform(1, 5)
    noise = np.random.normal(mean, standard_deviation, original_image.shape).astype(np.uint8)
    noisy_image = Image.fromarray(np.clip(original_image + noise, 0, 255))
    noisy_image = np.array(noisy_image)

    #Plotting the original image on the left, and its noisy variant on the right
    fig, axes = plt.subplots(nrows = 1, ncols = 2, figsize = (5,5))
    original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    axes[0].imshow(original_image)
    axes[0].set_title("Original Image", fontsize = BIG_FONT)

    noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_RGB2BGR)
    axes[1].imshow(noisy_image)
    axes[1].set_title("Noisy Image", fontsize = BIG_FONT)

    plt.tight_layout()
    plt.show()

    #============================================|2.3. Comparing Unbalanced and Balanced Training Sets|============================================
    #Finding the balanced training set's class distribution
    train_balanced_distribution = findClassDistribution(y_train_main_pool)

    #Plots the unbalanced training set's (from PneumoniaMNIST, without adding any augmented images to it) distribution to the left, and the balanced training set's distribution to the right.
    fig,axes = plt.subplots(nrows = 1, ncols = 2, figsize = (8,8))

    axes[0].pie(train_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT}, colors = my_colors)
    axes[0].set_title("Unbalanced Training Set", fontsize = BIG_FONT)

    axes[1].pie(train_balanced_distribution, autopct='%1.1f%%', textprops={'fontsize': MEDIUM_FONT}, colors = my_colors)
    axes[1].set_title("Balanced Training Set", fontsize = BIG_FONT)

    plt.legend(['Class 0', 'Class 1'], bbox_to_anchor = (1.05, 1.0), loc = 'upper left', fontsize = MEDIUM_FONT)
    plt.tight_layout()
    plt.show()

    #============================================|2.4. Scaling Down Images|============================================
    #Scaling down the train, val, and test images' pixels' intensities from the range 0-255 to 0-1
    X_train_main_pool = scaleDown(X_train_main_pool)
    X_val_scaled = scaleDown(X_val)
    X_test_scaled = scaleDown(X_test)

    #============================================|3. Model Training, Hyperparameter Tuning, and Testing|============================================
    #Arrays for storing the number of filters in each conv layers
    pyramid = [32, 48, 64]
    reverse_pyramid = [64, 48, 32]
    architectures = [pyramid, reverse_pyramid]

    #Refer to Section 4.3.1 in the report for where these numbers are obtained from
    optimal_epoch = [4, 2]
    optimal_set_size = [20000, 15000]

    #Training + testing models with the pyramid and reverse_pyramid architecture
    for i in range(2):
        trainTestCnn(architectures[i], X_train_main_pool, y_train_main_pool, X_val_scaled, y_val, X_test_scaled, y_test, optimal_epoch[i], optimal_set_size[i])

    #============================================|4. Learning Curve|============================================

    #Program will run this section only if the user chose Y (they want to see the learning curve) earlier.
    if(user_choice == 'Y'):

        #Array for storing different training set sizes
        training_set_sizes = [100, 500, 1000, 5000, 10000, 15000, 20000, 25000, 30000, 35000, 40000]

        #Arrays for storing the train and validation losses when a specific training set size is used
        train_loss = np.zeros((len(training_set_sizes), 1))
        val_loss = np.zeros((len(training_set_sizes), 1))

        #Training and evaluating the model's performance when different training set sizes and epochs are used
        for i in range(2):
            #============================================|4.1. Losses vs. Training Set Size|============================================
            #Training and evaluating the model's performance when different training set sizes are used.
            for j in range(len(training_set_sizes)):
                #Finding the training and validation losses
                current_train_loss, current_val_loss = findLossVsSampleSize(X_train_main_pool, y_train_main_pool, X_val_scaled, y_val, training_set_sizes[j], optimal_epoch[i], (28, 28, 1), architectures[i])

                #Stores the obtained losses in the arrays
                train_loss[j] = current_train_loss
                val_loss[j] = current_val_loss

            #Flattening out the train and validation losses
            train_loss = np.array(train_loss)
            train_loss = train_loss.flatten()

            val_loss = np.array(val_loss)
            val_loss = val_loss.flatten()

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

            #============================================|4.2. Losses vs. Epoch Number|============================================
            #Building the model
            cnn = buildCNN(architectures[i], (28,28,1))

            #Sampling optimal_set_size[i] images from the training set. i.e., the optimal number of images (for this model) will be used to train the model.
            X_train_new, y_train_new = sampleFromMainPool(X_train_main_pool, y_train_main_pool, optimal_set_size[i])

            #Storing the training and validation losses across each epoch. Model will be trained across 15 epochs.
            history = cnn.fit(X_train_new, y_train_new, epochs=15, batch_size=32, validation_data=(X_val_scaled, y_val))

            #Plotting the model's loss vs. epoch curve
            training_loss = history.history['loss']
            validation_loss = history.history['val_loss']
            epochs = range(1, len(training_loss) + 1)

            plt.figure(figsize=(8, 6))
            plt.plot(epochs, training_loss, label='Training')
            plt.plot(epochs, validation_loss, label='Validation')
            plt.title('Training and Validation Loss', fontsize=BIG_FONT)
            plt.legend(['Training Loss', 'Validation Loss'], fontsize=MEDIUM_FONT)
            plt.xlabel('Epochs', fontsize=MEDIUM_FONT)
            plt.ylabel('Loss', fontsize=MEDIUM_FONT)
            plt.xticks(fontsize=SMALL_FONT)
            plt.yticks(fontsize=SMALL_FONT)
            plt.grid()
            plt.show()
    else:
        return