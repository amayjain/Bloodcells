# Libraries needed

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import os

import imagesize

from sklearn.preprocessing import LabelEncoder
from sklearn import metrics

import tensorflow as tf
from tensorflow import keras
from keras import layers, callbacks

from scipy.ndimage import sobel



# ------------------------------------------------------------------------------------------------------------------------------------------------------------


# Functions

# Function - Open image paths in folder storing images
def image_df(folder_names):

    '''
    Outputs a dataframe for image paths (as strings), bloodcell type, and image dimension information.

    Args:
        1) folder_names (list): list of bloodcell type folders

    Returns:
        Dataframe with all image paths, bloodcell types, and image dimensions
    '''

    # initialize empty list to store dataframes that contain image strings and bloodcell type
    dfs = []

    # loop through bloodcell types and store image paths and bloodcell categories
    for i in range(len(folder_names)):

        # jpg string paths
        images = os.listdir('bloodcells_dataset/' + folder_names[i]) 

        # dataframe holding specific bloodcell type info (string path and type name)
        df = pd.DataFrame(data = {'images': images, 'type': folder_names[i]})

        # append dataframe to list
        dfs.append(df)

    # combine all dataframes
    all_data = pd.concat(dfs)

    # Remove image paths that may have been accidentally copied or contain .DS_Store
    all_data = all_data[all_data['images'].str.contains('.DS_Store') == False]
    all_data = all_data[all_data['images'].str.contains('copy') == False]

    # Convert bloodcell types to numbers for our model
    le = LabelEncoder()
    all_data['type_category'] = all_data['type'] # keep a copy of bloodcell types by name
    all_data['type'] = le.fit_transform(all_data['type'])

    # Store dimensions of image incase we find different dimensions 
    dimensions = pd.Series([imagesize.get('bloodcells_dataset/All_Images/' + x) for x in all_data['images']])
    widths, heights = map(list, zip(*dimensions))
    all_data['width'] = widths
    all_data['height'] = heights
    all_data['dimensions'] = all_data['width'].astype(str) + ' x ' + all_data['height'].astype(str)

    # Reset index 
    all_data = all_data.reset_index(drop = True)
    
    return all_data


# Function - Use pixel data to calculate edges
def edge_detection(input_data):

    '''
    Detects edges for images through the use of gradients
    
    Args:
        input_data (array): image array
    Returns:
        x_sobel (array): horizontal sobel-filtered array with same dimension as input_data
        y_sobel (array): vertical sobel-filtered array with same dimension as input_data
        norm (array): combined array magnitude after filter application
    '''
    
    input_data = input_data.astype('int32')
    
    x_sobel = sobel(input_data, axis = 0) # compute horizontal gradient
    
    y_sobel = sobel(input_data, axis = 1) # compute vertical gradient
    
    norm = np.sqrt(x_sobel**2 + y_sobel**2) # compute norm
    
    norm *= 255.0 / np.max(norm) # normalization
    
    return x_sobel, y_sobel, norm

# Function - Graph an example of edge detetcted images
def plot_edges(image, x_sobel, y_sobel, norm):

    '''
    Plot images after edge detection
    
    Args:
        1) input_data (array): array of image pixels
        2) x_sobel (array): horizontal sobel-filtered array with same dimension as input_data
        3) y_sobel (array): vertical sobel-filtered array with same dimension as input_data
        4) norm (array): combined array magnitude after filter application
    
    Returns:
        4 subplots revealing horizontal + vertical edge detection before and after
    '''
    
    image = image.astype('int32')
    
    x_sobel = x_sobel.astype("int32")
    
    y_sobel = y_sobel.astype("int32")
    
    norm = norm.astype("int32")
    
    fig, ax = plt.subplots(2, 2, figsize = (100, 100))
    
    # plt.gray()  
    # show the filtered result in grayscale
    
    ax[0, 0].imshow(image[1])
    
    ax[0, 1].imshow(x_sobel[1])
    
    ax[1, 0].imshow(y_sobel[1])
    
    ax[1, 1].imshow(norm[1])
    
    # plt.gray()  
    # show the filtered result in grayscale
    
    titles = ["original", "horizontal", "vertical", "norm"]
    
    for i, ax in enumerate(ax.ravel()):
        ax.set_title(titles[i])
        ax.axis("off")
    
    plt.show()


# Function - Train tensorflow models
def train_model(model, train_data, train_labels, val_data, val_test, test_data, test_labels, optimizer = 'adam', epochs = 5, batch_size = 64, callback_patience = 3):

    '''
    Trains a tensorflow model on train, val, and test data and return performance metrics

    Args:
        1) model (tensorflow model): model to train
        2) train_data (numpy array): array of pixel values for training
        3) train_labels: bloodcell type labels for training 
        4) val_data (numpy array): array of pixel values for validation
        5) val_labels: bloodcell type labels for validation 
        6) test_data (numpy array): array of pixel values for testing
        7) test_labels: bloodcell type labels for testing
        8) optimizer (string): optimizer to train model with
        9) epochs (int): # of epochs to train data
        10) batch_size (int): number of images to train per iteration
        11) callback_patience (int): number of epochs to stop model training after if there is no improvement in decreasing loss

    Returns:
        Model history storing train/val accuracies alongside predicted bloodcell types, test accuracy, and confusion matrix metrics
    '''

    # Add a callback to make sure that the model doesn't continue training if it doesn't see improvement after a certain amount of epochs (callback_patience)
    callback = callbacks.EarlyStopping(monitor = 'loss', 
                                       patience = callback_patience, 
                                       restore_best_weights = True)

    # Compile model with optimizer, loss function (classification adjusted), and metrics (accuracy for classification)
    model.compile(optimizer = optimizer,
                  loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits = True), 
                  metrics = ['accuracy'])

    # Train model with train and val data
    history = model.fit(train_data, 
                        train_labels, 
                        validation_data = (val_data, val_test),
                        epochs = epochs, 
                        batch_size = batch_size,
                        callbacks = [callback])

    # Store predictions of bloodecell types
    predictions = (model.predict(test_data)).argmax(axis = 1)

    # Store test accuracy 
    test_accuracy = np.sum(predictions == test_labels) / len(test_labels)

    # Store confusion matrix metrics
    confusion_matrix = metrics.confusion_matrix(test_labels, predictions)

    return history, predictions, test_accuracy, confusion_matrix


# Function - reshape 4d arrays into 2d arrays for FNN models
def image_reshape(X):
    
    """
    Reshapes a 3d/4d array of images into a 2d array

    Args:
        x (numpy array): 3d/4d array of images

    Returns:
        Reshaped 2d array of images
    """
    
    # Find number of images (1st dimension of 2d array)
    num_of_images = X.shape[0]

    # second dimension of 2d array (product of all remaining dimensions)
    resize = np.prod(X.shape[1:])

    # return 2d array
    return X.reshape(num_of_images, resize)    



# ------------------------------------------------------------------------------------------------------------------------------------------------------------



# Classes

# Class - Sample image data
class Sampling:

    '''
    Samples a bloodcell dataframe either by down-sampling or proportionally selecting the image count by bloodcell type

    Methods:
        1) __init__: initialize the data and sampling method
        2) sample_data: performs the sampling on the initialized data and given sampling method
        
    '''

    def __init__(self, data, sampling_method):

        self.data = data # bloodcell dataframe to sample
        
        self.sampling_method = sampling_method # downsample or porportional

    def sample_data(self, sampling_percent = 0.8):

        '''
        Samples the initialized data for the given sampling method

        Args:
            1) sampling_percent (float): Percent of data to use for training (only applicable for proportional sampling)

        Returns:
            Dataframe of sampled images and label
        '''

        # copy of data
        df = self.data.copy()

        # Find count of images by bloodcell type
        category_counts = df[['type']].value_counts().reset_index(name = 'count')

        # Will store training dataframes to concatenate later
        train_dfs = []

        # Down sampling
        if self.sampling_method == 'down':

            # Loop through each bloodcell type
            for i in range(len(category_counts)):

                # Select specific bloodcell type
                type = category_counts['type'][i]

                # If bloodcell type has more than 2000 images, sample 1500 images and if not, then sample 1000 images
                if category_counts['count'][i] >= 2000:
                    add_samples = df[df['type'] == type].sample(1500)
                else: 
                    add_samples = df[df['type'] == type].sample(1000)

                # Add sampled images to train_dfs list
                train_dfs.append(add_samples)

            # Combine all training dfs into one
            train = pd.concat(train_dfs)

            # Separate remaining data to make validation and test sets
            remaining_data = df[~df['images'].isin(train['images'])]

            # validation data consists of half of the remaining data
            validation = remaining_data.sample(int(len(remaining_data) / 2))

        # Proportional Sampling
        elif self.sampling_method == 'proportional': 

            # Number of training samples needed
            num_samples = int(sampling_percent * len(df))

            # Number of training and validation samples needed for each bloodcell type based on proportion
            # validation set is 20% of the training set size
            category_counts['prop'] = category_counts['count'] / len(df)
            category_counts['train_samples'] = (category_counts['prop'] * num_samples).astype('int32')
            category_counts['val_samples'] = (category_counts['train_samples'] * 0.25).astype('int32')
            category_counts['train_samples'] = category_counts['train_samples'] - category_counts['val_samples']

            # Store validation dataframes to concatenate later
            val_dfs = []

            # First sample data for training set 
            for i in range(len(category_counts)):

                # Select specific bloodcell type
                type = category_counts['type'][i]

                # Number of samples to take for specific bloodcell type
                samples = category_counts['train_samples'][i]

                # Separate training samples into its own dataframe
                add_samples = df[df['type'] == type].sample(samples)

                # Add sampled images to train_dfs list
                train_dfs.append(add_samples)

            # Combine all training dfs into one
            train = pd.concat(train_dfs)

             # Separate remaining data to make validation and test sets
            remaining_data = df[~df['images'].isin(train['images'])]

            # Sample remaining data for validation set
            for i in range(len(category_counts)):

                # Select specific bloodcell type
                type = category_counts['type'][i]

                # Number of samples to take for specific bloodcell type
                samples = category_counts['val_samples'][i]

                # Separate validation samples into its own dataframe
                add_samples = remaining_data[remaining_data['type'] == type].sample(samples)

                # Add sampled images to tval_dfs list
                val_dfs.append(add_samples)
            
            # Combine all validation dfs into one
            validation = pd.concat(val_dfs)

        # Inase there are NAs present in the train set, drop them
        # Convert all float columns to int columns (Bloodcell type marked as float during sampling process, eg. 1.0 instead of 1)
        train = train.dropna(how = 'all')
        float_cols = train.select_dtypes(np.number)
        train[float_cols.columns] = float_cols.astype('int32')

        # Test data is remanining data minus the validation data
        test = remaining_data[~remaining_data['images'].isin(validation['images'])]

        # Return train, val, test dataframes
        return train, validation, test


# Class - Convert images to pixel arrays            
class Convert_Images:

    '''
    Converts image string paths to numpy arrays

    Methods:
        1) __init__: initialize the data, image path names, and bloodcell type labels
        2) load_image: convert an image to a tensorflow array
        3) image_arrays_and_labels: return all images from the data as 4d numpy arrays alongside their corresponding bloodcell type labels
    '''

    def __init__(self, data, data_augment = False):

        self.data_augment = data_augment # will augment only training data

        if self.data_augment:
            self.data = pd.concat([data, data], axis = 0) # bloodcell dataframe that contains images to convert plus a copy for augmented images
        else:
            self.data = data # bloodcell dataframe that contains images to convert

        self.file_names = (self.data)['images'].apply(lambda x: 'bloodcells_dataset/All_Images/' + x) # extract image string paths from data

        self.labels = (self.data)['type'] # extract bloodcell type labels from data

    def load_image(self, file_name, resize, edge):

        '''
        Convert an image into a tensor

        Args:
            1) file_name (string): image string path
            2) resize (int): resize pixels of image
            3) edge (bool): indicate if preprocessing needs to be done for edge detection

        Returns:
            tensor with image pixel values
        '''

        # read in image
        raw = tf.io.read_file(file_name)

        # turn image into tensor
        tensor = tf.io.decode_image(raw, expand_animations = False)

        # resize image tensor
        tensor = tf.image.resize(tensor, size = [resize, resize])

        # normalize pixel values, but not for edge detection (needs its own processing which will be done outside the class)
        if edge == True:
            tensor = tf.cast(tensor, tf.float32)
        else:    
            tensor = tf.cast(tensor, tf.float32) / 255.0
        
        return tensor

    def image_arrays_and_labels(self, resize = 32, edge = False):

        '''
        Convert dataframe of image paths into numpy arrays and bloodcell type labels

        Args:
            1) resize (int): resize pixels of image
            2) edge (bool): indicate if preprocessing needs to be done for edge detection

        Returns:
            4d numpy array of image pixel values and bloodcell type labels
        '''

        # Convert image file names into tensorflow dataset
        dataset = tf.data.Dataset.from_tensor_slices(self.file_names)
        
        # Apply function to each image that will convert it into a tensor    
        dataset = dataset.map(lambda file_name: self.load_image(file_name, resize, edge))

        # Data augmentation
        if self.data_augment:
    
            data_augmentation = tf.keras.Sequential(
                
                [
                    
                  layers.RandomFlip('horizontal_and_vertical'),
                    
                  layers.RandomRotation(0.5),
                    
                ]
            )

            # First half of data will not be augmented and remain normal, while second half of data will be augmented
            augment = np.concatenate((np.zeros(int(len(self.data) / 2)), np.ones(int(len(self.data) / 2))))

            dataset = tf.data.Dataset.zip((dataset, tf.data.Dataset.from_tensor_slices(augment)))
    
            dataset = dataset.map(lambda image, augment: data_augmentation(image) if augment == 1 else image)

        # Convert tensors into numpy arrays
        images = np.array(list(dataset))
            
        return images, self.labels    