from __future__ import division
import numpy as np
from numpy.random import choice
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import os
from collections import Counter
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.utils import np_utils
from keras.optimizers import SGD
import cv2
import random
from tensorflow import keras
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical

import numpy as np
np.random.seed(42)

class classifier(object):

    def __init__(self, DATADIR, FOLDER, categories):
        self.DATADIR = DATADIR
        self.FOLDER = FOLDER
        self.categories = categories
        self.X = None
        self.y = None
        self.X_test = None
        self.X_train = None
        self.y_test = None
        self.y_test = None

    def load_data(self, img_rows, img_cols):
        self.img_rows = img_rows
        self.img_cols = img_cols
        data = []
        for category in self.categories.keys():
            path = self.DATADIR + self.FOLDER + category
            files = os.listdir(path)
            category_num = self.categories[category]
            for im in files:
                img_array = cv2.imread(os.path.join(path, im), cv2.IMREAD_GRAYSCALE)
                new_array = cv2.resize(img_array, (img_rows, img_cols))
                data.append((new_array, category_num))
        random.shuffle(data)
        X = []
        y = []
        for features, label in data:
            X.append(np.array(features))
            y.append(label)
        self.X = np.array(X).reshape(-1, img_rows, img_cols)
        self.y = np.array(y)
        return self.X, self.y

    def featurize_data(self, test_size=0.2, random_seed=42):
        X_train, X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,    test_size=test_size, random_seed=random_seed)
        X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols)
        X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols)
        X_train = X_train.astype('float32')
        X_test = X_test.astype('float32')
        self.X_train = X_train / 255
        self.X_test = X_test / 255
        print('X_train shape:', X_train.shape)
        print(X_train.shape[0], 'train samples')
        print(X_test.shape[0], 'test samples')
        return self.X_train, self.X_test, self.y_train, self.y_test



if __name__ == '__main__':

    DATADIR = '/Volumes/b/Galvanize/DS-RFT4/capstones-RFT4/datasets/covid_pneumonia/small_data_set'
    FOLDER = '/train/'
    categories = {'COVID-19/' : 0, 'normal/' : 1, 'pneumonia/' : 2} 

    batch_size = 15  # number of training samples used at a time to update the weights
    # nb_classes = 10    # number of output possibilities: [0 - 9] KEEP
    nb_epoch = 2       # number of passes through the entire train dataset before weights "final"
    img_rows, img_cols = 255, 255   # the size of the MNIST images KEEP
    input_shape = (img_rows, img_cols, 1)   # 1 channel image input (grayscale) KEEP
    nb_filters = 12    # number of convolutional filters to use
    pool_size = (2, 2)  # pooling decreases image size, reduces computation, adds translational invariance
    kernel_size = (4, 4)  # convolutional kernel size, slides over image to learn features

    X, y = load_data(DATADIR, FOLDER, categories, img_rows, img_cols)
    X_train, X_test, Y_train, Y_test = load_and_featurize_data(X, y, img_rows, img_cols)


    model = define_model(nb_filters, kernel_size, input_shape, pool_size)

    # during fit process watch train and test error simultaneously
    model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
              verbose=1, validation_data=(X_test, Y_test))

    score = model.evaluate(X_test, Y_test, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])  # this is the one we care about





