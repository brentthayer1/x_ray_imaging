import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras import metrics
import datetime

# DIRECTORIES
train_data_dir = '/Volumes/b/Galvanize/DS-RFT4/capstones-RFT4/datasets/covid_pneumonia/712865_1242442_bundle_archive/train'
test_data_dir = '/Volumes/b/Galvanize/DS-RFT4/capstones-RFT4/datasets/covid_pneumonia/712865_1242442_bundle_archive/test'

nb_classes = 3
class_weights = {0 : 4,
                1 : 1,
                2 : 1}
seed = 42

validation_split = 0.2
img_rows, img_cols = 255, 255

batch_size = 32
nb_epoch = 10     

# CONV2D 1
nb_filters_1 = 32
kernel_size_1 = (3, 3)
stride_size_1 = (1, 1)
pool_size_1 = (2, 2)
activation_1 = 'swish'
dropout_1 = 0.7

# CONV2D 2
nb_filters_2 = 64
kernel_size_2 = (3, 3)
stride_size_2 = (1, 1)
pool_size_2 = (2, 2)
activation_2 = 'swish'
dropout_2 = 0.8

# CONV2D 3
nb_filters_3 =  128
kernel_size_3 = (3, 3)
stride_size_3 = (2, 2)
pool_size_3 = (2, 2)
activation_3 = 'swish'
dropout_3 = 0.7

# CONV2D 4
nb_filters_4 = 256
kernel_size_4 = (3, 3)
stride_size_4 = (2, 2)
pool_size_4 = (2, 2)
activation_4 = 'swish'
dropout_4 = 0.7

# CONV2D 5
nb_filters_5 = 512
kernel_size_5 = (3, 3)
stride_size_5 = (2, 2)
pool_size_5 = (2, 2)
activation_5 = 'swish'
dropout_5 = 0.6

# DENSE 6
units_6 = 512
activation_6 = 'swish'
dropout_6 = 0.7

# DENSE 7
units_7 = 256
activation_7 = 'swish'
dropout_7 = 0.6

# DENSE 8
units_8 = 128
activation_8 = 'swish'
dropout_8 = 0.7

# DENSE 9
units_9 = 64
activation_9 = 'swish'
dropout_9 = 0.8

# DENSE 10
units_10 = 32
activation_10 = 'swish'
dropout_10 = 0.7

# DENSE 10
units_11 = 16
activation_11 = 'swish'
dropout_11 = 0.6

# DENSE 10
units_12 = 8
activation_12 = 'swish'
dropout_12 = 0.5

# COMPILE
compile_optimizer = 'adam'


# CNN
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_cols, img_rows)
    chanDim = 1
else:
    input_shape = (img_cols, img_rows, 1)
    chanDim = -1

model = Sequential()

model.add(Conv2D(nb_filters_1,
                    (kernel_size_1[0], kernel_size_1[1]),
                    strides=(stride_size_1[0], stride_size_1[1]),
                    padding='same', activation=activation_1,
                    input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size_1))
# model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(dropout_1)) 

model.add(Conv2D(nb_filters_2,
                    (kernel_size_2[0], kernel_size_2[1]),
                    strides=(stride_size_2[0], stride_size_2[1]),
                    padding='same', activation=activation_2))
model.add(MaxPooling2D(pool_size=pool_size_2))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(dropout_2)) 

model.add(Conv2D(nb_filters_3,
                    (kernel_size_3[0], kernel_size_3[1]),
                    strides=(stride_size_3[0], stride_size_3[1]),
                    padding='same', activation=activation_3))
model.add(MaxPooling2D(pool_size=pool_size_3))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(dropout_3)) 

model.add(Conv2D(nb_filters_4,
                    (kernel_size_4[0], kernel_size_4[1]),
                    strides=(stride_size_4[0], stride_size_4[1]),
                    padding='same', activation=activation_4))
model.add(MaxPooling2D(pool_size=pool_size_4))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(dropout_4))

model.add(Conv2D(nb_filters_5,
                    (kernel_size_5[0], kernel_size_5[1]),
                    strides=(stride_size_5[0], stride_size_5[1]),
                    padding='same', activation=activation_5))
model.add(MaxPooling2D(pool_size=pool_size_5))
model.add(BatchNormalization(axis=chanDim))
model.add(Dropout(dropout_5))

model.add(Flatten())

model.add(Dense(units_6, activation=activation_6))
model.add(Dropout(dropout_6))

model.add(Dense(units_7, activation=activation_7))
model.add(Dropout(dropout_7))

model.add(Dense(units_8, activation=activation_8))
model.add(Dropout(dropout_8))

model.add(Dense(units_9, activation=activation_9))
model.add(Dropout(dropout_9))

model.add(Dense(units_10, activation=activation_10))
model.add(Dropout(dropout_10))

model.add(Dense(units_11, activation=activation_11))
model.add(Dropout(dropout_11))

model.add(Dense(units_12, activation=activation_12))
model.add(Dropout(dropout_12))

model.add(Dense(nb_classes, activation='softmax'))

METRICS = [
            metrics.CategoricalAccuracy(name='ACCURACY'),
            metrics.Precision(name='PRECISION'),
            metrics.Recall(name='RECALL'),
            metrics.AUC(name='AUC'),
            metrics.TruePositives(name='TP'),
            metrics.TrueNegatives(name='TN'),
            metrics.FalsePositives(name='FP'),
            metrics.FalseNegatives(name='FN')]

model.compile(loss='categorical_crossentropy',
                optimizer=compile_optimizer,
                metrics=METRICS)

# GENERATORS
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split)

test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='training',
    seed=seed)

validation_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True,
    subset='validation',
    seed=seed)

test_generator = test_datagen.flow_from_directory(
    test_data_dir,
    target_size=(img_cols, img_rows),
    color_mode='grayscale',
    batch_size=batch_size,
    class_mode='categorical',
    seed=seed)

# TENSORBOARD
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

# WEIGHTS

# FIT
model.fit_generator(
    train_generator,
    steps_per_epoch= train_generator.samples // batch_size,
    epochs=nb_epoch,
    validation_data= validation_generator,
    validation_steps= validation_generator.samples // batch_size,
    class_weight=class_weights,
    callbacks=[tensorboard_callback])

# model.save_weights('first_try.h5')

model.summary()

# label_map = (train_generator.class_indices)
# print(label_map)