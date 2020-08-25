import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K
from keras import metrics
import datetime


seed = 42
validation_split = 0.2
img_rows, img_cols = 255, 255

train_data_dir = '/Volumes/b/Galvanize/DS-RFT4/capstones-RFT4/datasets/covid_pneumonia/712865_1242442_bundle_archive/train'
test_data_dir = '/Volumes/b/Galvanize/DS-RFT4/capstones-RFT4/datasets/covid_pneumonia/712865_1242442_bundle_archive/test'

batch_size = 32
nb_epoch = 1     

nb_filters_1 = 15
nb_filters_2 = 15
nb_filters_3 = 15

pool_size_1 = (3, 3)
pool_size_2 = (3, 3)
pool_size_3 = (3, 3)


kernel_size = (4, 4)
nb_classes = 3

if K.image_data_format() == 'channels_first':
    input_shape = (1, img_cols, img_rows)
else:
    input_shape = (img_cols, img_rows, 1)

model = Sequential()

model.add(Conv2D(nb_filters_1,
                    (kernel_size[0], kernel_size[1]),
                    padding='valid', activation='swish',
                    input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size_1))
model.add(Dropout(0.6)) 

model.add(Conv2D(nb_filters_2,
                    (kernel_size[0], kernel_size[1]),
                    padding='valid', activation='swish'))
model.add(MaxPooling2D(pool_size=pool_size_2))
model.add(Dropout(0.6)) 

model.add(Conv2D(nb_filters_2,
                    (kernel_size[0], kernel_size[1]),
                    padding='valid', activation='swish'))
model.add(MaxPooling2D(pool_size=pool_size_3))
model.add(Dropout(0.6)) 

model.add(Flatten())

model.add(Dense(32, activation='swish'))
model.add(Dropout(0.6))

model.add(Dense(32, activation='swish'))
model.add(Dropout(0.6))

model.add(Dense(32, activation='swish'))
model.add(Dropout(0.6))

model.add(Dense(nb_classes, activation='softmax'))
# model.add(Activation('softmax'))

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
                optimizer='adam',
                metrics=METRICS
                # metrics=['categorical_accuracy']
                )

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

log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

class_weight = {2 : 16.8414376,
                0 : 1,
                1 : 1.45950898}

model.fit_generator(
    train_generator,
    steps_per_epoch= train_generator.samples // batch_size,
    epochs=nb_epoch,
    validation_data= validation_generator,
    validation_steps= validation_generator.samples // batch_size,
    class_weight=class_weight,
    callbacks=[tensorboard_callback])

# model.save_weights('first_try.h5')

model.summary()