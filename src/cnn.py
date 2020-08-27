from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

downloaded = drive.CreateFile({'id':"1_gaHq3ubfIokmlLX7a47C7Sq01vq4utp"})
downloaded.GetContentFile('pneumonia.zip')
# downloaded = drive.CreateFile({'id':"1p5dyUcpCWGHwA_O5o6nUrL25UhkYqNGK"})
# downloaded.GetContentFile('pneum_normal_copy.zip')

!unzip pneumonia.zip
# !unzip pneum_normal_copy.zip

import tensorflow as tf
from tensorflow.keras.callbacks import TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, BatchNormalization
from keras import backend as K
from keras import metrics
import datetime
from time import time
from tensorboardcolab import TensorBoardColab, TensorBoardColabCallback
from keras.optimizers import Adam, RMSprop

# train_data_dir = '/content/pneum_normal_copy/train'
# test_data_dir = '/content/pneum_normal_copy/test'
train_data_dir = '/content/chest_xray/train'
val_data_dir = '/content/chest_xray/test'
test_data_dir_norm = '/content/chest_xray/val/NORMAL'
test_data_dir_pneum = '/content/chest_xray/val/PNEUMONIA'



#-------------------------------------------------------------------
seed = 42
nb_classes = 1
class_weights = {0 : 1.20,
                1 : 1.00}
# validation_split = 0.2
img_rows, img_cols = 300, 300
batch_size = 32
nb_epoch = 40     
#-------------------------------------------------------------------
                            # CONV2D 1
nb_filters_1 = 32
kernel_size_1 = (3, 3)
stride_size_1 = (1, 1)
pool_size_1 = (2, 2)
activation_1 = 'swish'
#-------------------------------------------------------------------
                            # CONV2D 2
nb_filters_2 = 64
kernel_size_2 = (3, 3)
stride_size_2 = (1, 1)
pool_size_2 = (2, 2)
activation_2 = 'swish'
#-------------------------------------------------------------------
                            # CONV2D 3
nb_filters_3 =  64
kernel_size_3 = (3, 3)
stride_size_3 = (2, 2)
pool_size_3 = (2, 2)
activation_3 = 'swish'
#-------------------------------------------------------------------
                            # CONV2D 4
nb_filters_4 = 128
kernel_size_4 = (3, 3)
stride_size_4 = (2, 2)
pool_size_4 = (2, 2)
activation_4 = 'swish'
#-------------------------------------------------------------------
                            # CONV2D 5
nb_filters_5 = 128
kernel_size_5 = (3, 3)
stride_size_5 = (2, 2)
pool_size_5 = (2, 2)
activation_5 = 'swish'
#-------------------------------------------------------------------
                            # DENSE 6
units_6 = 512
activation_6 = 'swish'
dropout_6 = 0.5
#-------------------------------------------------------------------
                            # DENSE 7
units_7 = 256
activation_7 = 'swish'
dropout_7 = 0.5
#-------------------------------------------------------------------
                            # COMPILE
compile_optimizer = Adam(lr=0.001)
#-------------------------------------------------------------------
if K.image_data_format() == 'channels_first':
    input_shape = (3, img_cols, img_rows)
    chanDim = 1
else:
    input_shape = (img_cols, img_rows, 3)
    chanDim = -1
#-------------------------------------------------------------------
                              # CNN
model = Sequential()
#-------------------------------------------------------------------
                            # CONV2D 1
model.add(Conv2D(nb_filters_1,
                    (kernel_size_1[0], kernel_size_1[1]),
                    strides=(stride_size_1[0], stride_size_1[1]),
                    padding='same', activation=activation_1,
                    input_shape=input_shape))
model.add(MaxPooling2D(pool_size=pool_size_1))
#-------------------------------------------------------------------
                            # CONV2D 2
model.add(Conv2D(nb_filters_2,
                    (kernel_size_2[0], kernel_size_2[1]),
                    strides=(stride_size_2[0], stride_size_2[1]),
                    padding='same', activation=activation_2))
model.add(MaxPooling2D(pool_size=pool_size_2))
#-------------------------------------------------------------------
                            # CONV2D 3
model.add(Conv2D(nb_filters_3,
                    (kernel_size_3[0], kernel_size_3[1]),
                    strides=(stride_size_3[0], stride_size_3[1]),
                    padding='same', activation=activation_3))
model.add(MaxPooling2D(pool_size=pool_size_3))
#-------------------------------------------------------------------
                            # CONV2D 4
model.add(Conv2D(nb_filters_4,
                    (kernel_size_4[0], kernel_size_4[1]),
                    strides=(stride_size_4[0], stride_size_4[1]),
                    padding='same', activation=activation_4))
model.add(MaxPooling2D(pool_size=pool_size_4))
#-------------------------------------------------------------------
                            # CONV2D 5
model.add(Conv2D(nb_filters_5,
                    (kernel_size_5[0], kernel_size_5[1]),
                    strides=(stride_size_5[0], stride_size_5[1]),
                    padding='same', activation=activation_5))
model.add(MaxPooling2D(pool_size=pool_size_5))
#-------------------------------------------------------------------
                            # FLATTEN
model.add(Flatten())
#-------------------------------------------------------------------
                            # DENSE 6
model.add(Dense(units_6, activation=activation_6))
model.add(Dropout(dropout_6))
#-------------------------------------------------------------------
                            # DENSE 7
model.add(Dense(units_7, activation=activation_7))
model.add(Dropout(dropout_7))
#-------------------------------------------------------------------
                            # DENSE 8
model.add(Dense(nb_classes, activation='sigmoid'))
#-------------------------------------------------------------------
                            # METRICS
METRICS = [ metrics.BinaryAccuracy(name='ACCURACY'),
            metrics.Precision(name='PRECISION'),
            metrics.Recall(name='RECALL'),
            metrics.AUC(name='AUC'),
            metrics.TruePositives(name='TP'),
            metrics.TrueNegatives(name='TN'),
            metrics.FalsePositives(name='FP'),
            metrics.FalseNegatives(name='FN')]
#-------------------------------------------------------------------
                            # COMPILE
model.compile(loss='binary_crossentropy',
                optimizer=compile_optimizer,
                metrics=METRICS)
#-------------------------------------------------------------------
                        # DATA GENERATORS
train_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
    # validation_split=validation_split
    )
val_datagen = ImageDataGenerator(
    # rescale=1. / 255,
    # shear_range=0.2,
    # zoom_range=0.2,
    # horizontal_flip=True
    )
test_datagen_norm = ImageDataGenerator()
test_datagen_pneum = ImageDataGenerator()
#-------------------------------------------------------------------
                          # GENERATORS
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_cols, img_rows),
    # color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=True
    # subset='training',
    )
validation_generator = val_datagen.flow_from_directory(
    val_data_dir,
    target_size=(img_cols, img_rows),
    # color_mode='grayscale',
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False
    # subset='validation',
    )
test_generator_norm = test_datagen_norm.flow_from_directory(
    test_data_dir_norm,
    target_size=(img_cols, img_rows),
    # color_mode='grayscale',
    batch_size=batch_size,
    shuffle=False,
    class_mode='binary'
    )
test_generator_pneum = test_datagen_pneum.flow_from_directory(
    test_data_dir_pneum,
    target_size=(img_cols, img_rows),
    # color_mode='grayscale',
    batch_size=batch_size,
    shuffle=False,
    class_mode='binary'
    )
#-------------------------------------------------------------------
                          # SUMMARY
model.summary()
#-------------------------------------------------------------------
                            # FIT
history = model.fit(
    train_generator,
    steps_per_epoch= train_generator.samples // batch_size,
    epochs=nb_epoch,
    validation_data= validation_generator,
    validation_steps= validation_generator.samples // batch_size,
    class_weight=class_weights
    # callbacks=[tensorboard_callback]
    )
#-------------------------------------------------------------------
label_map = (train_generator.class_indices)
print(label_map)
#-------------------------------------------------------------------


from sklearn.metrics import classification_report, roc_curve, auc, confusion_matrix, precision_score, recall_score, accuracy_score, roc_auc_score, plot_confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')

validation_generator.reset()
y_pred = model.predict(validation_generator).ravel()
fpr, tpr, thresholdss = roc_curve(validation_generator.classes, y_pred)
auc_ = auc(fpr, tpr)
#ROC---------------------------------------------
fig, ax = plt.subplots(1,2, figsize=(10, 5))
ax[0].plot([0, 1], [0, 1], 'k--')
ax[0].plot(fpr, tpr, label='area = {:.3f}'.format(auc_))
ax[0].set_xlabel('False Positive Rate')
ax[0].set_ylabel('True Positive Rate')
ax[0].set_title('ROC')
ax[0].legend(loc='best')
#AUC---------------------------------------------
ax[1].plot(history.history['AUC'])
ax[1].plot(history.history['val_AUC'])
ax[1].set_title('AUC')
ax[1].set_ylabel('AUC')
ax[1].set_xlabel('Epoch')
ax[1].legend(['train', 'val'], loc='best')
plt.tight_layout()


validation_generator.reset()
Y_pred = model.predict(validation_generator)
y_pred = np.argmax(Y_pred, axis=1)
#CONFUSION MATRIX--------------------------------
cm = confusion_matrix(validation_generator.classes, y_pred)
plt.matshow(cm, cmap='Blues')
plt.title('Confusion matrix')
plt.colorbar()
plt.show()
#ACCURACY---------------------------------------
fig, ax = plt.subplots(2, 2, figsize=(12,6))
ax[0][0].plot(history.history['ACCURACY'])
ax[0][0].plot(history.history['val_ACCURACY'])
ax[0][0].set_title('Model Accuracy')
ax[0][0].set_ylabel('Accuracy')
ax[0][0].set_xlabel('Epoch')
ax[0][0].legend(['train', 'val'], loc='best')
#LOSS-------------------------------------------
ax[0][1].plot(history.history['loss'])
ax[0][1].plot(history.history['val_loss'])
ax[0][1].set_title('Model Loss')
ax[0][1].set_ylabel('Loss')
ax[0][1].set_xlabel('Epoch')
ax[0][1].legend(['train', 'val'], loc='best')
#PRECISION--------------------------------------
ax[1][0].plot(history.history['PRECISION'])
ax[1][0].plot(history.history['val_PRECISION'])
ax[1][0].set_title('Model Precision')
ax[1][0].set_ylabel('Precision')
ax[1][0].set_xlabel('Epoch')
ax[1][0].legend(['train', 'val'], loc='best')
#RECALL----------------------------------------
ax[1][1].plot(history.history['RECALL'])
ax[1][1].plot(history.history['val_RECALL'])
ax[1][1].set_title('Model Recall')
ax[1][1].set_ylabel('Recall')
ax[1][1].set_xlabel('Epoch')
ax[1][1].legend(['train', 'val'], loc='best')
plt.tight_layout()


# NORMAL Predictions
yhat_probs = model.predict(test_generator_norm)
yhat_classes = (model.predict_classes(test_generator_norm) > 0.5).astype('int32')
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
for prob, cl in zip(yhat_probs, yhat_classes):
  print(f'{prob:0.3f} | {cl}')

# PNEUMONIA Predictions
yhat_probs = model.predict(test_generator_pneum)
yhat_classes = (model.predict_classes(test_generator_pneum) > 0.5).astype('int32')
yhat_probs = yhat_probs[:, 0]
yhat_classes = yhat_classes[:, 0]
for prob, cl in zip(yhat_probs, yhat_classes):
  print(f'{prob:0.3f} | {cl}')