from data_analyze import *
import os, random, cv2, time, keras, datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Adamax
from keras import regularizers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

# Disable all logging output from Tensorflow 
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed

# TODO
# refference
# https://www.kaggle.com/code/matthewjansen/transfer-learning-brain-tumor-classification
# Build augmentation layer
# augmentation_layer = Sequential([
#     layers.RandomFlip(mode='horizontal_and_vertical', seed=CFG.TF_SEED),
#     layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=CFG.TF_SEED),
# ], name='augmentation_layer')


def get_model(network_name, shape, num_classes):
    if network_name == 'CNN':
        model = Sequential([
            Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=shape),
            MaxPooling2D(pool_size=2),
            Conv2D(filters=32, kernel_size=3, activation='relu'),
            MaxPooling2D(pool_size=2),
            Flatten(),
            Dense(64, activation='relu'),
            Dense(num_classes, activation='softmax')
        ])
    elif network_name == 'VGG16':
        model = Sequential([
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=shape),
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(num_classes, activation='sigmoid')
        ])
    elif network_name == 'VGG19':
        model = tf.keras.Sequential([
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=shape),
            Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
            MaxPooling2D(pool_size=2, padding='same'),
            Flatten(),
            Dense(4096, activation='relu'),
            Dense(4096, activation='relu'),
            Dense(num_classes, activation='sigmoid')
        ])
    else:
        model = None
    return model


def NN(df: pd.DataFrame, network_name='CNN', epochs=20, batch_size=16, earlystop=True, 
       model_summary=False, logging=False, save=False, graphs=False):
    start_time = time.time()

    train_df, test_df = train_test_split(df, test_size=0.33, shuffle=True, random_state=123, stratify=df['class'])
    test_df, val_df = train_test_split(test_df, test_size=0.5, shuffle=True, random_state=123, stratify=test_df['class'])

    #   df:
    #                                               image_path           class
    # 0          ./archive/Astrocitoma T1/005_big_gallery.jpeg  Astrocitoma T1
    # 1          ./archive/Astrocitoma T1/006_big_gallery.jpeg  Astrocitoma T1
    

    # Initialize parameters
    num_classes = df['class'].unique().size
    channels = 3
    img_size = (224, 224)
    img_shape = (img_size[0], img_size[1], channels)
    df_labels = {
        'Astrocitoma T1': 0,
        'Astrocitoma T1C+': 1 ,
        'Astrocitoma T2': 2,
        'Carcinoma T1': 4,
        'Carcinoma T1C+': 5,
        'Carcinoma T2': 6,
        'Ependimoma T1': 7,
        'Ependimoma T1C+': 8,
        'Ependimoma T2': 9,
        'Ganglioglioma T1': 10,
        'Ganglioglioma T1C+': 11,
        'Ganglioglioma T2': 12,
        'Germinoma T1': 13,
        'Germinoma T1C+': 14,
        'Germinoma T2': 15,
        'Glioblastoma T1': 16,
        'Glioblastoma T1C+': 17,
        'Glioblastoma T2': 18,
        'Granuloma T1': 19,
        'Granuloma T1C+': 20,
        'Granuloma T2': 21,
        'Meduloblastoma T1': 22,
        'Meduloblastoma T1C+': 23,
        'Meduloblastoma T2': 24,
        'Meningioma T1': 25,
        'Meningioma T1C+': 26,
        'Meningioma T2': 27,
        'Neurocitoma T1': 28,
        'Neurocitoma T1C+': 29,
        'Neurocitoma T2': 30,
        'Oligodendroglioma T1': 31,
        'Oligodendroglioma T1C+': 32,
        'Oligodendroglioma T2': 33,
        'Papiloma T1': 34,
        'Papiloma T1C+': 35,
        'Papiloma T2': 36,
        'Schwannoma T1': 37,
        'Schwannoma T1C+': 38,
        'Schwannoma T2': 39,
        'Tuberculoma T1': 40,
        'Tuberculoma T1C+': 41,
        'Tuberculoma T2': 42,
        '_NORMAL T1': 43,
        '_NORMAL T2': 44,
    }

    # Get images, put in X,y variables 
    # region
    def get_xy(df):
        '''Get from df images, put in X,y'''
        X, y = [], []
        for ind in df.index:
            img = cv2.imread(str(df['image_path'][ind]))
            resized_img = cv2.resize(img, img_size)
            X.append(resized_img) 
            y.append(df_labels[df['class'][ind]])

        X = np.array(X)
        y = np.array(y)
        X = X/255
        return X, y

    X_train, y_train = get_xy(train_df)
    X_test, y_test = get_xy(test_df)
    X_val, y_val = get_xy(val_df)

    y_train = pd.get_dummies(y_train)
    y_test = pd.get_dummies(y_test)
    y_val = pd.get_dummies(y_val)
    # endregion

    # Initialize model
    model = get_model(network_name, img_shape, num_classes)
    if model == None:
        print('=== Incorrect network name ===')
        return
    if model_summary:
        model.summary()
    model.compile(optimizer="sgd", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc']) 
    if save:
        model.save(f'./models/{network_name}.keras')


    # Network training
    # Callback to prevent learning if loss of function stops improving 2 steps
    if earlystop:
        early_stop = EarlyStopping(monitor='loss',patience=2)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=logging, validation_data=(X_val, y_val), callbacks=[early_stop])
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=logging, validation_data=(X_val, y_val))
       
    
    # Network evaluation on a test set + summing up
    loss, accuracy = model.evaluate(X_test, y_test)

    if graphs:
        h1 = history.history
        acc_epochs = pd.DataFrame({'train': h1['acc'], 'val': h1['val_acc']})
        loss_epochs = pd.DataFrame({'train': h1['loss'], 'val': h1['val_loss']})

        fig = px.line(acc_epochs, x=acc_epochs.index, y=acc_epochs.columns[0::], title='Training and Evaluation Accuracy every Epoch', markers=True)
        fig.show()
        fig = px.line(loss_epochs, x=loss_epochs.index, y=loss_epochs.columns[0::], title='Training and Evaluation Loss every Epoch', markers=True)
        fig.show()

    end_time = time.time()
    training_time = end_time - start_time
    return model, [loss, accuracy, training_time]
