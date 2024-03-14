from data_analyze import *
import os, random, cv2, time, keras, datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn import experimental

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Adamax, SGD
from keras import regularizers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout
from keras.layers import BatchNormalization, Lambda, Resizing, MaxPooling2D, GlobalAveragePooling2D

physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

# Disable all logging output from Tensorflow 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed

# Build augmentation layer
# augmentation_layer = Sequential([
#     layers.RandomFlip(mode='horizontal_and_vertical', seed=CFG.TF_SEED),
#     layers.RandomZoom(height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=CFG.TF_SEED),
# ], name='augmentation_layer')
# Resizing(224, 224, interpolation="bilinear", input_shape=shape),
# Lambda(tf.nn.local_response_normalization),

def get_model(network_name, shape, num_classes):
    if network_name == 'CNN':
        model = Sequential()

        # Convolutional layer 1
        model.add(Conv2D(64,(7,7), input_shape=shape, padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        #Convolutional layer 2
        model.add(Conv2D(128,(7,7), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Convolutional layer 3
        model.add(Conv2D(128,(7,7), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Convolutional layer 4
        model.add(Conv2D(256,(7,7), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Convolutional layer 5
        model.add(Conv2D(256,(7,7), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        # Convolutional layer 6
        model.add(Conv2D(512,(7,7), padding='same', activation='relu'))
        model.add(BatchNormalization())
        model.add(MaxPooling2D(pool_size=(2,2)))

        model.add(Flatten())

        # Full connect layers

        model.add(Dense(units= 1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=1024, activation='relu'))
        model.add(Dropout(0.3))
        model.add(Dense(units=num_classes, activation='softmax'))
        
        # model = Sequential([
        #     Conv2D(filters=16, kernel_size=3, activation='relu', padding='same', input_shape=shape),
        #     MaxPooling2D(pool_size=(3,3), strides=(1,1)),
        #     BatchNormalization(),
            
        #     Conv2D(filters=32, kernel_size=3, activation='relu'),
        #     MaxPooling2D(pool_size=(3,3), strides=(1,1)),
        #     # BatchNormalization(),
            
        #     Flatten(),
        #     Dense(64, activation='relu'),
        #     Dropout(0.5),
        #     Dense(32, activation='relu'),
        #     Dropout(0.5),
        #     Dense(num_classes, activation='softmax')
        # ])
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
    elif network_name == 'AlexNet':
        model = tf.keras.Sequential([
            Conv2D(filters=96, kernel_size=(11,11), strides=(4,4), activation='relu', input_shape=shape),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(5,5), strides=(1,1), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            Conv2D(filters=384, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Flatten(),
            Dense(4096,activation='relu'),
            Dropout(0.5),
            Dense(4096,activation='relu'),
            Dropout(0.5),
            Dense(num_classes,activation='softmax')  
        ])
    
    elif network_name == 'InceptionV3':
        base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, input_shape=shape)
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ])
    elif network_name == 'EfficientNetV2':
        base_model = tf.keras.applications.EfficientNetV2B0(weights='imagenet', include_top=False, input_shape=shape)
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ])
    elif network_name == 'ResNet50':
        base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False, input_shape=shape)
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ])
    elif network_name == 'InceptionResNetV2':
        base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(weights='imagenet', include_top=False, input_shape=shape)
        base_model.trainable = False
        model = tf.keras.Sequential([
            base_model,
            GlobalAveragePooling2D(),
            Dropout(0.5),
            Dense(num_classes, activation='softmax'),
        ])
    
    elif network_name == 'AlexNet_own':
        model = tf.keras.Sequential([
            Conv2D(filters=256, kernel_size=(12,12), strides=(5,5), activation='relu', input_shape=shape),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(6,6), strides=(2,2), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(num_classes,activation='softmax')
        ])
    elif network_name == 'test':
        model = tf.keras.Sequential([
            Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), padding="same", activation='relu', input_shape=shape),
            MaxPooling2D(),
            Conv2D(filters=64, kernel_size=(3,3), padding="same", activation="relu"),
            MaxPooling2D(),
            Flatten(),
            Dense(64, activation = "relu"),
            Dense(32, activation = "relu"),
            Dense(num_classes, activation='softmax')
        ])

    else:
        model = None
    return model




def NN(network_name='CNN', epochs=20, batch_size=16, earlystop=True, 
       model_summary=False, logging=False, save=False, graphs=False):
    start_time = time.time()

    # Initialize parameters
    train_dir = './data_2/Training/'
    test_dir  = './data_2/Testing/'
    # train_dir = './.archive/archive (3)/Training/'
    # test_dir  = './.archive/archive (3)/Testing/'

    seed = 1
    channels = 3
    img_size = (150, 150)
    # img_size = (224, 224)
    img_shape = (img_size[0], img_size[1], channels)
    class_labels = {
        'notumor': 0,
        'pituitary': 1,
        'glioma': 2,
        'meningioma': 3,
    }
    num_classes = len(class_labels.keys())

    # getting data
    # region
    train_paths, train_labels = get_data(train_dir)
    test_paths, test_labels = get_data(test_dir)
    
    # Augmentation and preprocessing of training data
    train_datagen = ImageDataGenerator(rescale=1./255,
                                       rotation_range=10,
                                       brightness_range=(0.85, 1.15),
                                       width_shift_range=0.002,
                                       height_shift_range=0.002,
                                       shear_range=12.5,
                                       zoom_range=0,
                                       horizontal_flip=True,
                                       vertical_flip=False,
                                       fill_mode="nearest"
                                       )
    train_generator = train_datagen.flow_from_directory(train_dir,
                                                        target_size=img_size,
                                                        batch_size=batch_size,
                                                        class_mode="categorical",
                                                        seed=seed
                                                        )

    # Augmentation and preprocessing of training data
    # Just rescaling
    test_datagen = ImageDataGenerator(rescale=1./255)
    test_generator = test_datagen.flow_from_directory(test_dir,
                                                      target_size=img_size,
                                                      batch_size=batch_size,
                                                      class_mode="categorical",
                                                      shuffle=False,
                                                      seed=seed)

    steps_per_epoch = train_generator.samples // batch_size
    validation_steps = test_generator.samples // batch_size
    # endregion
    

    # Initialize model
    # region
    model = get_model(network_name, img_shape, num_classes)
    if model == None:
        print('=== Incorrect network name ===')
        return
    if model_summary:
        model.summary()
    
    # model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics=['acc'])
    # model.compile(optimizer="sgd", loss=tf.keras.losses.CategoricalCrossentropy(), metrics=['acc']) 
    model.compile(SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics= ['acc'])
    
    if save:
        model.save(f'./models/{network_name}.keras')
    # endregion

    # Network training
    # Callback to prevent learning if loss of function stops improving 2 steps
    cp_callback = ModelCheckpoint(
        filepath='./checkpoints/model.epoch{epoch:02d}-val_acc{val_acc:.4f}.hdf5',
        monitor='val_acc',
        save_freq='epoch',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    # rlr_callback = ReduceLROnPlateau(monitor = 'val_acc', factor = 0.2, patience = 5, verbose = 1)

    if earlystop:
        early_stop = EarlyStopping(monitor='loss',patience=2)
        history = model.fit(train_generator, epochs=epochs, 
                            batch_size=batch_size, verbose=logging, 
                            validation_data=test_generator, callbacks=[early_stop, cp_callback])
    else:
        history = model.fit(train_generator, epochs=epochs, 
                            batch_size=batch_size, verbose=logging, 
                            validation_data=test_generator, callbacks=[cp_callback])

        
    # Network evaluation on a test set + summing up
    loss, accuracy = model.evaluate(test_generator)

    if graphs:
        h1 = history.history
        acc_epochs = pd.DataFrame({'train': h1['acc'], 'val': h1['val_acc']})
        loss_epochs = pd.DataFrame({'train': h1['loss'], 'val': h1['val_loss']})

        # print(f'\nHighest val accuracy{max(acc_epochs.columns[0::])}')

        fig = px.line(acc_epochs, x=acc_epochs.index, y=acc_epochs.columns[0::], title=f'Training and Evaluation Accuracy every Epoch for "{network_name}"', markers=True)
        fig.show()
        fig = px.line(loss_epochs, x=loss_epochs.index, y=loss_epochs.columns[0::], title='Training and Evaluation Loss every Epoch', markers=True)
        fig.show()

    end_time = time.time()
    training_time = end_time - start_time
    return model, [loss, accuracy, training_time]
