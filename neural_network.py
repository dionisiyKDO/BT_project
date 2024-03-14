from data_analyze import *
import os, random, cv2, time, keras, datetime

import numpy as np
import pandas as pd
import tensorflow as tf
import plotly.express as px
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix, classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.optimizers import Adam, Adamax, SGD
from keras import regularizers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.layers import BatchNormalization, MaxPooling2D, GlobalAveragePooling2D

# Allow dynamically allocate memory on the GPU
physical_device = tf.config.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_device[0], True)

# Disable all logging output from Tensorflow 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
    # 0 = all messages are logged (default behavior)
    # 1 = INFO messages are not printed
    # 2 = INFO and WARNING messages are not printed
    # 3 = INFO, WARNING, and ERROR messages are not printed


def get_model(network_name, shape, num_classes):
    if network_name == 'test':
        model = Sequential([
            Conv2D(filters=256, kernel_size=(12,12), strides=(5,5), activation='relu', input_shape=shape),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Conv2D(filters=512, kernel_size=(6,6), strides=(2,2), activation='relu', padding="same"),
            Conv2D(filters=256, kernel_size=(3,3), strides=(1,1), activation='relu', padding="same"),
            MaxPooling2D(pool_size=(3,3), strides=(2,2)),
            BatchNormalization(),

            Flatten(),
            Dense(4096, activation='relu'),
            Dropout(0.5),
            Dense(4096, activation='relu'),
            Dropout(0.5),
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
    elif network_name == 'OwnV1':
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
            Dense(num_classes,activation='softmax')
        ])
    else:
        model = None
    return model




def NN(network_name='CNN', epochs=20, batch_size=16, earlystop=True, 
       model_summary=False, logging=False, save=False, graphs=False):
    start_time = time.time()

    # Initialize parameters
    # region
    train_dir = './data/Training/'
    test_dir  = './data/Testing/'
    seed = 1
    channels = 3
    img_size = (150, 150)
    img_shape = (img_size[0], img_size[1], channels)
    class_labels = {
        'notumor': 0,
        'pituitary': 1,
        'glioma': 2,
        'meningioma': 3,
    }
    num_classes = len(class_labels.keys())
    # endregion

    # Getting data
    # region
    # train_paths, train_labels = get_data(train_dir)
    # test_paths, test_labels = get_data(test_dir)
    
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
    
    model.compile(Adam(learning_rate= 0.001, beta_1=0.9, beta_2=0.999), loss= 'categorical_crossentropy', metrics=['acc'])
    # model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics=['acc'])
    # model.compile(SGD(learning_rate=0.001), loss='categorical_crossentropy', metrics= ['acc'])

    # if save:
    #     model.save(f'./models/{network_name}.keras')
    # endregion

    # Network training
    # region
    model_cp = ModelCheckpoint(
        filepath='./checkpoints/model.epoch{epoch:02d}-val_acc{val_acc:.4f}.hdf5',
        monitor='val_acc',
        save_freq='epoch',
        verbose=1,
        save_best_only=True,
        save_weights_only=True)
    model_es = EarlyStopping(monitor='loss', min_delta=1e-9, patience=8, verbose=True)
    model_rlr = ReduceLROnPlateau(monitor='val_loss', factor=0.3, patience=5, verbose=True)
    
    history = model.fit(train_generator,
                        steps_per_epoch=steps_per_epoch,
                        epochs=epochs,
                        validation_data=test_generator,
                        validation_steps=validation_steps,
                        verbose=logging, 
                        callbacks=[model_es, model_rlr, model_cp])
    # endregion

    # Network evaluation + results of trining
    # region
    loss, accuracy = model.evaluate(test_generator, steps=test_generator.samples//batch_size)

    if graphs:
        h1 = history.history
        acc_epochs = pd.DataFrame({'train': h1['acc'], 'val': h1['val_acc']})
        loss_epochs = pd.DataFrame({'train': h1['loss'], 'val': h1['val_loss']})

        fig = px.line(acc_epochs, x=acc_epochs.index, y=acc_epochs.columns[0::], title=f'Training and Evaluation Accuracy every Epoch for "{network_name}"', markers=True)
        fig.show()
        fig = px.line(loss_epochs, x=loss_epochs.index, y=loss_epochs.columns[0::], title='Training and Evaluation Loss every Epoch', markers=True)
        fig.show()
    # endregion

    end_time = time.time()
    training_time = end_time - start_time
    return model, [loss, accuracy, training_time]
