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

# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

# config = tf.compat.v1.ConfigProto()
# config.gpu_options.allow_growth = True
# sess = tf.compat.v1.Session(config=config)

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




def plot_training(hist):
    '''
    This function take training model and plot history of accuracy and losses with the best epoch in both of them.
    '''

    tr_acc = hist.history['accuracy']
    tr_loss = hist.history['loss']
    val_acc = hist.history['val_accuracy']
    val_loss = hist.history['val_loss']
    index_loss = np.argmin(val_loss)
    val_lowest = val_loss[index_loss]
    index_acc = np.argmax(val_acc)
    acc_highest = val_acc[index_acc]
    Epochs = [i+1 for i in range(len(tr_acc))]
    loss_label = f'best epoch= {str(index_loss + 1)}'
    acc_label = f'best epoch= {str(index_acc + 1)}'

    plt.figure(figsize= (20, 8))
    plt.style.use('fivethirtyeight')

    plt.subplot(1, 2, 1)
    plt.plot(Epochs, tr_loss, 'r', label= 'Training loss')
    plt.plot(Epochs, val_loss, 'g', label= 'Validation loss')
    plt.scatter(index_loss + 1, val_lowest, s= 150, c= 'blue', label= loss_label)
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(Epochs, tr_acc, 'r', label= 'Training Accuracy')
    plt.plot(Epochs, val_acc, 'g', label= 'Validation Accuracy')
    plt.scatter(index_acc + 1 , acc_highest, s= 150, c= 'blue', label= acc_label)
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.tight_layout
    plt.show()


def NN(df: pd.DataFrame, network_name='CNN', epochs=20, batch_size=16, earlystop=True, 
       model_summary=False, logging=False, save=False, graphs=False):
    start_time = time.time()
    results = []

    # Разделение выборки на тестовую и тренировочную 
    train_df, test_df = train_test_split(df, test_size=0.33, shuffle=True, random_state=123, stratify=df['class'])
    test_df, val_df = train_test_split(test_df, test_size=0.5, shuffle=True, random_state=123, stratify=test_df['class'])
    # Колво ключевых признаков, тест stratify
    # region
    # print(valid_df.head())
    # print(train_df.head())
    # print(test_df.head())
    # print(valid_df['class'].value_counts())
    # print(train_df['class'].value_counts())
    # print(test_df['class'].value_counts())
    # count_labels_plot(valid_df['class'].value_counts(), "Valid_df Labels distribution", "Label", 'Frequency')
    # count_labels_plot(train_df['class'].value_counts(), "Train_df Labels distribution", "Label", 'Frequency')
    # count_labels_plot( test_df['class'].value_counts(), "Test_df Labels distribution",  "Label", 'Frequency')
    # endregion

    #   df:
    #                                               image_path           class
    # 0          ./archive/Astrocitoma T1/005_big_gallery.jpeg  Astrocitoma T1
    # 1          ./archive/Astrocitoma T1/006_big_gallery.jpeg  Astrocitoma T1
    

    # Инициализация доп. параметров
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

    # Transform data 
    # region
    def get_xy(df):
        '''Get from df images, put in X, y'''
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

    # Инициализация сети
    model = get_model(network_name, img_shape, num_classes)

    # Проверка на то, создалась ли сеть
    if model == None:
        print('=== Неправильное имя сети ===')
        return

    # "Cводка" по сети
    if model_summary:
        model.summary()
    
    # Компиляция сети
    model.compile(optimizer="sgd",
                  loss=tf.keras.losses.CategoricalCrossentropy(),
                  metrics=['acc']) 

    # Обучение сети
    # Колбэк на предотвращение обучения если потери функции перестали улучшаться 2 шага
    if earlystop:
        early_stop = EarlyStopping(monitor='loss',patience=2)
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=logging, validation_data=(X_val, y_val), callbacks=[early_stop])
    else:
        history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=logging, validation_data=(X_val, y_val))
       
    
    # Сохранить модель если надо
    if save:
        model.save(f'{network_name}.keras')

    # Оценивание сети на тестовом наборе
    loss, accuracy = model.evaluate(X_test, y_test)

    # y_pred = model.predict(X_test, batch_size=16, verbose=1)

    if graphs:
        h1 = history.history
        acc_epochs = pd.DataFrame({'train': h1['acc'], 'val': h1['val_acc']})
        loss_epochs = pd.DataFrame({'train': h1['loss'], 'val': h1['val_loss']})

        fig = px.line(acc_epochs, x=acc_epochs.index, y=acc_epochs.columns[0::], title='Training and Evaluation Accuracy every Epoch', markers=True)
        fig.show()
        fig = px.line(loss_epochs, x=loss_epochs.index, y=loss_epochs.columns[0::], title='Training and Evaluation Loss every Epoch', markers=True)
        fig.show()

    # Подсчёт потраченного времени и возврат результатов 
    end_time = time.time()
    training_time = end_time - start_time
    return model, [loss, accuracy, training_time]


if __name__ == '__main__':
    path = './archive'
    df = get_df(path = path)
    NN(df)
