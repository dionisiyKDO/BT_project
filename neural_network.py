from data_analyze import *
import os, random, cv2, time, keras

import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
import plotly.express as px
import matplotlib.pyplot as plt

from plotly.offline import iplot, init_notebook_mode

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report

from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import Adam, Adamax
from keras import regularizers

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Activation, Dropout, BatchNormalization

from keras.applications import VGG19

# physical_device = tf.config.list_physical_devices('GPU')
# tf.config.experimental.set_memory_growth(physical_device[0], True)



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


def NN(df: pd.DataFrame):
    start_time = time.time()
    results = []

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


    channels = 3
    img_size = (224, 224)
    img_shape = (img_size[0], img_size[1], channels)

    def get_xy(df):
        X, y = [], [] # X = images, y = labels
        for ind in df.index:
            img = cv2.imread(str(df['image_path'][ind]))
            resized_img = cv2.resize(img, img_size) # Resizing the images to be able to pass on MobileNetv2 model
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


    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same', input_shape=img_shape),
        tf.keras.layers.Conv2D(filters=64, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=3, activation='relu', padding='same'),
        tf.keras.layers.MaxPooling2D(pool_size=2, padding='same'),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(4096, activation='relu'),
        tf.keras.layers.Dense(44, activation='sigmoid')
    ])

    model.summary()

    model.compile( optimizer="adam",
                    loss=tf.keras.losses.CategoricalCrossentropy(),
                    metrics=['acc']) 

    history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val), batch_size=64)
    model.evaluate(X_test,y_test)


    y_pred = model.predict(X_test, batch_size=64, verbose=1)



    acc = pd.DataFrame({'train': history.history['acc'], 'val': history.history['val_acc']})

    fig = px.line(acc, x=acc.index, y=acc.columns[0::], title='Training and Evaluation Accuracy every Epoch', markers=True)
    fig.show()

    loss = pd.DataFrame({'train': history.history['loss'], 'val': history.history['val_loss']})

    fig = px.line(loss, x=loss.index, y=loss.columns[0::], title='Training and Evaluation Loss every Epoch', markers=True)
    fig.show()



    model.save('VGG16.keras')
    # model = tf.keras.models.load_model('my_model.keras')








    # BT region
    # region
    # color = 'rgb'
    # channels = 3

    # img_size = (224, 224)
    # img_shape = (img_size[0], img_size[1], channels)
    
    # batch_size = 32
    # ts_length = len(test_df)
    # test_batch_size = max([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80])
    # test_steps = ts_length // test_batch_size
    # def scalar(img): return img
    

    # tr_gen = ImageDataGenerator(preprocessing_function= scalar, 
    #                             horizontal_flip= True)

    # ts_gen = ImageDataGenerator(preprocessing_function= scalar)

    # train_gen = tr_gen.flow_from_dataframe( train_df, 
    #                                     x_col= 'image_path', 
    #                                     y_col= 'class', 
    #                                     target_size= img_size, 
    #                                     class_mode= 'categorical',
    #                                     color_mode= color, 
    #                                     shuffle= True, 
    #                                     batch_size= batch_size)

    # valid_gen = ts_gen.flow_from_dataframe( valid_df, 
    #                                     x_col= 'image_path', 
    #                                     y_col= 'class', 
    #                                     target_size= img_size, 
    #                                     class_mode= 'categorical',
    #                                     color_mode= color, 
    #                                     shuffle= True, 
    #                                     batch_size= batch_size)

    # test_gen = ts_gen.flow_from_dataframe( test_df, 
    #                                     x_col= 'image_path', 
    #                                     y_col= 'class', 
    #                                     target_size= img_size, 
    #                                     class_mode= 'categorical',
    #                                     color_mode= color, 
    #                                     shuffle= False, 
    #                                     batch_size= test_batch_size)


    # base_model = tf.keras.applications.efficientnet.EfficientNetB5(include_top= False, 
    #                                                            weights= "imagenet", 
    #                                                            input_shape= img_shape,
    #                                                            pooling= 'max')

    # model = Sequential([
    #     base_model,
    #     BatchNormalization(axis= -1, momentum= 0.99, epsilon= 0.001),
    #     Dense(256, 
    #         kernel_regularizer= regularizers.l2(l= 0.016), 
    #         activity_regularizer= regularizers.l1(0.006),
    #         bias_regularizer= regularizers.l1(0.006), 
    #         activation= 'relu'),
        
    #     Dropout(rate= 0.45, 
    #             seed= 123),
        
    #     Dense(44, activation= 'softmax')
    # ])

    # model.compile(Adamax(learning_rate= 0.001), loss= 'categorical_crossentropy', metrics= ['accuracy'])

    # model.summary()

    # early_stop = EarlyStopping(monitor='val_loss', 
    #                        patience=5,
    #                        verbose=1)

    # checkpoint = ModelCheckpoint('model_weights.h5', 
    #                             monitor='val_loss', 
    #                             save_best_only=True, 
    #                             save_weights_only=True, 
    #                             mode='min', 
    #                             verbose=1)


    # history = model.fit(x= train_gen,
    #                     steps_per_epoch = 20,
    #                     epochs= 20, 
    #                     callbacks=[early_stop, checkpoint],
    #                     validation_data= valid_gen)

    # plot_training(history)

    # ts_length = len(test_df)
    # test_batch_size = max(sorted([ts_length // n for n in range(1, ts_length + 1) if ts_length%n == 0 and ts_length/n <= 80]))
    # test_steps = ts_length // test_batch_size

    # train_score = model.evaluate(train_gen, steps= test_steps, verbose= 1)
    # valid_score = model.evaluate(valid_gen, steps= test_steps, verbose= 1)
    # test_score = model.evaluate(test_gen, steps= test_steps, verbose= 1)

    # print("Train Loss: ", train_score[0])
    # print("Train Accuracy: ", train_score[1])
    # print('-' * 20)
    # print("Validation Loss: ", valid_score[0])
    # print("Validation Accuracy: ", valid_score[1])
    # print('-' * 20)
    # print("Test Loss: ", test_score[0])
    # print("Test Accuracy: ", test_score[1])


    # y_pred = model.predict(test_gen)

    # y_pred_labels = np.argmax(y_pred, axis=1)

    # y_true_labels = test_gen.classes
    # class_names = list(test_gen.class_indices.keys())

    # confusion_mtx = confusion_matrix(y_true_labels, y_pred_labels)

    # plt.figure(figsize=(10,8))
    # sns.heatmap(confusion_mtx, cmap="Blues", annot=True, fmt="d", xticklabels=class_names, yticklabels=class_names)
    # plt.title("Confusion Matrix")
    # plt.xlabel("Predicted Label")
    # plt.ylabel("True Label")
    # plt.show()



    # report = classification_report(y_true_labels, y_pred_labels, target_names=class_names)

    # print("Classification Report: ")
    # print(report)

    # return results
    # endregion


if __name__ == '__main__':
    path = './archive'
    df = get_df(path = path)
    NN(df)

    # model = keras.Model.load_weights('./model_weights.h5')
    # plot_training(model.history)
