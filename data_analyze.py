import pandas as pd
import numpy as np
import os, cv2
import matplotlib.image as mpimg
import plotly.express as px
from PIL import Image

def count_labels_plot(x: pd.Series, title: str, xlabel: str, ylabel: str):
    fig = px.bar(x=x.index, y=x, title=title)
    fig.update_layout(xaxis_title = xlabel, yaxis_title = ylabel)
    fig.show()


def get_df(path: str) -> pd.DataFrame:
    """Получить Датафрейм c путями к картинкам и соответсвующим классами"""
    if not os.path.exists(path):
         print("=== Неправильный путь к Датасету ===")
         exit()


    # class_name - _NORMAL T2
    # image_name - fda32d9ec9caf6acbe1bdd6c5c71cc_big_gallery.jpeg
    image_paths = []
    image_class = []
    for class_name in os.listdir(path):
        for image_name in os.listdir(f'{path}/{class_name}'):
            path_to_image = f'{path}/{class_name}/{image_name}'
            image_paths.append(path_to_image)
            image_class.append(class_name)
    df = pd.DataFrame({'image_path': image_paths, 'class': image_class})
    
    #   df:
    #                                               image_path           class
    # 0          ./archive/Astrocitoma T1/005_big_gallery.jpeg  Astrocitoma T1
    # 1          ./archive/Astrocitoma T1/006_big_gallery.jpeg  Astrocitoma T1
    return df

def get_images_avg_width_height(df: pd.DataFrame):
    avg_width, avg_height = 0,0
    
    for i in df.index:
        try:
            img_path = df['image_path'].loc[i]
            img = cv2.imread(img_path)
            avg_width += img.shape[0]
            avg_height += img.shape[1]
        except:
            pass

    avg_width  = avg_width  // len(df)
    avg_height = avg_height // len(df)
    
    return avg_width, avg_height

if __name__ == '__main__':
    path = './archive'
    df = get_df(path)

    print(f"{df.shape[0]} images with {len(df['class'].unique())} classes")
    
    data_avg_width, data_avg_height = get_images_avg_width_height(df)
    print(f"Average width and height for the dataset is {data_avg_width}x{data_avg_height} with aspect_ratio {data_avg_width/data_avg_height}")


    # print('\ndf.info(): \n',  df.info())
    # print('\ndf.head(): \n',  df.head())
    # print('\ndf.describe(): \n',  df.describe().T)
    
    # count_labels_plot(df['class'].value_counts(), "Labels distribution", "Label", 'Frequency')

