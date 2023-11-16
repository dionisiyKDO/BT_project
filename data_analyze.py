import plotly.express as px
import pandas as pd
import os

def get_df(path: str) -> pd.DataFrame:
    """Get a Dataframe with paths to images and corresponding classes"""
    if not os.path.exists(path):
         print("=== Incorrect path to Dataset ===")
         exit()

    image_paths = []
    image_class = []
    for class_name in os.listdir(path):
        for image_name in os.listdir(f'{path}/{class_name}'):
            path_to_image = f'{path}/{class_name}/{image_name}'
            image_paths.append(path_to_image)
            image_class.append(class_name)
    df = pd.DataFrame({'image_path': image_paths, 'class': image_class})
    
    return df