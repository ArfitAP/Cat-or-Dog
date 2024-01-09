from core.utils.BB_utils import resize_image_bb, resize_bb, create_bb_array
from core.utils.utils import filelist
import xml.etree.ElementTree as ET
from pathlib import Path
import pandas as pd
import numpy as np


def normalize(im):
    """Normalizes images with Imagenet stats."""
    imagenet_stats = np.array([[0.485, 0.456, 0.406], [0.229, 0.224, 0.225]])
    return (im - imagenet_stats[0])/imagenet_stats[1]


def generate_train_df(anno_path, images_path):
    annotations = filelist(anno_path, '.xml')
    anno_list = []
    for anno_path in annotations:
        root = ET.parse(anno_path).getroot()
        anno = {'filename': Path(str(images_path) + '/' + root.find("./filename").text),
                'width': root.find("./size/width").text, 'height': root.find("./size/height").text,
                'class': root.find("./object/name").text, 'xmin': int(root.find("./object/bndbox/xmin").text),
                'ymin': int(root.find("./object/bndbox/ymin").text),
                'xmax': int(root.find("./object/bndbox/xmax").text),
                'ymax': int(root.find("./object/bndbox/ymax").text)}
        anno_list.append(anno)
    return pd.DataFrame(anno_list)


def get_train_dataframe(anno_path, images_path):
    df_train = generate_train_df(anno_path, images_path)

    class_dict = {'cat': 0, 'dog': 1}
    df_train['class'] = df_train['class'].apply(lambda x: class_dict[x])

    # Populating Training DF with new paths and bounding boxes
    new_paths = []
    new_bbs = []
    train_path_resized = Path(str(images_path) + '_resized')
    for index, row in df_train.iterrows():
        new_path, new_bb = resize_image_bb(row['filename'], train_path_resized, create_bb_array(row.values), 300)
        new_paths.append(new_path)
        new_bbs.append(new_bb)
    df_train['new_path'] = new_paths
    df_train['new_bb'] = new_bbs

    return df_train


def get_validation_dataframe(anno_path, images_path):
    df_valid = generate_train_df(anno_path, images_path)

    class_dict = {'cat': 0, 'dog': 1}
    df_valid['class'] = df_valid['class'].apply(lambda x: class_dict[x])

    # Populating Validation DF with new paths and bounding boxes
    new_paths = []
    new_bbs = []
    train_path_resized = Path(str(images_path) + '_resized')
    for index, row in df_valid.iterrows():
        new_path, new_bb = resize_bb(row['filename'], train_path_resized, create_bb_array(row.values), 300)
        new_paths.append(new_path)
        new_bbs.append(new_bb)
    df_valid['new_path'] = new_paths
    df_valid['new_bb'] = new_bbs

    return df_valid


def generate_one_df(anno_path, image_path):

    root = ET.parse(anno_path).getroot()
    anno = {'filename': image_path,
            'width': root.find("./size/width").text, 'height': root.find("./size/height").text,
            'class': root.find("./object/name").text, 'xmin': int(root.find("./object/bndbox/xmin").text),
            'ymin': int(root.find("./object/bndbox/ymin").text),
            'xmax': int(root.find("./object/bndbox/xmax").text),
            'ymax': int(root.find("./object/bndbox/ymax").text)}
    return pd.DataFrame([anno])
