import sys

sys.path.append('..\\')
sys.path.append('core')

import cv2
from core.data import generate_one_df
from core.utils.BB_utils import create_mask, create_bb_array, mask_to_bb, show_corner_bb_subplot
import argparse

from core.utils.augmentation import transformsXY_FromArray
from core.utils.utils import read_image
import matplotlib.pyplot as plt


def show(args):

    images_path = 'D:/CatsVsDogsDataset/images'
    anno_path = 'D:/CatsVsDogsDataset/annotations'

    df_train = generate_one_df(anno_path + '/' + args.annotations, images_path + '/' + args.filename)
    sz = 300
    im = read_image(images_path + '/' + args.filename)
    bb = create_bb_array(df_train.iloc[0].values)
    im_resized = cv2.resize(im, (int(1.5 * sz), sz))
    Y_resized = cv2.resize(create_mask(bb, im), (int(1.5 * sz), sz))

    im_aug, bb_aug = transformsXY_FromArray(im_resized, Y_resized, True)

    fig, axs = plt.subplots(2, 1)
    show_corner_bb_subplot(axs[0], im_resized, mask_to_bb(Y_resized))
    axs[0].set_title("Original image resized")

    show_corner_bb_subplot(axs[1], im_aug, bb_aug)
    axs[1].set_title("Warped image")

    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='Cats_Test445.png', help="image name")
    parser.add_argument('--annotations', default='Cats_Test445.xml', help="annotations xml file")

    args = parser.parse_args()

    show(args)
