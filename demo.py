import sys

sys.path.append('\\')
sys.path.append('core')

from core.datasets.PetDataset import PetDataset
from core.BB_Model import BB_model
from core.utils.BB_utils import show_corner_bb
from core.utils.utils import read_image
import cv2
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse


def demo(image_path, model_path='models/BB_model.pth'):

    im = read_image(image_path)
    im = cv2.resize(im, (int(1.5*300), 300))

    # test Dataset
    test_ds = PetDataset(pd.DataFrame([{'path':image_path}])['path'],pd.DataFrame([{'bb':np.array([0,0,0,0])}])['bb'],pd.DataFrame([{'y':[0]}])['y'])
    x, y_class, y_bb = test_ds[0]

    xx = torch.FloatTensor(x[None,])

    model = BB_model().cuda()
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        out_class, out_bb = model(xx.cuda())

    bb_hat = out_bb.detach().cpu().numpy()
    bb_hat = bb_hat.astype(int)
    show_corner_bb(im, bb_hat[0])

    if torch.argmax(out_class) == 0:
        plt.title("Its a cat !")
    else:
        plt.title("Its a dog !")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_path', type=str, default='D:/CatsVsDogsDataset/images/Cats_Test0.png', help="path to image")
    parser.add_argument('--model_path', type=str, default='models/BB_model.pth', help="path to model")

    args = parser.parse_args()

    demo(args.image_path, args.model_path)
