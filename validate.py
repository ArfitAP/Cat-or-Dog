import sys

from core.utils.utils import count_parameters

sys.path.append('..\\')
sys.path.append('core')

import torch
from core.BB_Model import BB_model
from core.data import get_validation_dataframe
from core.datasets.PetDataset import PetDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse


def fetch_dataloader(batch_size=16):
    images_path = 'D:/CatsVsDogsDataset/images'
    anno_path = 'D:/CatsVsDogsDataset/annotations'

    df_valid = get_validation_dataframe(anno_path, images_path)
    df_valid = df_valid.reset_index(drop=True)
    X = df_valid[['new_path', 'new_bb']]
    Y = df_valid['class']
    X_val, y_val = X, Y

    valid_ds = PetDataset(X_val['new_path'],X_val['new_bb'], y_val)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    return valid_dl, len(X_val)


def val_metrics(model, valid_dl, C=1000):
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x, y_class, y_bb in valid_dl:
        batch = y_class.shape[0]
        x = x.cuda().float()
        y_class = y_class.cuda()
        y_bb = y_bb.cuda().float()
        out_class, out_bb = model(x)
        loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
        loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
        loss_bb = loss_bb.sum()
        loss = loss_class + loss_bb/C
        _, pred = torch.max(out_class, 1)
        correct += pred.eq(y_class).sum().item()
        sum_loss += loss.item()
        total += batch
    return sum_loss/total, correct/total


def validate(model_path='models/BB_model.pth'):
    model = BB_model().cuda()
    model.load_state_dict(torch.load(model_path))

    print("Number of parameters: " + str(count_parameters(model)))
    print("Preparing data ...")

    valid_dl, data_len = fetch_dataloader()

    print("Validation started, number of images: " + str(data_len))

    val_loss, val_acc = val_metrics(model, valid_dl)

    print("val_loss %.3f, val_acc %.3f" % (val_loss, val_acc))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, default='models/BB_model.pth', help="path to model")

    args = parser.parse_args()

    validate(args.model_path)
