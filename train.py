import sys

from core.utils.utils import count_parameters

sys.path.append('..\\')
sys.path.append('core')

import torch
from core.BB_Model import BB_model
import torch.optim as optim
from sklearn.model_selection import train_test_split
from core.data import get_train_dataframe
from core.datasets.PetDataset import PetDataset
from torch.utils.data import DataLoader
import torch.nn.functional as F
import argparse


def fetch_optimizer(model, args, data_len):

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = optim.Adam(parameters, lr=args.lr)

    steps_per_epoch = int(data_len / args.bs)
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer, args.lr, steps_per_epoch=steps_per_epoch,
                                              epochs=args.epochs + 1, pct_start=0.1, cycle_momentum=False,
                                              anneal_strategy='linear')

    return optimizer, scheduler


def fetch_dataloader(batch_size):
    images_path = 'D:/CatsVsDogsDataset/images'
    anno_path = 'D:/CatsVsDogsDataset/annotations'

    df_train = get_train_dataframe(anno_path, images_path)
    df_train = df_train.reset_index(drop=True)
    X = df_train[['new_path', 'new_bb']]
    Y = df_train['class']
    X_train, X_val, y_train, y_val = train_test_split(X, Y, test_size=0.1, random_state=42)

    train_ds = PetDataset(X_train['new_path'],X_train['new_bb'], y_train, transforms=True)
    valid_ds = PetDataset(X_val['new_path'],X_val['new_bb'], y_val)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size)

    return train_dl, valid_dl, len(X_train)


def train_epocs(model, optimizer, scheduler, train_dl, val_dl, epochs=10, C=1000):
    train_loss = 0
    idx = 0
    for i in range(epochs):
        model.train()
        total = 0
        sum_loss = 0
        for x, y_class, y_bb in train_dl:
            batch = y_class.shape[0]
            x = x.cuda().float()
            y_class = y_class.cuda()
            y_bb = y_bb.cuda().float()
            out_class, out_bb = model(x)
            loss_class = F.cross_entropy(out_class, y_class, reduction="sum")
            loss_bb = F.l1_loss(out_bb, y_bb, reduction="none").sum(1)
            loss_bb = loss_bb.sum()
            loss = loss_class + loss_bb/C
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            idx += 1
            total += batch
            sum_loss += loss.item()
        train_loss = sum_loss/total
        #val_loss, val_acc = val_metrics(model, valid_dl, C)
        #print("train_loss %.3f val_loss %.3f val_acc %.3f" % (train_loss, val_loss, val_acc))
        print("epoch:" + str(i) + ", train_loss %.3f" % (train_loss))

    PATH = './models/BB_model.pth'
    torch.save(model.state_dict(), PATH)

    return train_loss


def train(args):
    model = BB_model().cuda()

    print("Number of parameters: " + str(count_parameters(model)))
    print("Preparing data ...")

    train_dl, valid_dl, data_len = fetch_dataloader(args.bs)
    optimizer, scheduler = fetch_optimizer(model, args, data_len)

    print("Training started")

    train_epocs(model, optimizer, scheduler, train_dl, valid_dl, epochs=args.epochs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--lr', type=float, default='0.005', help="learning rate")
    parser.add_argument('--bs', type=int, default='32', help="batch size")
    parser.add_argument('--epochs', type=int, default='10', help="epochs")

    args = parser.parse_args()

    print("learning rate = " + str(args.lr) + ", batch size = " + str(args.bs) + ", epochs = " + str(args.epochs))

    train(args)
