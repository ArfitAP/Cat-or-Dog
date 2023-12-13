from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ColorJitter
import numpy as np
from core.data import normalize
from core.utils.augmentation import transformsXY


class PetDataset(Dataset):
    def __init__(self, paths, bb, y, transforms=False):
        self.transforms = transforms
        self.paths = paths.values
        self.bb = bb.values
        self.y = y.values
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        y_class = self.y[idx]
        x, y_bb = transformsXY(path, self.bb[idx], self.transforms, self.photo_aug)
        x = normalize(x)
        x = np.rollaxis(x, 2)
        return x, y_class, y_bb
