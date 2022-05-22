'''
Develop a data feeding class
--------------------------
Author: Muhammad Alrabeiah
Date: May, 2022
'''
import torch
from torch.utils.data import Dataset
import scipy.io as sio


class DataFeed(Dataset):
    """
    A class for loading images and labels. It includes a pre-processing
    step for images.
    :param root: this is the path to the data file (input and labels)---Must be a string
    :param trf: whether images needs too be processed or not---Must be a boolean
    :param stats: the mean and std for input images in case trf is True---Must be a dictionary
    with two keys: "mean" and "std"
    -------------------------------
    NOTEs:
    Argument "stats" must have two vectors. One is (3,) for image color means and the other is
    (3,) for image color stds
    """
    def __init__(self,
                 root,
                 trf,
                 stats):
        super().__init__()

        data_dict = sio.loadmat(root)
        self.imgs = data_dict['img']
        self.labels = data_dict['cls'].reshape(-1)
        self.trf = trf
        self.mean = stats['mean'].reshape(3,-1)
        self.std = stats['std'].reshape(3,-1)

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):
        x = self.imgs[idx,:].reshape(3,32,32)
        if self.trf:
            x = x/255
            x[0, :] = (x[0, :] - self.mean[0]) / self.std[0]
            x[1, :] = (x[1, :] - self.mean[1]) / self.std[1]
            x[1, :] = (x[2, :] - self.mean[2]) / self.std[2]
        x = torch.from_numpy(x)
        t = torch.tensor([self.labels[idx]],dtype=torch.int64)
        return [x,t]