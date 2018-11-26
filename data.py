from __future__ import print_function
import zipfile
import os
import pdb
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import h5py
import numpy as np
from PIL import Image
import torch.nn as nn
import torch

class TransposeDepthInput(object):
    def __call__(self, depth):
        depth = depth.transpose((2, 0, 1))
        depth = torch.from_numpy(depth)
        pdb.set_trace()
        pool = nn.AvgPool2d(2, padding = -1)
        depth = pool(depth)
        depth = pool(depth)
        return depth

rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

class RGBDataset(Dataset):
    def __init__(self, filename, type, transform = None):
        f = h5py.File(filename, 'r')
        if type == "training":
            self.images = f['images'][:16]
        elif type == "validation":
            self.images = f['images'][1024:1248]
        elif type == "test":
            self.images = f['images'][1248:]
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        # pdb.set_trace()
        image = image.transpose((2, 1, 0))
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        return image

class DepthDataset(Dataset):
    def __init__(self, filename, type, transform = None):
        f = h5py.File(filename, 'r')
        if type == "training":
            self.depths = f['depths'][:128]
        elif type == "validation":
            self.depths = f['depths'][1024:1248]
        elif type == "test":
            self.depths = f['depths'][1248:]
        self.transform = transform

    def __len__(self):
        return len(self.depths)

    def __getitem__(self, idx):
        depth = self.depths[idx]
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        if self.transform:
            depth = self.transform(depth)
        return depth

obj = DepthDataset('nyu_depth_v2_labeled.mat', 'training', transform = depth_data_transforms)
obj[0]

# class TransposeRGBInput(object):
#     def __call__(self, image):
#         image = image.transpose((2, 1, 0))
#         return torch.from_numpy(image)