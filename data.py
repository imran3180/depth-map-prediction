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
        depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
        depth = nn.functional.interpolate(depth, size=(55, 74), mode='bilinear', align_corners=False)
        depth = torch.log(depth[0])
        return depth

rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

depth_data_transforms = transforms.Compose([
    TransposeDepthInput(),
])

input_for_plot_transforms = transforms.Compose([
    transforms.Resize((55, 74)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

class NYUDataset(Dataset):
    def __init__(self, filename, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')
        if type == "training":
            self.images = f['images'][:1]
            self.depths = f['depths'][:1]
        elif type == "validation":
            self.images = f['images'][:1]
            self.depths = f['depths'][:1]
        elif type == "test":
            self.images = f['images'][:1]
            self.depths = f['depths'][:1]
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.transpose((2, 1, 0))
        image = Image.fromarray(image)
        if self.rgb_transform:
            image = self.rgb_transform(image)
        depth = self.depths[idx]
        depth = np.reshape(depth, (1, depth.shape[0], depth.shape[1]))
        depth = depth.transpose((2, 1, 0))
        if self.depth_transform:
            depth = self.depth_transform(depth)
        sample = {'image': image, 'depth': depth}
        return sample