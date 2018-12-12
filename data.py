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
import random

# class TransposeDepthInput(object):
#     def __call__(self, depth):
#         depth = depth.transpose((2, 0, 1))
#         depth = torch.from_numpy(depth)
#         depth = depth.view(1, depth.shape[0], depth.shape[1], depth.shape[2])
#         depth = nn.functional.interpolate(depth, size = (55, 74), mode='bilinear', align_corners=False)
#         depth = torch.log(depth)
#         return depth[0]

just_tensor_transform = transforms.Compose([
    transforms.ToTensor(),
])

rgb_data_transforms = transforms.Compose([
    # transforms.Resize((228, 304)),    # Different for Input Image & Depth Image
    transforms.CenterCrop((228, 304)),
    transforms.ToTensor(),
])

depth_data_transforms = transforms.Compose([
    # TransposeDepthInput(),
    transforms.CenterCrop((228, 304)),
    transforms.Resize((55, 74)),
    transforms.ToTensor()
])

input_for_plot_transforms = transforms.Compose([
    transforms.CenterCrop((228, 304)),
    transforms.Resize((55, 74)),    # Different for Input Image & Depth Image
    transforms.ToTensor(),
])

color_transform = transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1)

class NYUDataset(Dataset):
    def __init__(self, filename, type, rgb_transform = None, depth_transform = None):
        f = h5py.File(filename, 'r')
        if type == "training":
            self.images = f['images'][0:1024]
            self.depths = f['depths'][0:1024]
            # self.images = f['images'][0:4]
            # self.depths = f['depths'][0:4]
        elif type == "validation":
            self.images = f['images'][1024:1248]
            self.depths = f['depths'][1024:1248]
            # self.images = f['images'][1024:1028]
            # self.depths = f['depths'][1024:1028]
        elif type == "test":
            self.images = f['images'][1248:]
            self.depths = f['depths'][1248:]
            # self.images = f['images'][0:32]
            # self.depths = f['depths'][0:32]
        self.type = type
        self.rgb_transform = rgb_transform
        self.depth_transform = depth_transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        image = image.transpose((2, 1, 0))
        image = Image.fromarray(image)
        
        depth = self.depths[idx]
        depth = depth.transpose((1, 0))
        depth = Image.fromarray(depth)

        if self.type == "training":
            # random cropping
            i, j, th, tw = transforms.RandomCrop.get_params(image, (228, 304))
            image = transforms.functional.crop(image, i, j, th, tw)
            depth = transforms.functional.crop(depth, i, j, th, tw)

        if self.type == "training":
            # color transformation    
            # if random.random() < 0.3:
            #     image = color_transform(image)

            if random.random() < 0.5:
                image = transforms.functional.hflip(image)
                depth = transforms.functional.hflip(depth)
                # flip = random.randint(0, 3)
                # Flipping transformations    
                # if flip == 0:
                #     image = transforms.functional.hflip(image)
                #     depth = transforms.functional.hflip(depth)
                # elif flip == 1:
                #     image = transforms.functional.vflip(image)
                #     depth = transforms.functional.vflip(depth)
                # elif flip == 2:
                #     image = transforms.functional.hflip(image)
                #     image = transforms.functional.vflip(image)
                #     depth = transforms.functional.hflip(depth)
                #     depth = transforms.functional.vflip(depth)
                # elif flip == 3:
                #     image = transforms.functional.vflip(image)
                #     image = transforms.functional.hflip(image)
                #     depth = transforms.functional.vflip(depth)
                #     depth = transforms.functional.hflip(depth)

            # Rotational Transform    
            # if random.random() < 0.3:
            #     angle = random.uniform(-5, 5)
            #     image = transforms.functional.rotate(image, angle, resample=False, expand=False, center=None)
            #     depth = transforms.functional.rotate(depth, angle, resample=False, expand=False, center=None)

        

        if self.rgb_transform:
            image = self.rgb_transform(image)

        if self.depth_transform:
            depth = self.depth_transform(depth)

        depth = torch.log(depth)
        sample = {'image': image, 'depth': depth}
        return sample