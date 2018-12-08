import matplotlib
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from model import coarseNet, fineNet
import pdb
import numpy as np

def plot_grid(fig, rgb, depth, row_no):
    grid = ImageGrid(fig, 141, nrows_ncols = (row_no, 2), axes_pad=0.05, label_mode="1")
    # pdb.set_trace()
    for i in range(row_no):
        for j in range(2):
            if(j == 0):
                grid[i*2+j].imshow(np.transpose(rgb[i].numpy(), (1, 2, 0)), interpolation="nearest")
            if(j == 1):
                grid[i*2+j].imshow(np.transpose(depth[i][0].numpy(), (0, 1)), interpolation="nearest")


