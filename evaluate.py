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

parser = argparse.ArgumentParser(description='PyTorch depth prediction evaluation script')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder have you saved the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--model_no', type=int, default = 1, metavar='N',
                    help='Which model no to evaluate (default: 1(first model))')
parser.add_argument('--batch-size', type = int, default = 8, metavar = 'N',
                    help='input batch size for training (default: 8)')

args = parser.parse_args()

output_height = 55 
output_width = 74

coarse_state_dict = torch.load("models/" + args.model_folder + "/coarse_model_" + str(args.model_no) + ".pth")
fine_state_dict = torch.load("models/" + args.model_folder + "/fine_model_" + str(args.model_no) + ".pth")

coarse_model = coarseNet()
coarse_model.cuda()
fine_model = fineNet()
fine_model.cuda()

dtype=torch.cuda.FloatTensor

coarse_model.load_state_dict(coarse_state_dict)
fine_model.load_state_dict(fine_state_dict)
coarse_model.eval()
fine_model.eval()

from data import rgb_data_transforms, depth_data_transforms, input_for_plot_transforms, RGBDataset, DepthDataset # data.py in the same folde
test_rgb_loader = torch.utils.data.DataLoader(RGBDataset('nyu_depth_v2_labeled.mat', 'test', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
test_depth_loader = torch.utils.data.DataLoader(DepthDataset('nyu_depth_v2_labeled.mat', 'test', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
input_for_plot_loader = torch.utils.data.DataLoader(RGBDataset('nyu_depth_v2_labeled.mat', 'test', transform = input_for_plot_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)

def plot_grid(fig, plot_input, coarse_output, fine_output, actual_output, row_no):
    grid = ImageGrid(fig, 141, nrows_ncols=(row_no, 4), axes_pad=0.05, label_mode="1")
    for i in range(row_no):
        for j in range(4):
            if(j == 0):
                grid[i*4+j].imshow(np.transpose(plot_input[i], (1, 2, 0)), interpolation="nearest")
            if(j == 1):
                grid[i*4+j].imshow(np.transpose(coarse_output[i][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")
            if(j == 2):
                grid[i*4+j].imshow(np.transpose(fine_output[i][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")
            if(j == 3):
                grid[i*4+j].imshow(np.transpose(actual_output[i][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")

batch_idx = 0
for batch_idx,(rgb, depth, plot_input) in enumerate(zip(test_rgb_loader, test_depth_loader, input_for_plot_loader)):
    rgb, depth, plot_input = Variable(rgb.cuda(), requires_grad = False), Variable(depth.cuda(), requires_grad = False), Variable(plot_input.cuda(), requires_grad = False)
    #print('rgb size:{} depth size:{}'.format(rgb.size(), depth.size()))
    print('evaluating batch:' + str(batch_idx))
    coarse_output = coarse_model(rgb)
    fine_output = fine_model(rgb.type(dtype), coarse_output)
    depth_dim = list(depth.size())
    actual_output = depth[:,0,:,:].view(depth_dim[0], 1, output_height, output_width)
    F = plt.figure(1, (30, 60))
    F.subplots_adjust(left=0.05, right=0.95)
    plot_grid(F, plot_input, coarse_output, fine_output, actual_output, depth_dim[0])
    plt.savefig("plots/" + args.model_folder + "_" + str(args.model_no) + "_" + str(batch_idx) + ".jpg")
    plt.show()
    #batch_idx = batch_idx + 1
    #if batch_idx == 1: break