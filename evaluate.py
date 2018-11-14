from __future__ import print_function
import argparse
from tqdm import tqdm
import os
import PIL.Image as Image

import torch
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision.datasets as datasets

from model import coarseNet, fineNet
import pdb

parser = argparse.ArgumentParser(description='PyTorch GTSRB evaluation script')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder do you want to save the model')
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
fine_model = fineNet()

coarse_model.load_state_dict(coarse_state_dict)
fine_model.load_state_dict(fine_state_dict)
coarse_model.eval()
fine_model.eval()

from data import rgb_data_transforms, depth_data_transforms, input_for_plot_transforms
test_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/test_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
test_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/test_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
input_for_plot_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/test_images/rgb/', transform = input_for_plot_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)


batch_idx = 0
for(rgb, depth, plot_input) in zip(test_rgb_loader, test_depth_loader, input_for_plot_loader):
    rgb, depth, plot_input = Variable(rgb[0], requires_grad = False), Variable(depth[0], requires_grad = False), Variable(plot_input[0], requires_grad = False)
    coarse_output = coarse_model(rgb)
    fine_output = fine_model(rgb, coarse_output)
    actual_output = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
    batch_idx = batch_idx + 1
    if batch_idx == 1: break