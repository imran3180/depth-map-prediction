from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import pdb
from logger import Logger
import os

############## Image related
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
####################

# Training settings
parser = argparse.ArgumentParser(description='PyTorch depth map prediction example')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder do you want to save the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type = int, default = 1, metavar = 'N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default = 10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--suffix', type=str, default='', metavar='D',
                    help='suffix for the filename of models and output files')
args = parser.parse_args()

torch.manual_seed(args.seed)    # setting seed for random number generation

output_height = 55 
output_width = 74

from data import NYUDataset, rgb_data_transforms, depth_data_transforms
from image_helper import plot_grid

train_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat', 
                                                       'training', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = True, num_workers = 0)

val_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'validation', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 0)

test_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'test', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 0)

from model import coarseNet, fineNet
coarse_model = coarseNet()
fine_model = fineNet()

# coarse_optimizer = optim.SGD([
#                         {'params': coarse_model.conv1.parameters(), 'lr': 0.001},
#                         {'params': coarse_model.conv2.parameters(), 'lr': 0.001},
#                         {'params': coarse_model.conv3.parameters(), 'lr': 0.001},
#                         {'params': coarse_model.conv4.parameters(), 'lr': 0.001},
#                         {'params': coarse_model.conv5.parameters(), 'lr': 0.001},
#                         {'params': coarse_model.fc1.parameters(), 'lr': 0.1},
#                         {'params': coarse_model.fc2.parameters(), 'lr': 0.1}
#                     ], lr = 0.001, momentum = 0.9)

coarse_optimizer = optim.SGD([
                        {'params': coarse_model.conv1.parameters(), 'lr': 0.01},
                        {'params': coarse_model.conv2.parameters(), 'lr': 0.01},
                        {'params': coarse_model.conv3.parameters(), 'lr': 0.01},
                        {'params': coarse_model.conv4.parameters(), 'lr': 0.01},
                        {'params': coarse_model.conv5.parameters(), 'lr': 0.01},
                        {'params': coarse_model.fc1.parameters(), 'lr': 0.1},
                        {'params': coarse_model.fc2.parameters(), 'lr': 0.1}
                    ], lr = 0.01, momentum = 0.9)


# fine_optimizer = optim.SGD([
#                         {'params': coarse_model.conv1.parameters(), 'lr': 0.001},
#                         {'params': coarse_model.conv2.parameters(), 'lr': 0.01},
#                         {'params': coarse_model.conv3.parameters(), 'lr': 0.001}
#                     ], lr = 0.001, momentum = 0.9)

# fine_optimizer = optim.SGD([
#                         {'params': coarse_model.conv1.parameters(), 'lr': 0.01},
#                         {'params': coarse_model.conv2.parameters(), 'lr': 0.1},
#                         {'params': coarse_model.conv3.parameters(), 'lr': 0.01}
#                     ], lr = 0.01, momentum = 0.9)


# coarse_optimizer = optim.SGD(coarse_model.parameters(), lr=args.lr, momentum=args.momentum)
fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)


logger = Logger('./logs/' + args.model_folder)

def custom_loss_function(output, target):
    # di = torch.log(target) - torch.log(output)
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.sum()

def train_coarse(epoch):
    coarse_model.train()
    for batch_idx, data in enumerate(train_loader):
        # variable
        rgb, depth = torch.tensor(data['image'], requires_grad = True), torch.tensor(data['depth'], requires_grad = True)
        coarse_optimizer.zero_grad()
        output = coarse_model(rgb)
        # to check
        target = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
        loss = custom_loss_function(output, target)
        loss.backward()
        coarse_optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "coarse training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def train_fine(epoch):
    coarse_model.eval()
    fine_model.train()
    for batch_idx, data in enumerate(train_loader):
        # variable
        rgb, depth = torch.tensor(data['image'], requires_grad = True), torch.tensor(data['depth'], requires_grad = True)
        fine_optimizer.zero_grad()
        coarse_output = coarse_model(rgb)   # it should print last epoch error since coarse is fixed.
        output = fine_model(rgb, coarse_output)
        target = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
        loss = custom_loss_function(output, target)
        loss.backward()
        fine_optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "fine training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

def coarse_validation():
    coarse_model.eval()
    coarse_validation_loss = 0
    for batch_idx, data in enumerate(val_loader):
        # variable
        rgb, depth = torch.tensor(data['image'], requires_grad = False), torch.tensor(data['depth'], requires_grad = False)
        coarse_output = coarse_model(rgb)
        target = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
        coarse_validation_loss += custom_loss_function(coarse_output, target).item()
    coarse_validation_loss /= (batch_idx + 1)
    logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
    print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))

def fine_validation():
    fine_model.eval()
    fine_validation_loss = 0
    for batch_idx,data in enumerate(val_loader):
        # variable
        rgb, depth = torch.tensor(data['image'], requires_grad = False), torch.tensor(data['depth'], requires_grad = False)
        coarse_output = coarse_model(rgb)
        fine_output = fine_model(rgb, coarse_output)
        target = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
        fine_validation_loss += custom_loss_function(fine_output, target).item()
    fine_validation_loss /= (batch_idx + 1)
    logger.scalar_summary("fine validation loss", fine_validation_loss, epoch)
    print('\nValidation set: Average loss(Fine): {:.4f} \n'.format(fine_validation_loss))

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

for epoch in range(1, args.epochs + 1):
    print("********* Training the Coarse Model **************")
    train_coarse(epoch)
    coarse_validation()
    model_file = folder_name + "/" + 'coarse_model_' + str(epoch) + '.pth'
    if(epoch%100 == 0):
        torch.save(coarse_model.state_dict(), model_file)

coarse_model.eval() # stoping the coarse model to train.

for epoch in range(1, args.epochs + 1):
    print("********* Training the Fine Model ****************")
    train_fine(epoch)
    fine_validation()
    model_file = folder_name + "/" + 'fine_model_' + str(epoch) + '.pth'
    if(epoch%100 == 0):
        torch.save(fine_model.state_dict(), model_file)