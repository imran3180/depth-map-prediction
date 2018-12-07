from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from logger import Logger
import pdb
import os

# Training settings
parser = argparse.ArgumentParser(description='PyTorch depth map prediction example')
parser.add_argument('model_folder', type=str, metavar='F',
                    help='In which folder do you want to save the model')
parser.add_argument('--data', type=str, default='data', metavar='D',
                    help="folder where data is located. train_data.zip and test_data.zip need to be found in the folder")
parser.add_argument('--batch-size', type = int, default = 4, metavar = 'N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default = 10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--suffix', type=str, default='', metavar='D',
                    help='suffix for the filename of models and output files')
args = parser.parse_args()

torch.manual_seed(args.seed)

output_height = 55 
output_width = 74

### Data Initialization and Loading
from data import rgb_data_transforms, depth_data_transforms, RGBDataset, DepthDataset # data.py in the same folder
# initialize_data(args.data) # extracts the zip files, makes a validation set

train_rgb_loader = torch.utils.data.DataLoader(RGBDataset('nyu_depth_v2_labeled.mat', 'training', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=True, num_workers=1)
train_depth_loader = torch.utils.data.DataLoader(DepthDataset('nyu_depth_v2_labeled.mat', 'training', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=True, num_workers=1)
val_rgb_loader = torch.utils.data.DataLoader(RGBDataset('nyu_depth_v2_labeled.mat', 'validation', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
val_depth_loader = torch.utils.data.DataLoader(DepthDataset('nyu_depth_v2_labeled.mat', 'validation', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)

from model import coarseNet, fineNet
coarse_model = coarseNet()
coarse_model.cuda()
fine_model = fineNet()
# loss_function = nn.MSELoss()

# coarse_optimizer = optim.SGD(coarse_model.parameters(), lr=args.lr, momentum=args.momentum)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)

coarse_optimizer = optim.SGD([
                        {'params': coarse_model.conv1.parameters(), 'lr': 0.001},
                        {'params': coarse_model.conv2.parameters(), 'lr': 0.001},
                        {'params': coarse_model.conv3.parameters(), 'lr': 0.001},
                        {'params': coarse_model.conv4.parameters(), 'lr': 0.001},
                        {'params': coarse_model.conv5.parameters(), 'lr': 0.001},
                        {'params': coarse_model.fc1.parameters(), 'lr': 0.1},
                        {'params': coarse_model.fc2.parameters(), 'lr': 0.1}
                    ], lr = 0.001, momentum = 0.9)

fine_optimizer = optim.SGD([
                        {'params': coarse_model.conv1.parameters(), 'lr': 0.001},
                        {'params': coarse_model.conv2.parameters(), 'lr': 0.01},
                        {'params': coarse_model.conv3.parameters(), 'lr': 0.001}
                    ], lr = 0.001, momentum = 0.9)


logger = Logger('./logs/' + args.model_folder)

def custom_loss_function(output, target):
    epsilon = 0.0001
    # output = output + epsilon
    # target = target + epsilon
    # di = torch.log(target) - torch.log(output)
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    # pdb.set_trace()
    return loss.sum()

def train_coarse(epoch):
    coarse_model.train()
    for batch_idx, (rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        rgb, depth = Variable(rgb), Variable(depth)
        #rgb, depth = Variable(rgb[0]), Variable(depth[0])
        coarse_optimizer.zero_grad()
        output = coarse_model(rgb)
        #output = coarse_model(rgb)
        target = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
        # loss = loss_function(output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width))
        loss = custom_loss_function(output, target)
        loss.backward()
        coarse_optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "coarse training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_rgb_loader.dataset),
                100. * batch_idx / len(train_rgb_loader), loss.item()))
        #batch_idx = batch_idx + 1

def train_fine(epoch):
    coarse_model.eval()
    fine_model.train()
    for batch_idx,(rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        #rgb, depth = Variable(rgb[0]), Variable(depth[0])
        rgb, depth = Variable(rgb), Variable(depth)
        fine_optimizer.zero_grad()
        #coarse_output = coarse_model(rgb)   # it should print last epoch error since coarse is fixed.
        #output = fine_model(rgb, coarse_output)
        coarse_output = coarse_model(rgb)   # it should print last epoch error since coarse is fixed.
        output = fine_model(rgb, coarse_output)
        target = depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)
        # loss = loss_function(output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width))
        # pdb.set_trace()
        loss = custom_loss_function(output, target)
        loss.backward()
        fine_optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "fine training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_rgb_loader.dataset),
                100. * batch_idx / len(train_rgb_loader), loss.item()))
        #batch_idx = batch_idx + 1

def coarse_validation():
    coarse_model.eval()
    coarse_validation_loss = 0
    for batch_idx,(rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        #rgb, depth = Variable(rgb[0], requires_grad = False), Variable(depth[0], requires_grad = False)
        #coarse_output = coarse_model(rgb)
        rgb, depth = Variable(rgb, requires_grad = False), Variable(depth, requires_grad = False)
        coarse_output = coarse_model(rgb)
        coarse_validation_loss += custom_loss_function(coarse_output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)).item()
        #batch_idx = batch_idx + 1
        #if batch_idx == 1: break
    coarse_validation_loss /= batch_idx
    logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
    print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))

def fine_validation():
    fine_model.eval()
    fine_validation_loss = 0
    for batch_idx,(rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        rgb, depth = Variable(rgb, requires_grad = False), Variable(depth, requires_grad = False)
        coarse_output = coarse_model(rgb)
        fine_output = fine_model(rgb, coarse_output)
        fine_validation_loss += custom_loss_function(fine_output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)).item()
        #batch_idx = batch_idx + 1
        #if batch_idx == 1: break
    fine_validation_loss /= batch_idx
    logger.scalar_summary("fine validation loss", fine_validation_loss, epoch)
    print('\nValidation set: Average loss(Fine): {:.4f} \n'.format(fine_validation_loss))

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

for epoch in range(1, args.epochs + 1):
    print("********* Training the Coarse Model **************")
    train_coarse(epoch)
    coarse_validation()
    model_file = folder_name + "/" + 'coarse_model_' + str(epoch) + '.pth'
    torch.save(coarse_model.state_dict(), model_file)

coarse_model.eval()

for epoch in range(1, args.epochs + 1):
    print("********* Training the Fine Model ****************")
    train_fine(epoch)
    fine_validation()
    model_file = folder_name + "/" + 'fine_model_' + str(epoch) + '.pth'
    torch.save(fine_model.state_dict(), model_file)
