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
parser.add_argument('--batch-size', type = int, default = 8, metavar = 'N',
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
from data import initialize_data, rgb_data_transforms, depth_data_transforms # data.py in the same folder
initialize_data(args.data) # extracts the zip files, makes a validation set

train_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/train_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=True, num_workers=1)
train_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/train_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=True, num_workers=1)
val_rgb_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images/rgb/', transform = rgb_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)
val_depth_loader = torch.utils.data.DataLoader(datasets.ImageFolder(args.data + '/val_images/depth/', transform = depth_data_transforms), batch_size=args.batch_size, shuffle=False, num_workers=1)

from model import coarseNet, fineNet
coarse_model = coarseNet()
coarse_model.cuda()
fine_model = fineNet()
fine_model.cuda()
loss_function = nn.MSELoss()

coarse_optimizer = optim.SGD(coarse_model.parameters(), lr=args.lr, momentum=args.momentum)
fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)

dtype=torch.cuda.FloatTensor
logger = Logger('./logs/' + args.model_folder)

def train_coarse(epoch):
    coarse_model.train()
    batch_idx = 0
    for batch_idx, (rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        rgb, depth = Variable(rgb[0].cuda()), Variable(depth[0].cuda())
        #rgb, depth = Variable(rgb[0]), Variable(depth[0])
        coarse_optimizer.zero_grad()
        output = coarse_model(rgb.type(dtype))
        #output = coarse_model(rgb)
        loss = loss_function(output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width))
        loss.backward()
        coarse_optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "coarse training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_rgb_loader.dataset),
                100. * batch_idx / len(train_rgb_loader), loss.item()))
        #batch_idx = batch_idx + 1
        #if batch_idx == 1: break

def train_fine(epoch):
    fine_model.train()
    batch_idx = 0
    for batch_idx,(rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        #rgb, depth = Variable(rgb[0]), Variable(depth[0])
        rgb, depth = Variable(rgb[0].cuda()), Variable(depth[0].cuda())
        fine_optimizer.zero_grad()
        #coarse_output = coarse_model(rgb)   # it should print last epoch error since coarse is fixed.
        #output = fine_model(rgb, coarse_output)
        coarse_output = coarse_model(rgb.type(dtype))   # it should print last epoch error since coarse is fixed.
        output = fine_model(rgb.type(dtype), coarse_output.type(dtype))
        loss = loss_function(output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width))
        loss.backward()
        fine_optimizer.step()
        if batch_idx % args.log_interval == 0:
            training_tag = "fine training loss epoch:" + str(epoch)
            logger.scalar_summary(training_tag, loss.item(), batch_idx)

            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(rgb), len(train_rgb_loader.dataset),
                100. * batch_idx / len(train_rgb_loader), loss.item()))
        #batch_idx = batch_idx + 1
        #iif batch_idx == 1: break

coarse_output = torch.tensor([[0]])

def coarse_validation():
    coarse_model.eval()
    coarse_validation_loss = 0
    batch_idx = 0
    for batch_idx,(rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        #rgb, depth = Variable(rgb[0], requires_grad = False), Variable(depth[0], requires_grad = False)
        #coarse_output = coarse_model(rgb)
        rgb, depth = Variable(rgb[0].cuda(), requires_grad = False), Variable(depth[0].cuda(), requires_grad = False)
        coarse_output = coarse_model(rgb.type(dtype))
        coarse_validation_loss += loss_function(coarse_output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)).item()
        #fine_output = fine_model(rgb, coarse_output)
        #batch_idx = batch_idx + 1
        #if batch_idx == 2: break
    coarse_validation_loss /= batch_idx
    logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
    print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))

def fine_validation():
    fine_model.eval()
    fine_validation_loss = 0
    batch_idx = 0
    for batch_idx,(rgb, depth) in enumerate(zip(train_rgb_loader, train_depth_loader)):
        rgb, depth = Variable(rgb[0].cuda(), requires_grad = False), Variable(depth[0].cuda(), requires_grad = False)
        fine_output = fine_model(rgb.type(dtype), coarse_output)
        fine_validation_loss += loss_function(fine_output, depth[:,0,:,:].view(args.batch_size, 1, output_height, output_width)).item()
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

for epoch in range(1, args.epochs + 1):
    print("********* Training the Fine Model ****************")
    train_fine(epoch)
    fine_validation()
    model_file = folder_name + "/" + 'fine_model_' + str(epoch) + '.pth'
    torch.save(fine_model.state_dict(), model_file)
