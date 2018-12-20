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
parser.add_argument('--batch-size', type = int, default = 32, metavar = 'N',
                    help='input batch size for training (default: 8)')
parser.add_argument('--epochs', type=int, default = 10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                    help='SGD momentum (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=20, metavar='N',
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
                                            shuffle = True, num_workers = 5)

val_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'validation', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 5)

test_loader = torch.utils.data.DataLoader(NYUDataset( 'nyu_depth_v2_labeled.mat',
                                                       'test', 
                                                        rgb_transform = rgb_data_transforms, 
                                                        depth_transform = depth_data_transforms), 
                                            batch_size = args.batch_size, 
                                            shuffle = False, num_workers = 5)

from model import coarseNet, fineNet
coarse_model = coarseNet()
fine_model = fineNet()
coarse_model.cuda()
fine_model.cuda()

# Paper values for SGD
coarse_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.001},{'params': coarse_model.conv2.parameters(), 'lr': 0.001},{'params': coarse_model.conv3.parameters(), 'lr': 0.001},{'params': coarse_model.conv4.parameters(), 'lr': 0.001},{'params': coarse_model.conv5.parameters(), 'lr': 0.001},{'params': coarse_model.fc1.parameters(), 'lr': 0.1},{'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr = 0.001, momentum = 0.9)
fine_optimizer = optim.SGD([{'params': fine_model.conv1.parameters(), 'lr': 0.001},{'params': fine_model.conv2.parameters(), 'lr': 0.01},{'params': fine_model.conv3.parameters(), 'lr': 0.001}], lr = 0.001, momentum = 0.9)

# Changed values
# coarse_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.01},{'params': coarse_model.conv2.parameters(), 'lr': 0.01},{'params': coarse_model.conv3.parameters(), 'lr': 0.01},{'params': coarse_model.conv4.parameters(), 'lr': 0.01},{'params': coarse_model.conv5.parameters(), 'lr': 0.01},{'params': coarse_model.fc1.parameters(), 'lr': 0.1},{'params': coarse_model.fc2.parameters(), 'lr': 0.1}], lr = 0.01, momentum = 0.9)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)
# fine modified but default fine work more.
#fine_optimizer = optim.SGD([{'params': coarse_model.conv1.parameters(), 'lr': 0.01},{'params': coarse_model.conv2.parameters(), 'lr': 0.1},{'params': coarse_model.conv3.parameters(), 'lr': 0.01}], lr = 0.01, momentum = 0.9)

# default SGD optimiser - don't work
# # coarse_optimizer = optim.SGD(coarse_model.parameters(), lr=args.lr, momentum=args.momentum)
# fine_optimizer = optim.SGD(fine_model.parameters(), lr=args.lr, momentum=args.momentum)

# coarse_optimizer = optim.Adadelta(coarse_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)
# fine_optimizer = optim.Adadelta(fine_model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

# coarse_optimizer = optim.Adagrad(coarse_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)
# fine_optimizer = optim.Adagrad(fine_model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

# coarse_optimizer = optim.Adam(coarse_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
# fine_optimizer = optim.Adam(fine_model.parameters(), lr=0.01, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)

# coarse_optimizer = optim.Adamax(coarse_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)
# fine_optimizer = optim.Adamax(fine_model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# coarse_optimizer = optim.ASGD(coarse_model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)
# fine_optimizer = optim.ASGD(fine_model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)


dtype=torch.cuda.FloatTensor
logger = Logger('./logs/' + args.model_folder)

def custom_loss_function(output, target):
    # di = output - target
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = 0.5*torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()

def scale_invariant(output, target):
    # di = output - target
    di = target - output
    n = (output_height * output_width)
    di2 = torch.pow(di, 2)
    fisrt_term = torch.sum(di2,(1,2,3))/n
    second_term = torch.pow(torch.sum(di,(1,2,3)), 2)/ (n**2)
    loss = fisrt_term - second_term
    return loss.mean()

# def custom_loss_function(output, target):
#     diff = target - output
#     alpha = torch.sum(diff, (1,2,3))/(output_height * output_width)
#     loss_val = 0
#     for i in range(alpha.shape[0]):
#        loss_val += torch.sum(torch.pow(((output[i] - target[i]) - alpha[i]), 2))/(2 * output_height * output_width)
#     loss_val = loss_val/output.shape[0] 
#     return loss_val

# All Error Function
def threeshold_percentage(output, target, threeshold_val):
    d1 = torch.exp(output)/torch.exp(target)
    d2 = torch.exp(target)/torch.exp(output)
    # d1 = output/target
    # d2 = target/output
    max_d1_d2 = torch.max(d1,d2)
    zero = torch.zeros(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    one = torch.ones(output.shape[0], output.shape[1], output.shape[2], output.shape[3])
    bit_mat = torch.where(max_d1_d2.cpu() < threeshold_val, one, zero)
    count_mat = torch.sum(bit_mat, (1,2,3))
    threeshold_mat = count_mat/(output.shape[2] * output.shape[3])
    return threeshold_mat.mean()

def rmse_linear(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    diff = actual_output - actual_target
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1,2,3))/(output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return rmse.mean()

def rmse_log(output, target):
    diff = output - target
    # diff = torch.log(output) - torch.log(target)
    diff2 = torch.pow(diff, 2)
    mse = torch.sum(diff2, (1,2,3))/(output.shape[2] * output.shape[3])
    rmse = torch.sqrt(mse)
    return mse.mean()

def abs_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    abs_relative_diff = torch.abs(actual_output - actual_target)/actual_target
    abs_relative_diff = torch.sum(abs_relative_diff, (1,2,3))/(output.shape[2] * output.shape[3])
    return abs_relative_diff.mean()

def squared_relative_difference(output, target):
    actual_output = torch.exp(output)
    actual_target = torch.exp(target)
    # actual_output = output
    # actual_target = target
    square_relative_diff = torch.pow(torch.abs(actual_output - actual_target), 2)/actual_target
    square_relative_diff = torch.sum(square_relative_diff, (1,2,3))/(output.shape[2] * output.shape[3])
    return square_relative_diff.mean()    

def train_coarse(epoch):
    coarse_model.train()
    train_coarse_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = True), torch.tensor(data['depth'].cuda(), requires_grad = True)
        coarse_optimizer.zero_grad()
        output = coarse_model(rgb.type(dtype))
        loss = custom_loss_function(output, depth)
        loss.backward()
        coarse_optimizer.step()
        train_coarse_loss += loss.item()
    train_coarse_loss /= (batch_idx + 1)
    return train_coarse_loss
    # print('Epoch: {} Training set(Coarse) average loss: {:.4f}'.format(epoch, train_coarse_loss))
        # if batch_idx % args.log_interval == 0:
        #     training_tag = "coarse training loss epoch:" + str(epoch)
        #     logger.scalar_summary(training_tag, loss.item(), batch_idx)

        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(rgb), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def train_fine(epoch):
    coarse_model.eval()
    fine_model.train()
    train_fine_loss = 0
    for batch_idx, data in enumerate(train_loader):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = True), torch.tensor(data['depth'].cuda(), requires_grad = True)
        fine_optimizer.zero_grad()
        coarse_output = coarse_model(rgb.type(dtype))   # it should print last epoch error since coarse is fixed.
        output = fine_model(rgb.type(dtype), coarse_output.type(dtype))
        loss = custom_loss_function(output, depth)
        loss.backward()
        fine_optimizer.step()
        train_fine_loss += loss.item()
    train_fine_loss /= (batch_idx + 1)
    return train_fine_loss
    # print('Epoch: {} Training set(Fine) average loss: {:.4f}'.format(epoch, train_fine_loss))
        # if batch_idx % args.log_interval == 0:
        #     training_tag = "fine training loss epoch:" + str(epoch)
        #     logger.scalar_summary(training_tag, loss.item(), batch_idx)

        #     print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
        #         epoch, batch_idx * len(rgb), len(train_loader.dataset),
        #         100. * batch_idx / len(train_loader), loss.item()))

def coarse_validation(epoch, training_loss):
    coarse_model.eval()
    coarse_validation_loss = 0
    scale_invariant_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0

    for batch_idx, data in enumerate(val_loader):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = False), torch.tensor(data['depth'].cuda(), requires_grad = False)
        coarse_output = coarse_model(rgb.type(dtype))
        coarse_validation_loss += custom_loss_function(coarse_output, depth).item()
        # all error functions
        scale_invariant_loss += scale_invariant(coarse_output, depth)
        delta1_accuracy += threeshold_percentage(coarse_output, depth, 1.25)
        delta2_accuracy += threeshold_percentage(coarse_output, depth, 1.25*1.25)
        delta3_accuracy += threeshold_percentage(coarse_output, depth, 1.25*1.25*1.25)
        rmse_linear_loss += rmse_linear(coarse_output, depth)
        rmse_log_loss += rmse_log(coarse_output, depth)
        abs_relative_difference_loss += abs_relative_difference(coarse_output, depth)
        squared_relative_difference_loss += squared_relative_difference(coarse_output, depth)

    coarse_validation_loss /= (batch_idx + 1)
    delta1_accuracy /= (batch_idx + 1)
    delta2_accuracy /= (batch_idx + 1)
    delta3_accuracy /= (batch_idx + 1)
    rmse_linear_loss /= (batch_idx + 1)
    rmse_log_loss /= (batch_idx + 1)
    abs_relative_difference_loss /= (batch_idx + 1)
    squared_relative_difference_loss /= (batch_idx + 1)
    logger.scalar_summary("coarse validation loss", coarse_validation_loss, epoch)
    # print('\nValidation set: Average loss(Coarse): {:.4f} \n'.format(coarse_validation_loss))
    print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(epoch, training_loss, 
        coarse_validation_loss, delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss, rmse_log_loss, 
        abs_relative_difference_loss, squared_relative_difference_loss))

def fine_validation(epoch, training_loss):
    fine_model.eval()
    fine_validation_loss = 0
    scale_invariant_loss = 0
    delta1_accuracy = 0
    delta2_accuracy = 0
    delta3_accuracy = 0
    rmse_linear_loss = 0
    rmse_log_loss = 0
    abs_relative_difference_loss = 0
    squared_relative_difference_loss = 0
    for batch_idx,data in enumerate(val_loader):
        # variable
        rgb, depth = torch.tensor(data['image'].cuda(), requires_grad = False), torch.tensor(data['depth'].cuda(), requires_grad = False)
        coarse_output = coarse_model(rgb.type(dtype))
        fine_output = fine_model(rgb.type(dtype), coarse_output.type(dtype))
        fine_validation_loss += custom_loss_function(fine_output, depth).item()
        # all error functions
        scale_invariant_loss += scale_invariant(fine_output, depth)
        delta1_accuracy += threeshold_percentage(fine_output, depth, 1.25)
        delta2_accuracy += threeshold_percentage(fine_output, depth, 1.25*1.25)
        delta3_accuracy += threeshold_percentage(fine_output, depth, 1.25*1.25*1.25)
        rmse_linear_loss += rmse_linear(fine_output, depth)
        rmse_log_loss += rmse_log(fine_output, depth)
        abs_relative_difference_loss += abs_relative_difference(fine_output, depth)
        squared_relative_difference_loss += squared_relative_difference(fine_output, depth)
    fine_validation_loss /= (batch_idx + 1)
    scale_invariant_loss /= (batch_idx + 1)
    delta1_accuracy /= (batch_idx + 1)
    delta2_accuracy /= (batch_idx + 1)
    delta3_accuracy /= (batch_idx + 1)
    rmse_linear_loss /= (batch_idx + 1)
    rmse_log_loss /= (batch_idx + 1)
    abs_relative_difference_loss /= (batch_idx + 1)
    squared_relative_difference_loss /= (batch_idx + 1)
    logger.scalar_summary("fine validation loss", fine_validation_loss, epoch)
    # print('\nValidation set: Average loss(Fine): {:.4f} \n'.format(fine_validation_loss))
    print('Epoch: {}    {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}      {:.4f}'.format(epoch, training_loss, 
        fine_validation_loss, delta1_accuracy, delta2_accuracy, delta3_accuracy, rmse_linear_loss, rmse_log_loss, 
        abs_relative_difference_loss, squared_relative_difference_loss))

folder_name = "models/" + args.model_folder
if not os.path.exists(folder_name): os.mkdir(folder_name)

print("********* Training the Coarse Model **************")
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.618)     (0.891)     (0.969)     (0.871)     (0.283)     (0.228)     (0.223)")

for epoch in range(1, args.epochs + 1):
    # print("********* Training the Coarse Model **************")
    training_loss = train_coarse(epoch)
    coarse_validation(epoch, training_loss)
    model_file = folder_name + "/" + 'coarse_model_' + str(epoch) + '.pth'
    if(epoch%10 == 0):
        torch.save(coarse_model.state_dict(), model_file)

coarse_model.eval() # stoping the coarse model to train.

print("********* Training the Fine Model ****************")
print("Epochs:     Train_loss  Val_loss    Delta_1     Delta_2     Delta_3    rmse_lin    rmse_log    abs_rel.  square_relative")
print("Paper Val:                          (0.611)     (0.887)     (0.971)     (0.907)     (0.285)     (0.215)     (0.212)")
for epoch in range(1, args.epochs + 1):
    # print("********* Training the Fine Model ****************")
    training_loss = train_fine(epoch)
    fine_validation(epoch, training_loss)
    model_file = folder_name + "/" + 'fine_model_' + str(epoch) + '.pth'
    if(epoch%5 == 0):
        torch.save(fine_model.state_dict(), model_file)
