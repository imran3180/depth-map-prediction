import matplotlib
import argparse
from PIL import Image

import torch
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from model import coarseNet, fineNet
import pdb
import numpy as np

import torchvision.transforms as transforms

parser = argparse.ArgumentParser(description='PyTorch depth prediction test run script')
parser.add_argument('--coarse_model_path', type=str, default='coarse_model.pth', metavar='F',
                    help='path of coarse_model')
parser.add_argument('--fine_model_path', type=str, default= 'fine_model.pth', metavar='F',
                    help='path of fine_model')
parser.add_argument('--path', type=str, default='sample_input2.jpg', metavar='D',
                    help="path of the image. By default it will run on the sample.jpg which comes with the repository")

args = parser.parse_args()

coarse_state_dict = torch.load(args.coarse_model_path, map_location=lambda storage, loc: storage)
fine_state_dict = torch.load(args.fine_model_path, map_location=lambda storage, loc: storage)

coarse_model = coarseNet()
fine_model = fineNet()

coarse_model.load_state_dict(coarse_state_dict)
fine_model.load_state_dict(fine_state_dict)
coarse_model.eval()
fine_model.eval()

rgb_data_transforms = transforms.Compose([
    transforms.Resize((228, 304)),
    transforms.ToTensor(),
])

input_for_plot_transforms = transforms.Compose([
    transforms.Resize((55, 74)), # for Input to be equal to output size
    transforms.ToTensor(),
])

image = Image.open(args.path)
image = np.transpose(image, (0, 1, 2))

image = Image.fromarray(image)
input_image = input_for_plot_transforms(image)
image = rgb_data_transforms(image)
image = image.view(1, 3, 228, 304)

coarse_output = coarse_model(image)
fine_output = fine_model(image, coarse_output)

plt.figure(1, figsize=(9, 3))
plt.subplot(131)
plt.gca().set_title('input')
plt.imshow(np.transpose(input_image, (1, 2, 0)), interpolation="nearest")
plt.subplot(132)
plt.gca().set_title('coarse_output')
plt.imshow(np.transpose(coarse_output[0][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")
plt.subplot(133)
plt.gca().set_title('fine_output')
plt.imshow(np.transpose(fine_output[0][0].detach().cpu().numpy(), (0, 1)), interpolation="nearest")
plt.suptitle('Depth Map Prediction of Input Image')
plt.show()
