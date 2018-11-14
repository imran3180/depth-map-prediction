import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
import pdb


class coarseNet(nn.Module):
    def __init__(self,init_weights=True):
        super(coarseNet, self).__init__()
        self.conv1 = nn.Conv2d(3,128,kernel_size=3,stride=4,padding=0)
        self.conv2 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.conv3 = nn.Conv2d(128,128,kernel_size=3,stride=1,padding=0)
        self.fc1 = nn.Linear(94208, 4096)
        self.fc2 = nn.Linear(4096, 4070)
        if init_weights:
            self._initialize_weights()


    def forward(self, x):                       # [n, c,  H,   W ]
                                                # [8, 3, 228, 304]
        x = self.conv1(x)                       # [8, 128, 57, 76]
        x = F.max_pool2d(x, 3, stride=2)        # [8, 128, 28, 37]
        x = self.conv2(x)                       # [8, 128, 26, 35]
        x = F.max_pool2d(x, 2, stride =1)       # [8, 128, 25, 34]
        x = self.conv3(x)                       # [8, 128, 23, 32]
        x = x.view(x.size(0), -1)               # [8, 94208]
        x = F.relu(self.fc1(x))                 # [8, 4096]
        x = F.relu(self.fc2(x))                 # [8, 4070]     => 55x74 = 4070
        x = x.view(-1, 1, 55, 74)
        return x

    # Pre-train Imagenet Model ??
    # Why random guassian model.    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
                
                
class fineNet(nn.Module):
    def __init__(self, init_weights=True):
        super(fineNet, self).__init__()
        self.conv1 = nn.Conv2d(3,63,kernel_size=9,stride=2,padding=1)
        self.conv2 = nn.Conv2d(64,64,kernel_size=5,stride=1,padding=1)
        self.conv3 = nn.Conv2d(64,1,kernel_size=1,stride=1,padding=1)
        if init_weights:
            self._initialize_weights()


    def forward(self, x, y):
                                                # [8, 3, 228, 304]
        x = self.conv1(x)                       # [8, 63, 111, 149]
        x = F.max_pool2d(x, 3, stride=2)        # [8, 63, 55, 74]
        x = torch.cat((x,y),1)                  # x - [8, 63, 55, 74] y - [8, 1, 55, 74] => x = [8, 64, 55, 74]
        x = self.conv2(x)                       # [8, 64, 53, 72]
        x = self.conv3(x)                       # [8, 1, 55, 74]
        return x
    
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()