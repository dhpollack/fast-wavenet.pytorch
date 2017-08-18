import torch
import torch.nn as nn
from torch.autograd import Variable
from wavenet.layers import *

'''Simple test model to use Conv1dExt module

'''

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = Conv1dExt(in_channels=1,
                               out_channels=4,
                               kernel_size=1,
                               bias=False)
        self.conv2 = Conv1dExt(in_channels=1,
                               out_channels=4,
                               kernel_size=1,
                               bias=False)
        self.conv3 = Conv1dExt(in_channels=4,
                               out_channels=4,
                               kernel_size=1,
                               bias=False)
        self.conv4 = Conv1dExt(in_channels=4,
                               out_channels=2,
                               kernel_size=1,
                               bias=True)
        self.conv1.input_tied_modules = [self.conv3]
        self.conv1.output_tied_modules = [self.conv2]
        self.conv2.input_tied_modules = [self.conv3]
        self.conv2.output_tied_modules = [self.conv1]
        self.conv3.input_tied_modules = [self.conv4]

    def forward(self, x):
        x1 = self.conv1(x)
        x2 = self.conv2(x)
        x = nn.functional.relu(x1 + x2)
        x = nn.functional.relu(self.conv3(x))
        x = nn.functional.relu(self.conv4(x))
        return x
