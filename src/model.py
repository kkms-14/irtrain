"""
@author: Zongyi Li
This file is the Fourier Neural Operator for 2D problem such as the Darcy Flow discussed in Section 5.2 in the [paper](https://arxiv.org/pdf/2010.08895.pdf).
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.utils.data import DataLoader, random_split
import matplotlib.pyplot as plt

import operator
from functools import reduce
from functools import partial

from timeit import default_timer



import time

torch.manual_seed(0)
np.random.seed(0)


################################################################
# fourier layer
################################################################
class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super(SpectralConv2d, self).__init__()

        """
        2D Fourier layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.scale = (1 / (in_channels * out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(self.scale * torch.rand(in_channels, out_channels, self.modes1, self.modes2, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        # (batch, in_channel, x,y ), (in_channel, out_channel, x,y) -> (batch, out_channel, x,y)
        return torch.einsum("bixy,ioxy->boxy", input, weights)

    def forward(self, x):
        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft2(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-2), x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)
        out_ft[:, :, :self.modes1, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, :self.modes1, :self.modes2], self.weights1)
        out_ft[:, :, -self.modes1:, :self.modes2] = \
            self.compl_mul2d(x_ft[:, :, -self.modes1:, :self.modes2], self.weights2)

        #Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)))
        return x

class ConvBlock(nn.Module):
    def __init__(self, width, modes1, modes2):
        super(ConvBlock, self).__init__()

        self.width = width
        self.modes1 = modes1 #Number of Fourier modes to multiply, at most floor(N/2) + 1
        self.modes2 = modes2

        self.conv = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        self.w = nn.Conv2d(self.width, self.width, 1)

    def forward(self, x):
        x1 = self.conv(x)
        x2 = self.w(x)
        x = x1 + x2
        x = F.gelu(x)
        return x


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, neck):
        super(FNO2d, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, x=s, y=s, c=3)
        output: the solution 
        output shape: (batchsize, x=s, y=s, c=1)
        """

        self.modes1 = modes1
        self.modes2 = modes2
        self.width = width
        self.neck = neck
        self.padding = 9 # pad the domain if input is non-periodic
        self.fc0 = nn.Linear(3, self.width) # input channel is 3: (a(x, y), x, y)

        # self.conv0 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv1 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv2 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.conv3 = SpectralConv2d(self.width, self.width, self.modes1, self.modes2)
        # self.w0 = nn.Conv2d(self.width, self.width, 1)
        # self.w1 = nn.Conv2d(self.width, self.width, 1)
        # self.w2 = nn.Conv2d(self.width, self.width, 1)
        # self.w3 = nn.Conv2d(self.width, self.width, 1)

        self.block_0_0 = ConvBlock(self.width, self.modes1, self.modes2)
        # self.block_0_1 = ConvBlock(self.width, self.modes1, self.modes2)
        # self.block_0_2 = ConvBlock(self.width, self.modes1, self.modes2)
        # self.block_0_3 = ConvBlock(self.width, self.modes1, self.modes2)


        self.fc1 = nn.Linear(self.width, self.neck)
        self.fc2 = nn.Linear(self.neck, 1)

    def forward(self, x):
        grid = self.get_grid(x.shape, x.device)
        x = torch.cat((x, grid), dim=-1)
        x = self.fc0(x)
        x = x.permute(0, 3, 1, 2)
        x = F.pad(x, [0,self.padding, 0,self.padding])

        # x1 = self.conv0(x)
        # x2 = self.w0(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv1(x)
        # x2 = self.w1(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv2(x)
        # x2 = self.w2(x)
        # x = x1 + x2
        # x = F.gelu(x)

        # x1 = self.conv3(x)
        # x2 = self.w3(x)
        # x = x1 + x2

        x0 = self.block_0_0(x)
        # x0 = self.block_0_1(x0)
        # x0 = self.block_0_2(x0)
        # x0 = self.block_0_3(x0)

        x = x0[..., :-self.padding, :-self.padding]
        x = x.permute(0, 2, 3, 1)
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)
        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x, size_y = shape[0], shape[1], shape[2]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1, 1).repeat([batchsize, 1, size_y, 1])
        gridy = torch.tensor(np.linspace(0, 1, size_y), dtype=torch.float)
        gridy = gridy.reshape(1, 1, size_y, 1).repeat([batchsize, size_x, 1, 1])
        return torch.cat((gridx, gridy), dim=-1).to(device)

class UNet(nn.Module):
    def __init__(self, in_channels , out_channels):
        super(UNet, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels

        # contracting path (encoder)
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.relu4 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.relu5 = nn.ReLU(inplace=True)
        self.conv6 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        self.relu6 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv7 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.relu7 = nn.ReLU(inplace=True)
        self.conv8 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu8 = nn.ReLU(inplace=True)

        # expanding path (decoder)
        self.upconv1 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.conv9 = nn.Conv2d(512, 256, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm2d(256)
        self.relu9 = nn.ReLU(inplace=True)
        self.conv10 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm2d(256)
        self.relu10 = nn.ReLU(inplace=True)

        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.conv11 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm2d(128)
        self.relu11 = nn.ReLU(inplace=True)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm2d(128)
        self.relu12 = nn.ReLU(inplace=True)

        self.upconv3 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.conv13 = nn.Conv2d(128, 64, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm2d(64)
        self.relu13 = nn.ReLU(inplace=True)
        self.conv14 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm2d(64)
        self.relu14 = nn.ReLU(inplace=True)

        self.conv15 = nn.Conv2d(64, out_channels, kernel_size=1)

    def forward(self, x):
        # contracting path (encoder)
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv2(x1)
        x1 = self.bn2(x1)
        x1 = self.relu2(x1)
        x2 = self.pool1(x1)

        x3 = self.conv3(x2)
        x3 = self.bn3(x3)
        x3 = self.relu3(x3)
        x3 = self.conv4(x3)
        x3 = self.bn4(x3)
        x3 = self.relu4(x3)
        x4 = self.pool2(x3)

        x5 = self.conv5(x4)
        x5 = self.bn5(x5)
        x5 = self.relu5(x5)
        x5 = self.conv6(x5)
        x5 = self.bn6(x5)
        x5 = self.relu6(x5)
        x6 = self.pool3(x5)

        x7 = self.conv7(x6)
        x7 = self.bn7(x7)
        x7 = self.relu7(x7)
        x7 = self.conv8(x7)
        x7 = self.bn8(x7)
        x7 = self.relu8(x7)


        # expanding path (decoder)
        x = self.upconv1(x7)
        # x5 = x5[:, :, :x.size(2), :x.size(3)]
        x = F.pad(x, (0, 0, x5.size(2)-x.size(2), 0), mode='constant', value=0)
        x = F.pad(x, (0 , x5.size(3)-x.size(3)), mode='constant', value=0)
        x = torch.cat([x5, x], dim=1)
        x = self.conv9(x)
        x = self.bn9(x)
        x = self.relu9(x)
        x = self.conv10(x)
        x = self.bn10(x)
        x = self.relu10(x)

        x = self.upconv2(x)
        # x3 = x3[:, :, :x.size(2), :x.size(3)]
        x = F.pad(x, (0, 0, x3.size(2)-x.size(2), 0), mode='constant', value=0)
        x = F.pad(x, (0 , x3.size(3)-x.size(3)), mode='constant', value=0)
        x = torch.cat([x3, x], dim=1)
        x = self.conv11(x)
        x = self.bn11(x)
        x = self.relu11(x)
        x = self.conv12(x)
        x = self.bn12(x)
        x = self.relu12(x)

        x = self.upconv3(x)
        # x1 = x1[:, :, :x.size(2), :x.size(3)]
        x = F.pad(x, (0, 0, x1.size(2)-x.size(2), 0), mode='constant', value=0)
        x = F.pad(x, (0 , x1.size(3)-x.size(3)), mode='constant', value=0)
        x = torch.cat([x1, x], dim=1)
        x = self.conv13(x)
        x = self.bn13(x)
        x = self.relu13(x)
        x = self.conv14(x)
        x = self.bn14(x)
        x = self.relu14(x)

        x = self.conv15(x)

        print(x.size())


        return x