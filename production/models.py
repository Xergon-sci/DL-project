'''
File    :   models.py
Time    :   2022/01/15 12:29:05
Author  :   Michiel Jacobs 
Version :   1.0
Contact :   michiel.jacobs@vub.be
License :   (C)Copyright 2022, Michiel Jacobs
'''

import torch
from torch import nn

class LeNet5(nn.Module):

    def __init__(self, n_features):
        super(LeNet5, self).__init__()

        # Convolutional feature extractor
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # Feed foreward network
        self.f6 = nn.Linear(in_features=120, out_features=84)
        # Customizable output features, allows us to work with different sets without refedining the model
        self.f7 = nn.Linear(in_features=84, out_features=n_features)

        # Define activation function and pooling layers, which are the same troughout the entire network
        self.tanh = nn.Tanh()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # RBF not available in pytorch, replacing with softmax which is good for classification
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.c1(x)
        x = self.tanh(x)
        x = self.pool(x) #s2

        x = self.c3(x)
        x = self.tanh(x)
        x = self.pool(x) #s4

        x = self.c5(x)
        x = self.tanh(x)

        x = torch.flatten(x, 1)

        x = self.f6(x)
        x = self.tanh(x)

        x = self.f7(x)
        x = self.softmax(x)
        return x

class LeNet5Variant(nn.Module):

    def __init__(self, n_features):
        super(LeNet5Variant, self).__init__()

        # Convolutional feature extractor
        self.c1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=5)
        self.c3 = nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5)
        self.c5 = nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5)

        # Feed foreward network
        self.f6 = nn.Linear(in_features=120, out_features=84)
        # Customizable output features, allows us to work with different sets without refedining the model
        self.f7 = nn.Linear(in_features=84, out_features=n_features)

        # Define activation function and pooling layers, which are the same troughout the entire network
        self.relu = nn.ReLU()
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)
        # RBF not available in pytorch, replacing with softmax which is good for classification
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.c1(x)
        x = self.relu(x)
        x = self.pool(x) #s2

        x = self.c3(x)
        x = self.relu(x)
        x = self.pool(x) #s4

        x = self.c5(x)
        x = self.relu(x)

        x = torch.flatten(x, 1)

        x = self.f6(x)
        x = self.relu(x)

        x = self.f7(x)
        x = self.softmax(x)
        return x