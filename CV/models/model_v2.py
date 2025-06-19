#pass through CNN layers
#determine the kernel size and maxpooling
#pass through linear layers

import torch
import torch.nn as nn
import numpy as np
import random

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

#assumes an image size of 128x128, which is reasonable size of object detection / image classification
class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 8, 5) #in channel, out channel, kernel size
        self.bn1 = nn.BatchNorm2d(8)
        self.pool = nn.MaxPool2d(2, 2) #d
        self.conv2 = nn.Conv2d(8, 16, 5)
        self.bn2 = nn.BatchNorm2d(16)

        self.linear1 = nn.Linear(16 * 29 * 29, 256) #more uniform reduction
        self.linear2 = nn.Linear(256, 64)
        self.linear3 = nn.Linear(64, 10)

        self.leaky_relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(p=0.2)
    
    def forward(self, x):
        x = self.pool(self.leaky_relu(self.bn1(self.conv1(x))))
        x = self.pool(self.leaky_relu(self.bn2(self.conv2(x))))

        x = torch.flatten(x, 1)
        x = self.leaky_relu(self.linear1(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.linear2(x))
        x = self.dropout(x)
        x = self.leaky_relu(self.linear3(x))

        return x