#pass through CNN layers
#determine the kernel size and maxpooling
#pass through linear layers

import torch
import torch.nn as nn
import numpy as np
import random
import timm

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class DeiT(nn.Module):
    def __init__(self):
        super().__init__()
        num_classes = 10
        self.model = timm.create_model('deit_tiny_patch16_224', num_classes=num_classes, pretrained=True, drop_rate=0.2)

    def forward(self, x):
        x = self.model(x)
        return x
        
    
