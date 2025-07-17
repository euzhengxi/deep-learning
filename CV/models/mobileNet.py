#pass through CNN layers
#determine the kernel size and maxpooling
#pass through linear layers

import torch
import torch.nn as nn
import numpy as np
import random
from torchvision.models import mobilenet_v3_small, MobileNet_V3_Small_Weights

# Set the seed for reproducibility
seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

class MobileNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = mobilenet_v3_small()

        in_features = self.model.classifier[0].in_features
        num_classes = 10
        self.model.classifier = nn.Sequential(
            nn.Linear(in_features, 128), 
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes),
        )
    
    def forward(self, x):
        x = self.model.features(x)
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.model.classifier(x)

        return x
        
    
