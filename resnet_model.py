import torch
import torch.nn as nn
from torchvision import models

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = models.resnet50(pretrained=False)
        num_ftrs = self.network.fc.in_features
        self.network.fc = nn.Linear(num_ftrs, 6)

    def forward(self, xb):
        return torch.sigmoid(self.network(xb))
