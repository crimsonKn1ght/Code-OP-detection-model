import torch.nn as nn
from torchvision import models


class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=1):
        super().__init__()
        resnet = models.resnet34(weights=None)
        resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

        self.init = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.pool = resnet.maxpool
        self.l1 = resnet.layer1
        self.l2 = resnet.layer2

    def forward(self, x):
        x0 = self.init(x)
        x1 = self.pool(x0)
        x2 = self.l1(x1)
        x3 = self.l2(x2)
        return x3, x2, x0
