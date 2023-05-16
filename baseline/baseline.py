import torch
import torch.nn as nn
import numpy as np
from torchsummary import summary

class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        layer1 = [
            nn.Conv2d(1, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(32)
        ]

        layer2 = [
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(64)
        ]

        layer3 = [
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(128)
        ]

        layer4 = [
            nn.Conv2d(128, 128, kernel_size=4, dilation=2, stride=2, padding=3, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(128)
        ]

        layer5 = [
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True),
            nn.ReLU(True),
            nn.BatchNorm2d(32)
        ]

        layer6 = nn.Conv2d(32, 2, kernel_size=1, stride=1, padding=0, bias=True)

        layer7 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)

        self.encoder = nn.Sequential(*layer1, *layer2, *layer3, *layer4)
        self.decoder = nn.Sequential(*layer5, layer6, layer7)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    

if __name__ == '__main__':
    model = BaselineModel()
    summary(model, (1, 256, 256))