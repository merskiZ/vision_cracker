import torch
import torch.nn as nn
from torchvision.models import alexnet

class AlexNet(nn.Module):
    """
    AlexNet pretrained,
    """
    def __init__(self, pretrained=True, finetune=False):
        super(AlexNet, self).__init__()
        self.alex = alexnet(pretrained=pretrained)
        for param in self.alex.parameters():
            param.requires_grad = finetune
        self.maxpool1 = nn.MaxPool2d(3, 2)

    def forward(self, x):
        x = self.alex.features[:-1](x)
        output = self.maxpool1(x)
        return output