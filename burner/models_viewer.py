import torch
import torch.nn as nn
from torchvision.models import alexnet

class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.alex = alexnet(pretrained=True)
        for param in self.alex.parameters():
            param.requires_grad = False
        self.maxpool1 = nn.MaxPool2d(3, 2)

    def forward(self, x):
        x = self.alex.features[:-1](x)
        output = self.maxpool1(x)
        return output


if __name__ == '__main__':
    alex = AlexNet()

    test_input = torch.zeros((1, 3, 227, 227))

    print(alex.forward(test_input).shape)