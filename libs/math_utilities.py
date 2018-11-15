import random
import torch
import torch.nn as nn
from torch.autograd import Variable

def random_generator(min=0.0, max=0.3):
    base_num = 1000000
    while 1:
        yield random.randrange(min * base_num, max * base_num) / float(base_num)


class DynamicGaussianNoise(nn.Module):
    # TODO: add device selection support
    def __init__(self, shape, mean=0., std=.05):
        super(DynamicGaussianNoise, self).__init__()
        self.noise = Variable(torch.zeros(shape[0], shape[1]))
        self.std = std
        self.mean = mean

    def forward(self, x):
        if not self.training:
            return x

        self.noise.data.normal_(self.mean, std=self.std)
        return x + self.noise