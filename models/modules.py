import torch
import torch.nn as nn

def conv2d_maxpool(kernel_size,
                   in_channels,
                   out_channels,
                   conv_stride,
                   conv_padding,
                   pool_stride,
                   pool_kernel_size):
    conv = nn.Conv2d(in_channels, out_channels,
                          kernel_size, stride=conv_stride,
                          padding=conv_padding)
    maxpool = nn.MaxPool2d(kernel_size=pool_kernel_size,
                           stride=pool_stride)
    return nn.Sequential([conv, maxpool])


