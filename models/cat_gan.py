"""
Cat Gan in pytorch.

for convtranspose2d, H_out = (H_in - 1) * stride[0] - 2 * padding[0] + kernel_size[0] + output_padding[0]
                     W_out = (W_in - 1) * stride[1] - 2 * padding[1] + kernel_size[1] + output_padding[1]

for conv2d,          H_out = floor((H_in + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1) / stride[0] + 1)
                     W_out = floor((W_in + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1) / stride[0] + 1)
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.datasets import ImageFolder
from torchvision import transforms
import torchvision.utils as vutils
import pandas

from libs.utilities import Utilities
from libs.math_utilities import random_generator, DynamicGaussianNoise

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--train_folder',
                           help='The data folder for training data')
    argparser.add_argument('--valid_folder',
                           help='The data folder for validation set')
    argparser.add_argument('--output',
                           help='The output folder')
    argparser.add_argument('--ngpu', default=0,
                           type=int,
                           help='')
    argparser.add_argument('-bn', '--batch_size',
                           default=4,
                           type=int,
                           help='')
    argparser.add_argument('-lr',
                           '--learning_rate',
                           default=2e-4,
                           type=float,
                           help='controls the learning rate for the optimizer')
    argparser.add_argument('--beta1',
                           default=0.9,
                           type=float,
                           help='beta1 value for adam')
    argparser.add_argument('--epoch',
                           default=10,
                           type=int,
                           help='epoch for training')
    return argparser.parse_args()

image_size = 128
gen_input_size = 1024
randomer = random_generator(min=0.0, max=1.)
flip_threshold = 0.95

randomer_low = random_generator(min=0.0, max=0.3)
randomer_high = random_generator(min=0.7, max=1.2)

transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)

class Generator(nn.Module):
    def __init__(self, ngpu=0):
        super(Generator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            # add VAE part
            ######################################
            nn.Conv2d(image_size, 64, 3, stride=1, padding=1, bias=False), # 128x128x64
            nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False), # 64x64x128
            nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False), # 32x32x256
            nn.Conv2d(256, 512, 3, stride=2, padding=1, bias=False), # 16x16x512
            nn.Conv2d(512, 1024, 3, stride=4, padding=1, bias=False), # 4x4x1024
            nn.Conv2d(1024, 1024, 3, stride=4, padding=1, bias=False), # 1x1x1024
            #####################################

            nn.ConvTranspose2d(gen_input_size, 1024, 4, 1, bias=False), # 4x4x1024
            nn.BatchNorm2d(1024),
            nn.ConvTranspose2d(1024, 512, 5, 2, 2, 1, bias=False), # 8x8x512
            nn.BatchNorm2d(512),
            nn.ConvTranspose2d(512, 256, 5, 2, 2, 1, bias=False), # 16x16x256
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 128, 5, 2, 2, 1, bias=False), # 32x32x128
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 64, 5, 2, 2, 1, bias=False),  # 64x64x64
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 3, 5, 2, 2, 1, bias=False),  # 128x128x32
            # nn.BatchNorm2d(32),
            # nn.ConvTranspose2d(32, 3, 5, 2, 2, 1, bias=False),  # 256x256x3
            nn.Tanh()
        )

    def forward(self, x):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


class Discriminator(nn.Module):
    def __init__(self, ngpu=0):
        super(Discriminator, self).__init__()
        self.ngpu = ngpu
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 64 x 64 x 64
            nn.Conv2d(32, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.2, inplace=True),
            # 32 x 32 x 128
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            # 16 x 16 x 256
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            # 8 x 8 x 512
            nn.Conv2d(256, 512, 4, 4, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            # # 4 x 4 x 1024
            # nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
            nn.Conv2d(512, 1, 2, 1, 0, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):

        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.net, x, range(self.ngpu))
        else:
            output = self.net(x)
        return output


def flip_label(label):
    """
    flip the label from 1/0 to 0/1 if the random number > 0.5
    :param label:
    :return:
    """
    flip_num = randomer.__next__()
    if flip_num > flip_threshold:
        return 1 - label
    else:
        return label

def generate_soft_labels(label):
    """
    generate soft labels depends on the value of the hard labels.
    if label == 1, the range is (0.7, 1.2),
    if label == 0, the range is (0.0, 0.3)
    :param label:
    :return:
    """
    if label == 0:
        return randomer_low.__next__()
    elif label == 1:
        return randomer_high.__next__()

def main():
    args = parse_arguments()
    ngpu = args.ngpu
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")

    cat_dog_set = ImageFolder(args.train_folder, transform=transform)
    data_loader = torch.utils.data.DataLoader(cat_dog_set,
                                              batch_size=args.batch_size,
                                              shuffle=True,
                                              num_workers=2)

    # initialize generator net
    generator = Generator(ngpu).to(device)
    generator.apply(weights_init)
    print(generator)

    # initialize discriminator
    discriminator = Discriminator(ngpu).to(device)
    discriminator.apply(weights_init)
    print(discriminator)

    # setup loss computation
    criterion = nn.BCELoss()

    # TODO: replace uniformly sampled noise to Gaussian distribution
    # fixed_noise = torch.randn(args.batch_size, gen_input_size, 1, 1, device=device)
    # fixed_noise = torch.zeros(args.batch_size, gen_input_size)
    fixed_noise = torch.zeros(args.batch_size, image_size, image_size)
    gaussian_gen = DynamicGaussianNoise(fixed_noise.shape, device)
    fixed_noise = gaussian_gen.forward(fixed_noise)
    fixed_noise = fixed_noise.unsqueeze(-1)
    # fixed_noise = fixed_noise.unsqueeze(-1)

    real_label = 0
    fake_label = 1

    # setup optimizer
    optim_g = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    for epoch in range(args.epoch):
        try:
            for i, data in enumerate(data_loader):
                # train real
                discriminator.zero_grad()
                real_cpu = data[0].to(device)
                batch_size = real_cpu.size(0)
                label = torch.full((batch_size,), generate_soft_labels(flip_label(real_label)), device=device)

                output = discriminator(real_cpu)
                loss_real = criterion(output, label)
                loss_real.backward()
                D_x = output.mean().item()

                # train fake from generator and discriminator
                # noise = torch.randn(batch_size, gen_input_size, 1, 1, device=device)
                # TODO: replaced with gaussian noise
                # noise = torch.zeros(args.batch_size, gen_input_size)
                noise = torch.zeros(args.batch_size, image_size, image_size)
                noise = gaussian_gen.forward(noise)
                noise = noise.unsqueeze(-1)
                # noise = noise.unsqueeze(-1)

                fake = generator(noise)
                label.fill_(generate_soft_labels(flip_label(fake_label)))
                output = discriminator(fake.detach())
                loss_fake = criterion(output, label)
                loss_fake.backward()
                D_G_z1 = output.mean().item()
                d_loss = loss_fake + loss_real
                optim_d.step()

                # update generator
                label.fill_(generate_soft_labels(real_label))
                output = discriminator(fake)
                loss_generator = criterion(output, label)
                loss_generator.backward()
                D_G_z2 = output.mean().item()
                optim_g.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, args.epoch, i, len(data_loader),
                         d_loss.item(), loss_generator.item(), D_x, D_G_z1, D_G_z2))
                if i % 20 == 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % args.output,
                                      normalize=True)
                    fake = generator(fixed_noise)
                    vutils.save_image(fake.detach(),
                                      '%s/fake_samples_epoch_%03d_batch_%04d.png' % (args.output, epoch, i),
                                      normalize=True)
        except Exception as e:
            print(e, 'to the next epoch')
        # do checkpointing
        torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.output, epoch))
        torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.output, epoch))

if __name__ == '__main__':
    main()