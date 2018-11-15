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
from torch.autograd import Variable
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
                           default=32,
                           type=int,
                           help='')
    argparser.add_argument('-lr',
                           '--learning_rate',
                           default=5e-5,
                           type=float,
                           help='controls the learning rate for the optimizer')
    argparser.add_argument('--beta1',
                           default=0.99,
                           type=float,
                           help='beta1 value for adam')
    argparser.add_argument('--epoch',
                           default=100000,
                           type=int,
                           help='epoch for training')
    return argparser.parse_args()

gen_input_size = 64
randomer = random_generator(min=0.0, max=1.)
flip_threshold = 0.99
noise_mean = 0.
noise_std = 0.0005

randomer_low = random_generator(min=0.0, max=0.3)
randomer_high = random_generator(min=0.7, max=1.2)

transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02).cuda()
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02).cuda()
        m.bias.data.fill_(0).cuda()
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02).cuda()
        m.bias.data.fill_(0).cuda()

class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        # self.net = nn.Sequential(
        self.fc1 = nn.Linear(64, 8 * 8 * 1024)
        self.bn0 = nn.BatchNorm2d(8 * 8 * 1024)
        self.deconv1 = nn.ConvTranspose2d(1024, 512, 3, 1, 1, bias=False) # 16x16x1024
        self.bn1 = nn.BatchNorm2d(512)
        self.lkrl1 = nn.LeakyReLU(0.2)
        self.deconv2 = nn.ConvTranspose2d(512, 256, 3, 2, 1, 1, bias=False) # 32x32x64
        self.bn2 = nn.BatchNorm2d(256)
        self.lkrl2 = nn.LeakyReLU(0.2)
        self.deconv3 = nn.ConvTranspose2d(256, 256, 3, 2, 1, 1, bias=False) # 64x64x64
        self.bn3 = nn.BatchNorm2d(256)
        self.lkrl3 = nn.LeakyReLU(0.2)
        self.deconv4 = nn.ConvTranspose2d(256, 128, 3, 2, 1, 1, bias=False) # 128x128x64
        self.bn4 = nn.BatchNorm2d(128)
        self.lkrl4 = nn.LeakyReLU(0.2)
        self.deconv5 = nn.ConvTranspose2d(128, 3, 3, 2, 1, 1, bias=False)  # 256x256x3
        self.tanh = nn.Tanh()
        # self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.fc1(x)
        x = x.view(-1, 1024, 8, 8)
        # x = self.bn0(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = self.lkrl1(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = self.lkrl2(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = self.lkrl3(x)

        x = self.deconv4(x)
        x = self.bn4(x)
        x = self.lkrl4(x)

        x = self.deconv5(x)
        x = self.tanh(x)

        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv2d(3, 32, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.lkrl1 = nn.LeakyReLU(0.2)
        # 64 x 64 x 64
        self.conv2 = nn.Conv2d(32, 64, 3, 2, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.lkrl2 = nn.LeakyReLU(0.2, inplace=True)
        # 32 x 32 x 128
        self.conv3 = nn.Conv2d(64, 256, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.lkrl3 = nn.LeakyReLU(0.2, inplace=True)
        # 16 x 16 x 256
        self.conv4 = nn.Conv2d(256, 512, 3, 2, 1, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.lkrl4 = nn.LeakyReLU(0.2, inplace=True)
        # 8 x 8 x 512
        self.conv5 = nn.Conv2d(512, 512, 3, 2, 1, bias=False)
        self.bn5 = nn.BatchNorm2d(512)
        self.lkrl5 = nn.LeakyReLU(0.2, inplace=True)
        # # 4 x 4 x 1024
        # nn.Conv2d(1024, 1, 4, 1, 0, bias=False),
        self.conv6 = nn.Conv2d(512, 1, 4, 1, 0, bias=False)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lkrl1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.lkrl2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.lkrl3(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = self.lkrl4(x)

        x = self.conv5(x)
        x = self.bn5(x)
        x = self.lkrl5(x)

        x = self.conv6(x)
        x = self.sigmoid(x)

        return x


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

def generate_soft_labels(label, shape):
    """
    generate soft labels depends on the value of the hard labels.
    if label == 1, the range is (0.7, 1.2),
    if label == 0, the range is (0.0, 0.3)
    :param label:
    :return:
    """
    if label == 0:
        return torch.empty(shape[0]).uniform_(0.0, 0.1)
    elif label == 1:
        return torch.empty(shape[0]).uniform_(0.9, 1.0)

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
    if ngpu == 0:
        generator = Generator()
    else:
        print("generator initialized with cuda")
        generator = Generator().cuda()
    generator.apply(weights_init)
    print(generator)

    # initialize discriminator
    if ngpu == 0:
        discriminator = Discriminator()
    else:
        print("discriminator initialized with cuda")
        discriminator = Discriminator().cuda()
    discriminator.apply(weights_init)
    print(discriminator)

    # setup loss computation
    criterion = nn.BCELoss()

    # TODO: replace uniformly sampled noise to Gaussian distribution
    # fixed_noise = torch.randn(args.batch_size, gen_input_size, 1, 1, device=device)
    fixed_noise = torch.zeros(args.batch_size, gen_input_size)
    if ngpu == 0:
        gaussian_gen = DynamicGaussianNoise(fixed_noise.shape,
                                            mean=noise_mean, std=noise_std)
    else:
        gaussian_gen = DynamicGaussianNoise(fixed_noise.shape,
                                            mean=noise_mean, std=noise_std).cuda(device=device)
    fixed_noise = gaussian_gen.forward(fixed_noise)
    # fixed_noise = fixed_noise.view(args.batch_size, 8, 8)
    # fixed_noise = fixed_noise.unsqueeze(-1)
    # fixed_noise = fixed_noise.unsqueeze(1)

    real_label = 0
    fake_label = 1

    # setup optimizer
    optim_g = optim.Adam(generator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))
    optim_d = optim.Adam(discriminator.parameters(), lr=args.learning_rate, betas=(args.beta1, 0.999))

    for epoch in range(args.epoch):
        # try:
        for i, data in enumerate(data_loader):
            # train real
            discriminator.zero_grad()
            real_cpu = data[0].to(device)
            batch_size = real_cpu.size(0)
            # label = torch.full((batch_size,),
            #                    generate_soft_labels(flip_label(real_label)),
            #                    device=device)
            label = generate_soft_labels(flip_label(real_label), (batch_size, 1)).to(device)

            output = discriminator.forward(real_cpu)
            output = output.squeeze(-1)
            output = output.squeeze(-1)
            loss_real = criterion(output, label)
            # loss_real.backward()
            D_x = output.mean().item()

            # train fake from generator and discriminator
            # noise = torch.randn(batch_size, gen_input_size, 1, 1, device=device)
            # TODO: replaced with gaussian noise
            noise = Variable(torch.zeros(args.batch_size, gen_input_size))
            noise = gaussian_gen.forward(noise)
            # noise = noise.view(args.batch_size, 8, 8)
            # noise = noise.unsqueeze(-1)
            # noise = noise.unsqueeze(1)
            noise = Variable(noise)

            fake = generator(noise)
            # label.fill_(generate_soft_labels(flip_label(fake_label)))
            label = generate_soft_labels(flip_label(fake_label), (batch_size, 1))
            output = discriminator(fake.detach())
            output = output.squeeze(-1)
            output = output.squeeze(-1)
            loss_fake = criterion(output, label)
            # loss_fake.backward()
            D_G_z1 = output.mean().item()
            d_loss = loss_fake + loss_real
            d_loss.backward()
            optim_d.step()

            # update generator
            # label.fill_(generate_soft_labels(real_label))
            label = generate_soft_labels(flip_label(real_label), (batch_size, 1))
            output = discriminator(fake)
            output = output.squeeze(-1)
            output = output.squeeze(-1)
            loss_generator = criterion(output, label)
            loss_generator.backward()
            D_G_z2 = output.mean().item()
            optim_g.step()

            print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                  % (epoch, args.epoch, i, len(data_loader),
                     d_loss.item(), loss_generator.item(), D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                                  '%s/real_samples.png' % args.output,
                                  normalize=True)
                fake = generator(fixed_noise)
                vutils.save_image(fake.detach(),
                                  '%s/fake_samples_epoch_%03d_batch_%04d.png' % (args.output, epoch, i),
                                  normalize=True)

        # except Exception as e:
        #     print(e)
        #     continue
        # do checkpointing
        torch.save(generator.state_dict(), '%s/netG_epoch_%d.pth' % (args.output, epoch))
        torch.save(discriminator.state_dict(), '%s/netD_epoch_%d.pth' % (args.output, epoch))

if __name__ == '__main__':
    main()