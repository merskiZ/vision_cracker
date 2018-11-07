"""
implementation of YOLO:
┌────────────┬────────────────────────┬───────────────────┐
│    Name    │        Filters         │ Output Dimension  │
├────────────┼────────────────────────┼───────────────────┤
│ Conv 1     │ 7 x 7 x 64, stride=2   │ 224 x 224 x 64    │
│ Max Pool 1 │ 2 x 2, stride=2        │ 112 x 112 x 64    │
│ Conv 2     │ 3 x 3 x 192            │ 112 x 112 x 192   │
│ Max Pool 2 │ 2 x 2, stride=2        │ 56 x 56 x 192     │
│ Conv 3     │ 1 x 1 x 128            │ 56 x 56 x 128     │
│ Conv 4     │ 3 x 3 x 256            │ 56 x 56 x 256     │
│ Conv 5     │ 1 x 1 x 256            │ 56 x 56 x 256     │
│ Conv 6     │ 1 x 1 x 512            │ 56 x 56 x 512     │
│ Max Pool 3 │ 2 x 2, stride=2        │ 28 x 28 x 512     │
│ Conv 7     │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 8     │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 9     │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 10    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 11    │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 12    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 13    │ 1 x 1 x 256            │ 28 x 28 x 256     │
│ Conv 14    │ 3 x 3 x 512            │ 28 x 28 x 512     │
│ Conv 15    │ 1 x 1 x 512            │ 28 x 28 x 512     │
│ Conv 16    │ 3 x 3 x 1024           │ 28 x 28 x 1024    │
│ Max Pool 4 │ 2 x 2, stride=2        │ 14 x 14 x 1024    │
│ Conv 17    │ 1 x 1 x 512            │ 14 x 14 x 512     │
│ Conv 18    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 19    │ 1 x 1 x 512            │ 14 x 14 x 512     │
│ Conv 20    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 21    │ 3 x 3 x 1024           │ 14 x 14 x 1024    │
│ Conv 22    │ 3 x 3 x 1024, stride=2 │ 7 x 7 x 1024      │
│ Conv 23    │ 3 x 3 x 1024           │ 7 x 7 x 1024      │
│ Conv 24    │ 3 x 3 x 1024           │ 7 x 7 x 1024      │
│ FC 1       │ -                      │ 4096              │
│ FC 2       │ -                      │ 7 x 7 x 30 (1470) │
└────────────┴────────────────────────┴───────────────────┘
"""

import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchvision.transforms import transforms, ToTensor
from data.voc_iterator import VOCDataset

use_cuda = torch.cuda.is_available()

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()
        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.conv_layer_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.conv_layer_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.conv_layer_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.maxpool_3 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_layer_7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_layer_8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_layer_9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_layer_10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_layer_11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_layer_12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_layer_13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.conv_layer_14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.conv_layer_15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.conv_layer_16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.maxpool_4 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_layer_17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv_layer_18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv_layer_19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.conv_layer_20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.conv_layer_21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv_layer_22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.conv_layer_23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.conv_layer_24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)

        # fc part
        self.fc_1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc_2 = nn.Linear(4096, 7 * 7 * 30)

    def forward(self, x):
        x = nn.LeakyReLU(self.conv_layer_1(x), 0.1)
        x = self.maxpool_1
        x = nn.LeakyReLU(self.conv_layer_2(x), 0.1)
        x = self.maxpool_2(x)
        x = nn.LeakyReLU(self.conv_layer_3(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_4(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_5(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_6(x), 0.1)
        x = self.maxpool_3(x)
        x = nn.LeakyReLU(self.conv_layer_7(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_8(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_9(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_10(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_11(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_12(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_13(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_14(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_15(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_16(x), 0.1)
        x = self.maxpool_4(x)
        x = nn.LeakyReLU(self.conv_layer_17(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_18(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_19(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_20(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_21(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_22(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_23(x), 0.1)
        x = nn.LeakyReLU(self.conv_layer_24(x), 0.1)

        x = x.view(-1, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = x.view((7, 7, 30))

        return x

class BoxRegressionNet(nn.Module):
    def __init__(self,
                 s,
                 num_classes,
                 num_boxes):
        super(BoxRegressionNet, self).__init__()
        self.s = s
        self.fc_regress = nn.Linear(30, num_boxes * 5 + num_classes)

    def forward(self, x):
        grid_preds = []
        for i in range(self.s):
            for j in range(self.s):
                grid_pred = self.fc_regress(x[i, j, :])
                grid_preds.append(grid_pred)
        return grid_preds

def train(input_folder,
          annotation_folder,
          annotation_map,
          output_folder,
          batch_size,
          num_classes,
          num_boxes,
          s,
          lr,
          beta1,
          epochs):
    # get data generator ready
    dataset = VOCDataset(image_folder=input_folder,
                         annotation_folder=annotation_folder,
                         annotation_map_file=annotation_map)
    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=True)
    # get nets ready
    yolo = None
    if use_cuda:
        yolo = Yolo().cuda()
    else:
        yolo = Yolo()

    box_regress = None
    if use_cuda:
        box_regress = BoxRegressionNet(s,
                                       num_boxes=num_boxes,
                                       num_classes=num_classes).cuda()
    else:
        box_regress = BoxRegressionNet(s,
                                       num_boxes=num_boxes,
                                       num_classes=num_classes)

    # image preprocessing
    preprocess = transforms.Compose([ToTensor()])

    # get optimizer ready
    optimizer = optim.Adam([yolo, box_regress], lr=lr, betas=(beta1, 0.999))

    for epoch in range(epochs):
        for i, (image, label) in enumerate(data_loader):
            image_tensor = preprocess(image)

            yolo_out = yolo(image_tensor)
            box_preds = box_regress(yolo_out)

            