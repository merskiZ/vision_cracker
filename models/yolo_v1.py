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
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchvision.transforms import transforms, ToTensor
from data.datasets import VOCDataset

from models.feature_extractors import AlexNet

use_cuda = torch.cuda.is_available()

class Yolo(nn.Module):
    def __init__(self):
        super(Yolo, self).__init__()

        self.alex = AlexNet(pretrained=True, finetune=False)

        # fc part
        self.fc_1 = nn.Linear(6 * 6 * 256, 9216)
        self.fc_2 = nn.Linear(9216, 7 * 7 * 30)

    def forward(self, x):

        x = x.view(-1, 1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = x.view((7, 7, 30))

        return x

# classification net use to do pretrain the Yolo net
class ClassificationNet(nn.Module):
    def __init__(self, num_classes):
        super(ClassificationNet, self).__init__()

        self.num_classes = num_classes
        self.fc = nn.Linear(7 * 7 * 30, num_classes)

    def forward(self, x):
        out = self.fc(x)
        return out

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

def train_step(data_loader, optimizer,
               preprocess, yolo,
               class_prediction, box_regress,
               pretrain, step):
    yolo.train()
    if pretrain:
        class_prediction.train()
    else:
        box_regress.train()
    for i, (image, label) in enumerate(data_loader):
        if use_cuda:
            image_tensor = Variable(preprocess(image)).cuda()
            label = Variable(label).cuda()
        else:
            image_tensor = Variable(preprocess(image))
            label = Variable(label)

        yolo_out = yolo.forward(image_tensor)
        optimizer.zero_grad()
        if pretrain:
            classes_pred = class_prediction.forward(yolo_out)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(classes_pred, label)
        else:
            box_preds = box_regress.forward(yolo_out)
            criterion = nn.MSELoss()
            loss = criterion()
        loss.backward()
        optimizer.step()
        print("Train step {}, loss {}".format(step, loss.item()))
        step += 1
    return step


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
          epochs,
          checkpoint=None,
          ckpt_save_epoch=100,
          pretrain=False,
          pretrain_epochs=1000):
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

    # if we want to pretrain the model, then we need to use a classification net
    # instead of a box regression net follows the yolo net
    class_prediction = None
    box_regress = None
    if pretrain:
        if use_cuda:
            box_regress = BoxRegressionNet(s,
                                           num_boxes=num_boxes,
                                           num_classes=num_classes).cuda()
        else:
            box_regress = BoxRegressionNet(s,
                                           num_boxes=num_boxes,
                                           num_classes=num_classes)
    else:
        if use_cuda:
            class_prediction = ClassificationNet(num_classes=num_classes).cuda()
        else:
            class_prediction = ClassificationNet(num_classes=num_classes)

    # image preprocessing
    preprocess = transforms.Compose([ToTensor()])

    # get optimizer ready
    if pretrain:
        optimizer = optim.Adam([yolo, class_prediction], lr=lr, betas=(beta1, 0.999))
    else:
        optimizer = optim.Adam([yolo, box_regress], lr=lr, betas=(beta1, 0.999))

    step = 1
    for epoch in range(epochs):
        step = train_step(data_loader, optimizer,
                          preprocess, yolo,
                          class_prediction, box_regress,
                          pretrain, step)

        if epochs % ckpt_save_epoch == 0:
            torch.save(yolo.state_dict(),
                       os.path.join(output_folder,
                                    'ckpt_yolo_{}.tar'.format(step)))
