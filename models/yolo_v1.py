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
import argparse
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data.dataloader import DataLoader
import torch.optim as optim
from torchvision.transforms import transforms, ToTensor, Resize

from tensorboardX import SummaryWriter

from data.datasets import VOCDataset
from data.data_utilities import PadCollate

use_cuda = torch.cuda.is_available()
writer = SummaryWriter(log_dir='/tmp/log_dir')

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-if', '--input_folder',
                           type=str,
                           required=True,
                           help='The input image folder')
    argparser.add_argument('-am', '--annotation_map',
                           type=str,
                           required=True,
                           help='The annotation map to map')
    argparser.add_argument('-l', '--log_dir',
                           type=str,
                           default='/tmp/vision_cracker_log',
                           help='the place to save checkpoint and ')
    argparser.add_argument('--batch_size',
                           default=2,
                           type=int,
                           help='The number of batch sizes that is extracted from  dataset')
    argparser.add_argument('--resize_shape',
                           default=448,
                           type=int,
                           help='The input shape after resizing')
    argparser.add_argument('--num_classes',
                           default=20,
                           type=int,
                           help='The number of classes that in the dataset')
    argparser.add_argument('--num_boxes',
                           default=2,
                           type=int,
                           help='The number of boxes for each grid prediction')
    argparser.add_argument('--learning_rate',
                           default=0.0001,
                           type=float,
                           help='The learning rate for the adam optimizer')
    argparser.add_argument('--beta1',
                           default=0.999,
                           type=float,
                           help='The beta value to control the momentum in adam optimizer')
    argparser.add_argument('--lambda_coord',
                           default=5.,
                           type=float,
                           help='lambda weight for the coordinate loss')
    argparser.add_argument('--lambda_noobj',
                           default=.5,
                           type=float,
                           help='lambda weight for the confidence loss')
    argparser.add_argument('--epochs',
                           default=10000,
                           type=int,
                           help='The epochs we want to run for training')
    return argparser.parse_args()


class Yolo(nn.Module):
    def __init__(self, num_classes, num_boxes, batch_size):
        super(Yolo, self).__init__()

        self.batch_size = batch_size
        self.depth_per_cell = num_boxes * 5 + num_classes

        self.conv_layer_1 = nn.Conv2d(in_channels=3, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.leaky_relu_1 = nn.LeakyReLU(0.1)
        self.maxpool_1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_2 = nn.Conv2d(in_channels=64, out_channels=192, kernel_size=3, padding=1)
        self.leaky_relu_2 = nn.LeakyReLU(0.1)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv_layer_3 = nn.Conv2d(in_channels=192, out_channels=128, kernel_size=1)
        self.leaky_relu_3 = nn.LeakyReLU(0.1)
        self.conv_layer_4 = nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1)
        self.leaky_relu_4 = nn.LeakyReLU(0.1)
        self.conv_layer_5 = nn.Conv2d(in_channels=256, out_channels=256, kernel_size=1)
        self.leaky_relu_5 = nn.LeakyReLU(0.1)
        self.conv_layer_6 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1)
        self.leaky_relu_6 = nn.LeakyReLU(0.1)
        self.maxpool_3 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_layer_7 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.leaky_relu_7 = nn.LeakyReLU(0.1)
        self.conv_layer_8 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.leaky_relu_8 = nn.LeakyReLU(0.1)
        self.conv_layer_9 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.leaky_relu_9 = nn.LeakyReLU(0.1)
        self.conv_layer_10 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.leaky_relu_10 = nn.LeakyReLU(0.1)
        self.conv_layer_11 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.leaky_relu_11 = nn.LeakyReLU(0.1)
        self.conv_layer_12 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.leaky_relu_12 = nn.LeakyReLU(0.1)
        self.conv_layer_13 = nn.Conv2d(in_channels=512, out_channels=256, kernel_size=1)
        self.leaky_relu_13 = nn.LeakyReLU(0.1)
        self.conv_layer_14 = nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding=1)
        self.leaky_relu_14 = nn.LeakyReLU(0.1)
        self.conv_layer_15 = nn.Conv2d(in_channels=512, out_channels=512, kernel_size=1)
        self.leaky_relu_15 = nn.LeakyReLU(0.1)
        self.conv_layer_16 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.leaky_relu_16 = nn.LeakyReLU(0.1)
        self.maxpool_4 = nn.MaxPool2d(stride=2, kernel_size=2)
        self.conv_layer_17 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.leaky_relu_17 = nn.LeakyReLU(0.1)
        self.conv_layer_18 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.leaky_relu_18 = nn.LeakyReLU(0.1)
        self.conv_layer_19 = nn.Conv2d(in_channels=1024, out_channels=512, kernel_size=1)
        self.leaky_relu_19 = nn.LeakyReLU(0.1)
        self.conv_layer_20 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=1)
        self.leaky_relu_20 = nn.LeakyReLU(0.1)
        self.conv_layer_21 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.leaky_relu_21 = nn.LeakyReLU(0.1)
        self.conv_layer_22 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1, stride=2)
        self.leaky_relu_22 = nn.LeakyReLU(0.1)
        self.conv_layer_23 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.leaky_relu_23 = nn.LeakyReLU(0.1)
        self.conv_layer_24 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding=1)
        self.leaky_relu_24 = nn.LeakyReLU(0.1)

        # fc part
        self.fc_1 = nn.Linear(7 * 7 * 1024, 4096)
        self.fc_2 = nn.Linear(4096, 7 * 7 * self.depth_per_cell)

    def forward(self, x):
        x = self.leaky_relu_1(self.conv_layer_1(x))
        x = self.maxpool_1(x)
        x = self.leaky_relu_2(self.conv_layer_2(x))
        x = self.maxpool_2(x)
        x = self.leaky_relu_3(self.conv_layer_3(x))
        x = self.leaky_relu_4(self.conv_layer_4(x))
        x = self.leaky_relu_5(self.conv_layer_5(x))
        x = self.leaky_relu_6(self.conv_layer_6(x))
        x = self.maxpool_3(x)
        x = self.leaky_relu_7(self.conv_layer_7(x))
        x = self.leaky_relu_8(self.conv_layer_8(x))
        x = self.leaky_relu_9(self.conv_layer_9(x))
        x = self.leaky_relu_10(self.conv_layer_10(x))
        x = self.leaky_relu_11(self.conv_layer_11(x))
        x = self.leaky_relu_12(self.conv_layer_12(x))
        x = self.leaky_relu_13(self.conv_layer_13(x))
        x = self.leaky_relu_14(self.conv_layer_14(x))
        x = self.leaky_relu_15(self.conv_layer_15(x))
        x = self.leaky_relu_16(self.conv_layer_16(x))
        x = self.maxpool_4(x)
        x = self.leaky_relu_17(self.conv_layer_17(x))
        x = self.leaky_relu_18(self.conv_layer_18(x))
        x = self.leaky_relu_19(self.conv_layer_19(x))
        x = self.leaky_relu_20(self.conv_layer_20(x))
        x = self.leaky_relu_21(self.conv_layer_21(x))
        x = self.leaky_relu_22(self.conv_layer_22(x))
        x = self.leaky_relu_23(self.conv_layer_23(x))
        x = self.leaky_relu_24(self.conv_layer_24(x))

        x = x.view(self.batch_size, -1)
        x = self.fc_1(x)
        x = self.fc_2(x)
        x = x.view((self.batch_size, 7 * 7, self.depth_per_cell))

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


def compare_existence(pred_logits, label_logits):
    """
    compare and get which classes are existed both in prediction and label
    :param pred_logits:
    :param label_logits:
    :return:
    """
    indices = []
    for i in range(len(pred_logits)):
        if pred_logits[i] == 1 and pred_logits[i] == label_logits[i]:
            indices.append(i)
    return indices


def calculate_iou(pred_box, labeled_box):
    """
    calculate the iou between different predict and labeled boxes
    :param pred_box:
    :param labeled_box:
    :return:
    """
    # convert from (x, y, w, h) to (xmin, ymin, xmax, ymax)
    p_box = [pred_box[0], pred_box[1],
             pred_box[0] + pred_box[2],
             pred_box[1] + pred_box[3]]
    l_box = [labeled_box[0],
             labeled_box[1],
             labeled_box[0] + labeled_box[2],
             labeled_box[1] + labeled_box[3]]
    # calculate the intersection part
    xmin = max(p_box[0], l_box[0])
    ymin = max(p_box[1], l_box[1])
    xmax = min(p_box[2], l_box[2])
    ymax = min(p_box[3], l_box[3])

    w_inter = xmax - xmin
    h_inter = ymax - ymin

    # area of predicted and labeled boxes
    pred_area = pred_box[2] * pred_box[3]
    label_area = labeled_box[2] * labeled_box[3]

    # compute iou ratio
    inter_area = w_inter * h_inter
    total_area = pred_area + label_area - inter_area
    iou = inter_area / float(total_area)

    return iou


def find_max_overlap(pred_boxes, labeled_boxes):
    """
    compare the predicted boxes and labeled boxes, find the pair of boxes that has most of the overlap.
    return the
    :param pred_boxes:
    :param labeled_boxes:
    :return:
    """
    pairs = {}
    # find the max iou between predicted and labeled boxes
    for i in range(pred_boxes.shape[0]):
        max_iou = 0.
        for j in range(labeled_boxes.shape[0]):
            if all([x == 0.0 for x in labeled_boxes[j].data.tolist()]):
                continue
            iou = calculate_iou(pred_boxes[i], labeled_boxes[j])
            if iou <= 0:
              pairs[i] = -1
            elif iou > max_iou:
                pairs[i] = j
                max_iou = iou
    return pairs


def coordinate_loss(match_pairs, pred_boxes, labeled_boxes, criterion):
    """
    calculate the coordinate loss based on the matching pair information
    :param match_pairs:
    :param pred_boxes:
    :param labeled_boxes:
    :param criterion:
    :return:
    """
    loss = torch.zeros(1)
    for i in range(pred_boxes.shape[0]):
        for j in range(labeled_boxes.shape[0]):
            if match_pairs[i] == j:
                # mse of x and y
                loss += criterion(pred_boxes[i][0], labeled_boxes[j][0])\
                        + criterion(pred_boxes[i][1], labeled_boxes[j][1])
                # mse of sqrt of w and h
                # pred_boxes[i] = torch.clamp(pred_boxes[i], min=0)
                loss += criterion(pred_boxes[i][2] ** 1, labeled_boxes[j][2] ** 1) \
                        + criterion(pred_boxes[i][3] ** 1, labeled_boxes[j][3] ** 1)
    return loss


def confidence_loss(match_pair, pred_conf, lambda_noobj, criterion):
    """
    calculate the confidence loss for both having object and no object scenarios
    :param match_pair:
    :param pred_conf:
    :param lambda_noobj:
    :param criterion:
    :return:
    """
    loss = torch.zeros(1)
    for i in range(pred_conf.shape[0]):
        if match_pair[i] != -1:
            loss += criterion(pred_conf[i], torch.ones(1))
        else:
            loss += lambda_noobj * criterion(pred_conf[i], torch.zeros(1))
    return loss


def logits_loss(pred_logits, labeled_logits, criterion):
    """
    calculate the logits loss between pred_logits and labeleld logits
    :param pred_logits:
    :param labeled_logits:
    :param criterion
    :return:
    """
    return criterion(pred_logits, labeled_logits)


def losses(output, logits, bboxes,
           lambda_coord, lambda_noobj,
           num_boxes, num_classes,
           step, name='train'):
    """
    calculate the losses by comparing bboxes and classes
    :param output:
    :param labels:
    :return:
    """
    criterion = nn.MSELoss()
    total_loss = Variable(torch.zeros(1))
    coord_loss = Variable(torch.zeros(1))
    conf_loss = Variable(torch.zeros(1))
    class_logits_loss = Variable(torch.zeros(1))
    # output shape is according to (batch, 30 x 30 tensors, tensor depth)
    # tensor depth is (num_boxes * 5 + num_classes)
    # here we assign the logits of prediction as [0 ... 1 (class logits, length equals to number of classes)
    # 0.1 0.9 ... (number of boxes) x_1 y_1 w_1 h_1 ... x_n y_n w_n h_n (predict boxes according to the number of boxes)]
    for k in range(output.shape[0]):
        label_logits = logits[k]
        label_boxes = bboxes[k]
        for i in range(output.shape[1]):
            pred_tensor = output[k, i, :]
            pred_logits = pred_tensor[: num_classes]
            pred_confid = pred_tensor[num_classes: num_classes + num_boxes]
            pred_boxes = pred_tensor[num_classes + num_boxes: ].view([num_boxes, -1])
            max_pairs = find_max_overlap(pred_boxes, label_boxes)
            if len(max_pairs.keys()) == 0:
                continue

            # add coordinate loss
            coord_loss += lambda_coord * coordinate_loss(max_pairs, pred_boxes, label_boxes, criterion)
            # total_loss += coord_loss

            # add confidence loss
            conf_loss += confidence_loss(max_pairs, pred_confid, lambda_noobj, criterion)
            # total_loss += conf_loss
            # add class logits loss
            if len(max_pairs.keys()) > 0:
                class_logits_loss += logits_loss(pred_logits, label_logits, criterion)
                # total_loss += class_logits_loss

        total_loss += coord_loss + conf_loss + class_logits_loss
        # use tensorboard to track the performance
        writer.add_scalar('loss_{}/total_loss'.format(name), total_loss[0], step)
        writer.add_scalar('loss_{}/coord_loss'.format(name), coord_loss[0], step)
        writer.add_scalar('loss_{}/conf_loss'.format(name), conf_loss[0], step)
        writer.add_scalar('loss_{}/class_loss'.format(name), class_logits_loss[0], step)

            # print("[DEBUG] coordinates loss: {}, "
            #       "confidence loss: {}, "
            #       "logits loss: {}, "
            #       "match pairs {}".format(coord_loss.data.tolist(),
            #                               conf_loss.data.tolist(),
            #                               class_logits_loss.data.tolist(),
            #                               max_pairs))
    return total_loss

def train_step(data_loader, optimizer,
               yolo, class_prediction,
               pretrain, step,
               lambda_coord, lambda_noobj,
               num_boxes, num_classes):
    """
    run one epoch of training
    :param data_loader:
    :param optimizer:
    :param transforms:
    :param yolo:
    :param class_prediction:
    :param pretrain:
    :param step:
    :return:
    """
    yolo.train()
    if pretrain:
        class_prediction.train()

    for image, logits, bboxes in data_loader:
        if use_cuda:
            image_tensor = Variable(image).cuda()
            logits = Variable(logits).cuda()
            bboxes = Variable(bboxes).cuda()
        else:
            image_tensor = Variable(image)
            logits = Variable(logits)
            bboxes = Variable(bboxes)

        yolo_out = yolo.forward(image_tensor)
        optimizer.zero_grad()
        if pretrain:
            classes_pred = class_prediction.forward(yolo_out)
            criterion = nn.CrossEntropyLoss()
            loss = criterion(classes_pred, logits)
        else:
            # criterion = nn.MSELoss()
            # loss = criterion()
            loss = losses(yolo_out, logits, bboxes,
                          lambda_coord, lambda_noobj,
                          num_boxes, num_classes,
                          step, name='train')
        loss.backward()
        optimizer.step()
        print("Train step {}, loss {}".format(step, loss.item()))
        step += 1
    return step


def test_step(data_loader, yolo,
              step, lambda_coord,
              lambda_noobj, num_boxes,
              num_classes):
    """

    :param data_loader:
    :param yolo:
    :param step:
    :param lambda_coord:
    :param lambda_noobj:
    :param num_boxes:
    :param num_classes:
    :return:
    """
    yolo.eval()
    for i, (image, logits, bboxes) in enumerate(data_loader):
        if use_cuda:
            image_tensor = Variable(image).cuda()
            logits = Variable(logits).cuda()
            bboxes = Variable(bboxes).cuda()
        else:
            image_tensor = Variable(image)
            logits = Variable(logits)
            bboxes = Variable(bboxes)

        yolo_out = yolo.forward(image_tensor)
        loss = losses(yolo_out, logits, bboxes,
                      lambda_coord, lambda_noobj,
                      num_boxes, num_classes,
                      step, name='test')
        print("[Testing Set] Current step {}, Total loss is {}".format(step, loss.item()))


def train(input_folder,
          annotation_map,
          output_folder,
          batch_size,
          resize_shape,
          num_classes,
          num_boxes,
          lr,
          beta1,
          epochs,
          lambda_coord=5,
          lambda_noobj=0.5,
          checkpoint=None,
          ckpt_save_epoch=100,
          pretrain=False,
          pretrain_epochs=1000):
    """
    setup environment and variables to run training
    :param input_folder:
    :param annotation_map:
    :param output_folder:
    :param batch_size:
    :param num_classes:
    :param num_boxes:
    :param lr:
    :param beta1:
    :param epochs:
    :param lambda_coord:
    :param lambda_noobj:
    :param checkpoint:
    :param ckpt_save_epoch:
    :param pretrain:
    :param pretrain_epochs:
    :return:
    """

    # image preprocessing
    trans = transforms.Compose([Resize((resize_shape, resize_shape)), ToTensor()])

    # get data generator ready
    pad_collate = PadCollate()
    train_folder = os.path.join(input_folder, 'train')
    train_dataset = VOCDataset(image_folder=train_folder,
                               annotation_folder=train_folder,
                               annotation_map_file=annotation_map,
                               transforms=trans)
    train_data_loader = DataLoader(train_dataset,
                                   batch_size=batch_size,
                                   shuffle=True,
                                   drop_last=True,
                                   collate_fn=pad_collate
                                   )

    test_folder = os.path.join(input_folder, 'test')
    test_dataset = VOCDataset(image_folder=test_folder,
                              annotation_folder=test_folder,
                              annotation_map_file=annotation_map,
                              transforms=trans)
    test_data_loader = DataLoader(test_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  drop_last=True,
                                  collate_fn=pad_collate)
    # get nets ready
    if use_cuda:
        yolo = Yolo(num_classes, num_boxes, batch_size).cuda()
    else:
        yolo = Yolo(num_classes, num_boxes, batch_size)

    # if we want to pretrain the model, then we need to use a classification net
    # instead of a box regression net follows the yolo net
    class_prediction = None
    if pretrain:
        if use_cuda:
            class_prediction = ClassificationNet(num_classes=num_classes).cuda()
        else:
            class_prediction = ClassificationNet(num_classes=num_classes)

    # get optimizer ready
    if pretrain:
        optimizer = optim.Adam([yolo.parameters(), class_prediction.parameters()],
                               lr=lr,
                               betas=(beta1, 0.999))
    else:
        optimizer = optim.Adam(yolo.parameters(), lr=lr, betas=(beta1, 0.999))

    step = 1
    for epoch in range(epochs):
        step = train_step(train_data_loader, optimizer,
                          yolo, class_prediction,
                          pretrain, step,
                          lambda_coord, lambda_noobj,
                          num_boxes, num_classes)

        test_step(test_data_loader,
                  yolo, step,
                  lambda_coord, lambda_noobj,
                  num_boxes, num_classes)

        if epochs % ckpt_save_epoch == 0:
            torch.save(yolo.state_dict(),
                       os.path.join(output_folder,
                                    'ckpt_yolo_{}.tar'.format(step)))

def main():
    args = parse_arguments()
    train(args.input_folder,
          args.annotation_map,
          args.log_dir,
          args.batch_size,
          args.resize_shape,
          args.num_classes,
          args.num_boxes,
          args.learning_rate,
          args.beta1,
          args.epochs,
          args.lambda_coord,
          args.lambda_noobj)


if __name__ == '__main__':
    main()