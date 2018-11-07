import cv2
import numpy as np
from torch.utils.data import Dataset
from libs.utilities import get_file_list
from libs.annotation_file_readers import XMLReader
from libs.image_process import resize_image

class VOCDataset(Dataset):
    def __init__(self, image_folder,
                 annotation_folder,
                 annotation_map_file,
                 image_pattern='.jpg',
                 annotation_pattern='.xml',
                 normalization=False,
                 resize_x=448,
                 resize_y=448,
                 num_classes=20):
        self.images = get_file_list(image_folder, image_pattern)
        self.annotations = get_file_list(annotation_folder, annotation_pattern)
        self.annotation_mapping = np.load(annotation_map_file)
        self.xml_reader = XMLReader()
        self.normalize = normalization
        self.target_size = (resize_x, resize_y)
        self.num_classes = num_classes

        assert len(self.images) == len(self.annotations), \
            "[DEBUG] The numbers of images and annotations are not the same"

        self.images = sorted(self.images)
        self.annotations = sorted(self.annotations)
        self.list_ids = list(range(len(self.images)))

    def __len__(self):
        return len(self.list_ids)

    def organize_labels(self, label_dict):
        """
        reformat the labels into t
        :param label_dict:
        :return:
        """
        image_width = label_dict['image_width']
        image_height = label_dict['image_height']

        bboxes = label_dict['bbox']
        bboxes = [[x[0] / image_width,
                  x[1] / image_height,
                  x[2] / image_width,
                  x[3] / image_height] for x in bboxes]
        # convert the box from 4 anchors to x,y,w,h
        bboxes = [[x[0], x[1], x[2] - x[0], x[3] - x[1]] for x in bboxes]

        classes = [self.annotation_mapping[x] for x in label_dict['object_name']]
        classes_indices = [[0] * self.num_classes] * len(classes)

        # assign the corresponding class logit to 1
        for i in range(len(classes)):
            classes_indices[i][classes[i]] = 1

        # combine logits and bounding boxes together
        labels = []
        for i in range(len(bboxes)):
            label = classes_indices[i] + bboxes[i]
            labels.append(label)
        return labels

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        image = resize_image(image, self.target_size)
        labels_dict = self.xml_reader.read_xml_file(self.annotations[index])
        labels = self.organize_labels(labels_dict)
        return image, labels