import cv2
from torch.utils.data import Dataset
from libs.utilities import get_file_list

class VOCDataset(Dataset):
    def __init__(self, image_folder, annotation_folder,
                 image_pattern='.jpg', annotation_pattern='.xml'):
        self.images = get_file_list(image_folder, image_pattern)
        self.annotations = get_file_list(annotation_folder, annotation_pattern)

        assert len(self.images) == len(self.annotations), \
            "[DEBUG] The numbers of images and annotations are not the same"

        self.images = sorted(self.images)
        self.annotations = sorted(self.annotations)
        self.list_ids = list(range(len(self.images)))

    def __len__(self):
        return len(self.list_ids)

    def __getitem__(self, index):
        image = cv2.imread(self.images[index])
        