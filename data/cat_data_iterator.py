import pandas
import cv2
import numpy as np
import random
from src.libs.image_process import resize_image, normalize_image


class CatDataIterator(object):
    def __init__(self, data_frame_file,
                 image_size=(256, 256)):
        self.data_frame = pandas.DataFrame(data_frame_file)
        self.image_size = image_size
        self.seed = random.randint(1, 99)

    def get_data(self, batch_size=8):
        data_frame = self.data_frame.sample(frac=1.0,
                                            random_state=self.seed)
        filenames = list(data_frame['filename'])
        for i in range(0, len(filenames), batch_size):
            start = i
            end = i + batch_size
            images = np.zeros((batch_size,
                               self.image_size[0],
                               self.image_size[1],
                               3))
            for j in range(start, end):
                image = cv2.imread(filenames[j])
                image = normalize_image(image)
                image = resize_image(image, target_size=self.image_size)
                images[j, ...] = image
            yield images
