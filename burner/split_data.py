"""
Split a given dataset raw images and annotations into different splits and conforms the input split ratios
"""

import os
import argparse
from random import sample
import subprocess
import numpy as np

import tqdm

from libs.utilities import Utilities, get_file_list

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('--input_folder',
                           type=str,
                           required=True,
                           help='The input folder contains both annotation and image directories')
    argparser.add_argument('--output_folder',
                           type=str,
                           default='/tmp/split_experiment',
                           help='The output folder that contains split images and annotations folder')
    argparser.add_argument('--train_ratio',
                           default=0.8,
                           type=float,
                           help='the training fold ratio')
    return argparser.parse_args()


def create_folders(folders):
    """
    create a list of folders
    :param folders:
    :return:
    """
    for folder in folders:
        Utilities.create_folder(folder)


def copy_files(files_list, image_folder, labels_folder):
    """
    copy a list of file to a given existed folder
    :param files_list:
    :param image_folder
    :param labels_folder
    :return:
    """
    for line in tqdm.tqdm(files_list):
        cp_img_cmd = 'cp {} {}'.format(line[0], os.path.join(image_folder, os.path.basename(line[0])))
        cp_label_cmd = 'cp {} {}'.format(line[1], os.path.join(labels_folder, os.path.basename(line[1])))
        subprocess.call(cp_img_cmd.split())
        subprocess.call(cp_label_cmd.split())


def main():
    args = parse_arguments()
    input_dir = args.input_folder
    output_dir = args.output_folder
    train_ratio = args.train_ratio

    # create output folders
    train_output = os.path.join(output_dir, 'train')
    train_image_output = os.path.join(train_output, 'images')
    train_annotation_output = os.path.join(train_output, 'annotations')
    test_output = os.path.join(output_dir, 'test')
    test_image_output = os.path.join(test_output, 'images')
    test_annotation_output = os.path.join(test_output, 'annotations')
    create_folders([train_image_output, train_annotation_output,
                    test_image_output, test_annotation_output])

    # sample train and test files and copy them to different folders
    image_files = get_file_list(input_dir, '.jpg')
    label_files = [x.replace('JPEGImages', 'Annotations').replace('.jpg', '.xml') for x in image_files]

    files_zip = np.array(list(zip(image_files, label_files)))

    num_files = len(image_files)
    indices = list(range(num_files))

    train_indices = sample(indices, int(num_files * train_ratio))
    test_indices = list(set(train_indices) ^ set(indices))

    train_files = list(files_zip[train_indices])
    test_files = list(files_zip[test_indices])
    print("Train files length {}, test files length {}".format(len(train_files), len(test_files)))

    copy_files(train_files, train_image_output, train_annotation_output)
    copy_files(test_files, test_image_output, test_annotation_output)

if __name__ == '__main__':
    main()