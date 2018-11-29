"""
parse all the label names and create a mapping between class names and class ids
"""

import os
import argparse
import pickle
import tqdm
import pandas as pd
from libs.annotation_file_readers import XMLReader
from libs.utilities import Utilities, get_file_list


def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input_folder',
                           type=str,
                           required=True,
                           help='The input folder that contains xml files of the label annotations')
    argparser.add_argument('-o', '--output_file',
                           type=str,
                           default='/tmp/xml_annotations.csv',
                           help='The output csv file that contains the annotation information')
    argparser.add_argument('-m', '--mapping_file',
                           type=str,
                           default='/tmp/annotation_map.pkl',
                           help='The mapping file between object names to class ids')
    return argparser.parse_args()

def get_mapping(data_frame):
    """
    get the object name to class id mapping
    :param data_frame:
    :return:
    """
    class_names = list(set([x for y in data_frame['object_name'] for x in y]))
    ids = list(range(len(class_names)))
    mapping = dict(zip(class_names, ids))

    return mapping

def main():
    args = parse_arguments()

    output_folder = os.path.dirname(args.output_file)
    Utilities.create_folder(output_folder)

    # parse the xmls
    xml_reader = XMLReader()
    label_files = get_file_list(args.input_folder, '.xml')
    res = {'filename': [],
           'image_width': [],
           'image_height': [],
           'image_channel': [],
           'object_name': [],
           'bbox': []}
    for line in tqdm.tqdm(label_files):
        parsed = xml_reader.read_xml_file(line)

        for key in res:
            res[key].append(parsed[key])
    data_frame = pd.DataFrame(res)
    data_frame.to_csv(args.output_file)

    # get the mapping
    mapping = get_mapping(data_frame)
    mapping_folder = os.path.dirname(args.mapping_file)
    Utilities.create_folder(mapping_folder)
    pickle.dump(mapping, open(args.mapping_file, 'wb'))

if __name__ == '__main__':
    main()

