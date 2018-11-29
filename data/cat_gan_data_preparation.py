import os
import pandas
import argparse
from libs import utilities

def parse_arguments():
    argparser = argparse.ArgumentParser()
    argparser.add_argument('-i', '--input_folder',
                           help='The input folder for the data')
    argparser.add_argument('-o', '--output_file',
                           help='The output folder for the data record')
    return argparser.parse_args()


def main():
    args = parse_arguments()
    utilities.Utilities.create_folder(os.path.dirname(args.output_file))

    res_dict = {'filename': []}

    for root, dirs, files in os.walk(args.input_folder):
        for f in files:
            if '.jpg' in f:
                res_dict['filename'].append(f)
    dataframe = pandas.DataFrame(res_dict)
    dataframe['id'] = list(range(1, len(res_dict['filename']) + 1))
    dataframe.to_csv(args.output_file)

if __name__ == '__main__':
    main()