import os
import glob
import subprocess
import tqdm

if __name__ == '__main__':
    input_folder = '/Users/yameng/workspace/datasets/cats_and_dogs_filtered/'

    images = glob.glob(os.path.join(input_folder, '*', '*', '*.jpg'))

    for line in tqdm.tqdm(images):
        new_name = line.replace('cat.', 'cat').replace('dog.', 'dog')
        cmd = 'mv {} {}'.format(line, new_name)
        subprocess.call(cmd.split())

