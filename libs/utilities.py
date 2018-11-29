import os


def get_file_list(input_folder, pattern):
    """
    return a list of file paths that conforms the given pattern in the input folder
    :param input_folder:
    :param pattern:
    :return:
    """
    paths = []
    for root, dirs, files in os.walk(input_folder):
        for f in files:
            if pattern in f:
                paths.append(os.path.join(root, f))
    return paths


class Utilities(object):
    @staticmethod
    def create_folder(folder):
        if not os.path.exists(folder):
            os.makedirs(folder)

