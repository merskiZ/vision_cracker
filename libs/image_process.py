import cv2


def resize_image(image, target_size):
    """
    resize image to a given target size
    :param image:
    :param target_size:
    :return:
    """
    image = cv2.resize(image, target_size)
    return image


def normalize_image(image):
    """
    convert an image from 0 to 255 to 0 to 1
    :param image:
    :return:
    """
    return image / 255.
