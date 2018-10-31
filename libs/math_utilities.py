import random
import torch

def random_generator(epoch=1000, min=0.0, max=0.3):
    base_num = 1000000
    for e in range(epoch):
        yield random.randrange(min * base_num, max * base_num) / float(base_num)