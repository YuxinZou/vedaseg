import argparse
import os


import cv2
import numpy as np
import matplotlib.pyplot as plt

from vedaseg.runners import InferenceRunner
from vedaseg.utils import Config
from vedaseg.datasets import build_dataset
from vedaseg.transforms import build_transform
import random
import numpy as np
np.random.seed(0)
random.seed(0)
def main():
    cfg_path = '/home/admin123/PycharmProjects/github/meizhi/vedaseg/configs/action_segmentation.py'
    cfg = Config.fromfile(cfg_path)
    cfg = cfg.test.data
    print(cfg)
    transform = build_transform(cfg['transforms'])
    print(transform)
    dataset = build_dataset(cfg['dataset'], dict(transform=transform))
    image, mask = dataset[0]
    print(image.shape)
    print(mask.shape)

if __name__ == '__main__':
    main()
