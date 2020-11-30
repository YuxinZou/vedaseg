import argparse
import os

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vedaseg.runners import InferenceRunner
from vedaseg.utils import Config
from vedaseg.datasets import build_dataset
from vedaseg.transforms import build_transform
from vedaseg.models import build_model
import random
import numpy as np
import torch

np.random.seed(0)
random.seed(0)


def main():
    cfg_path = '/home/admin123/PycharmProjects/github/meizhi/vedaseg/configs/action_segmentation.py'
    cfg = Config.fromfile(cfg_path)
    cfg = cfg.inference.model
    print(cfg)
    model = build_model(cfg)
    dummy_input = torch.rand(1, 3, 256, 96, 96)

    print(model)
    output = model(dummy_input)
    print(output.shape)


if __name__ == '__main__':
    main()
