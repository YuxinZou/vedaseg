import argparse
import os
import sys
import json

import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../'))

import cv2
import numpy as np
import matplotlib.pyplot as plt

from vedaseg.runners import InferenceRunner
from vedaseg.utils import Config
from vedaseg.datasets import build_dataset
from vedaseg.transforms import build_transform

CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
           'CliffDiving', 'CricketBowling', 'CricketShot',
           'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow',
           'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
           'SoccerPenalty', 'ThrowDiscus', 'VolleyballSpiking')


def generate_sequence(idx, data, fps=10):
    data = data.ravel()
    print(data)
    result = []
    start = 0
    index = 0
    for i in data:
        index += 1
        if i == 1:
            if start == 0:
                start = i
            else:
                result.append(dict(segment=[float(start/fps), float(i/fps)], label=CLASSES[idx]))
                start = 0
    return result



def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference a segmentatation model')
    parser.add_argument('config', type=str,
                        help='config file path')
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint file path')
    parser.add_argument('--out', default='./result',
                        help='folder to store results')

    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    cfg_path = args.config
    cfg = Config.fromfile(cfg_path)

    inference_cfg = cfg['inference']
    common_cfg = cfg.get('common')

    runner = InferenceRunner(inference_cfg, common_cfg)
    runner.load_checkpoint(args.checkpoint)

    transform = build_transform(cfg.test.data['transforms'])
    dataset = build_dataset(cfg.test.data['dataset'], dict(transform=transform))
    print(len(dataset))
    image, mask = dataset[0]
    print(dataset[0])
    print(torch.sum(mask))
    output = runner(image, mask)
    output = output
    res = []
    for i in range(output.shape[0]):
        re = generate_sequence(i, output[i])
        res.extend(re)

    print(res)

if __name__ == '__main__':
    main()
