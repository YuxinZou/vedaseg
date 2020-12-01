import os

import cv2
import numpy as np
import json

from vedaseg.datasets.base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class ActionDataset(BaseDataset):
    CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
               'CliffDiving', 'CricketBowling', 'CricketShot',
               'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow',
               'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
               'SoccerPenalty', 'ThrowDiscus', 'VolleyballSpiking')

    def __init__(self,
                 root,
                 ann_file,
                 img_prefix,
                 nclasses=20,
                 size=(96, 96),
                 fps=10,
                 transform=None,
                 multi_label=True,
                 ):
        super().__init__()
        self.root = root
        self.data = json.load(open(os.path.join(self.root, ann_file), 'r'))
        self.multi_label = multi_label
        self.transform = transform
        self.img_prefix = img_prefix
        self.nclasses = nclasses
        self.fps = fps
        self.size = size
        self.video_names = list(self.data.keys())
        if self.root is not None:
            self.img_prefix = os.path.join(self.root, self.img_prefix)

        self.cap = cv2.VideoCapture()

    def __getitem__(self, item):
        fname = os.path.join(self.img_prefix, self.video_names[item] + '.mp4')

        gt = self.data[self.video_names[item]]
        duration = int(gt['duration_second'] * self.fps)
        mask = np.zeros((self.nclasses, duration))
        for anno in gt['annotations']:
            segment = [int(i * self.fps) for i in anno['segment']]
            label = anno['label']
            index = self.CLASSES.index(label)
            mask[index, segment[0]:segment[1]] = 1

        # mask shape C*T
        data = dict(image=fname, duration=duration, mask=mask)
        image, mask = self.process(data)
        return image.float(), mask.long()

    def __len__(self):
        return len(self.video_names)