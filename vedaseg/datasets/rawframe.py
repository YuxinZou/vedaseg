import os

import cv2
import numpy as np
import json

from vedaseg.datasets.base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class RawFrameDataset(BaseDataset):
    CLASSES = ('BaseballPitch', 'BasketballDunk', 'Billiards', 'CleanAndJerk',
               'CliffDiving', 'CricketBowling', 'CricketShot',
               'Diving', 'FrisbeeCatch', 'GolfSwing', 'HammerThrow',
               'HighJump', 'JavelinThrow', 'LongJump', 'PoleVault', 'Shotput',
               'SoccerPenalty', 'TennisSwing', 'ThrowDiscus', 'VolleyballSpiking')

    def __init__(self,
                 root,
                 ann_file,
                 img_prefix,
                 nclasses=20,
                 fps=10,
                 transform=None,
                 multi_label=True,
                 mask_value=255,
                 ):
        super().__init__()
        self.root = root
        self.data = json.load(open(os.path.join(self.root, ann_file), 'r'))
        self.multi_label = multi_label
        self.transform = transform
        self.img_prefix = img_prefix
        self.mask_value = mask_value
        self.nclasses = nclasses
        self.fps = fps
        self.video_names = list(self.data.keys())
        if self.root is not None:
            self.img_prefix = os.path.join(self.root, self.img_prefix)

        self.cap = cv2.VideoCapture()


    def __getitem__(self, idx):
        """Get training/test data after pipeline.

        Args:
            idx (int): Index of data.

        Returns:
            dict: Training/test data (with annotation if `test_mode` is set
                True).
        """
        while True:
            data = self.prepare(idx)
            if data is None:
                idx = self._rand_another(idx)
                continue

            return data

    def prepare(self, item):
        frame_dir = os.path.join(self.img_prefix, self.video_names[item])
        gt = self.data[self.video_names[item]]
        fnames = sorted(os.listdir(frame_dir))
        fnames = [os.path.join(frame_dir, img) for img in fnames]

        labels = []
        segments = []

        if len(gt['annotations']) == 0:
            return None

        for anno in gt['annotations']:
            segments.append([int(i * self.fps) for i in anno['segment']])
            labels.append(self.CLASSES.index(anno['label']))

        # mask shape C*T
        data = dict(image=fnames, duration=len(fnames),
                    labels=np.array(labels), segments=np.array(segments))
        image, mask = self.process(data)

        return image.float(), mask.long()

    def __len__(self):
        return len(self.video_names)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(len(self.video_names))[0]
        return np.random.choice(pool)