import os

import cv2
import numpy as np
import json

from vedaseg.datasets.base import BaseDataset
from .registry import DATASETS


@DATASETS.register_module
class CCTVRawFrameDataset(BaseDataset):
    CLASSES = ('Embrace', 'ShakeHands', 'Meeting', 'ReleaseConference',
               'Conference', 'Photograph', 'TakeOffPlane', 'MilitaryParade',
               'MilitaryExercise', 'RocketLaunching', 'ConferenceSpeech',
               'Interview')

    def __init__(self,
                 root,
                 ann_file,
                 img_prefix,
                 nclasses=12,
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
        ignore_segments = []

        for anno in gt['annotations']:
            segment = [int(i * self.fps) for i in anno['segment']]
            if segment[1] - segment[0] <= 0:
                continue

            if anno['label'] == 'Ignore':
                ignore_segments.append(segment)
                continue

            segments.append(segment)
            labels.append(self.CLASSES.index(anno['label']))

        if len(segments) == 0:
            return None

        # mask shape C*T
        data = dict(image=fnames, duration=len(fnames),
                    labels=np.array(labels), segments=np.array(segments),
                    ignore_segments=np.array(ignore_segments))
        image, mask = self.process(data)

        return image.float(), mask.long(), self.video_names[item]

    def __len__(self):
        return len(self.video_names)

    def _rand_another(self, idx):
        """Get another random index from the same group as the given index."""
        pool = np.where(len(self.video_names))[0]
        return np.random.choice(pool)

    def get_all_gts(self):
        """Fetch groundtruth instances of the entire dataset."""
        gts = {}
        for video_info, anno in self.data.items():
            video = video_info
            # frame_dir = os.path.join(self.img_prefix, video_info)
            # total_frames = len(os.listdir(frame_dir))
            for gt in anno["annotations"]:
                if gt['label'] == 'Ignore':
                    continue
                class_idx = self.CLASSES.index(gt['label'])
                gt_info = [
                    gt['segment'][0] / anno['duration_second'],
                    gt['segment'][1] / anno['duration_second']
                ]
                gts.setdefault(class_idx, {}).setdefault(video,
                                                         []).append(gt_info)

        return gts
