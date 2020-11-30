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

        # res = []
        # gt = self.data[self.video_names[item]]
        # duration_second = gt['duration_second']
        # frame_count = int(self.fps * gt['duration_second'])
        # print(frame_count)
        # fname = os.path.join(self.img_prefix, self.video_names[item] + '.mp4')
        # self.cap.set(cv2.CAP_PROP_FPS, 10)
        # self.cap.open(fname)
        # while self.cap.isOpened():
        #     ret, frame = self.cap.read()
        #     if ret is False:
        #         break
        #     out_img = cv2.resize(frame, self.size)
        #     res.append(out_img)
        #
        # res = np.array(res)

#
#
# import logging
# import time
# from queue import Queue
# from threading import Thread
#
# import cv2
# import torch
# from torch.nn import functional as F
# from torchvision.transforms import Normalize
#
#
# class Reader:
#     def __init__(self, size, frame_interval, buffer_size=16,
#                  mean=[123.675, 116.28, 103.53],
#                  std=[58.395, 57.12, 57.375]):
#         self.size = size
#         self.frame_interval = frame_interval
#         self.buffer_size = buffer_size
#         self.logger = logging.getLogger()
#         self.norm = Normalize(mean=mean, std=std)
#
#         self.cap = cv2.VideoCapture()
#         self.q = Queue(buffer_size)
#         # self.t = self.gen_thread()
#
#     @property
#     def video_time(self):
#         fps = self.cap.get(cv2.CAP_PROP_FPS)
#         frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
#         time_length = frame_count/(fps+1e-15)
#
#         return time_length
#
#     @property
#     def video_resolution(self):
#         w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
#         h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
#
#         return w, h
#
#     @property
#     def frame_count(self):
#         return int(1000*self.video_time/self.frame_interval)
#
#     @property
#     def target_size(self):
#         w, h = self.video_resolution
#         size = self.size
#
#         if h <= w:
#             nh = min(size)
#             nw = max(size)
#         else:
#             nh = max(size)
#             nw = min(size)
#
#         return nw, nh
#
#     def _reader(self):
#         self.logger.debug(f'duration: {self.video_time:.2f}')
#         self.logger.debug(f'resolution: {self.video_resolution}')
#         for msec in range(0, int(self.video_time*1000),
#                           int(self.frame_interval)):
#             self.cap.set(propId=cv2.CAP_PROP_POS_MSEC,
#                          value=msec)
#             ret, frame = self.cap.read()
#             if ret:
#                 torch_frame, meta = self.transform(frame)
#                 self.q.put((torch_frame, frame, meta))
#             else:
#                 break
#         while not self.q.qsize() == 0:
#             time.sleep(0.1)
#
#     def set_video(self, video):
#         """
#         Set video then return original video resolution and target resolution.
#         """
#         self.cap.open(video)
#         if not self.cap.isOpened():
#             self.logger.warning(f'can not open video {video}')
#             return None, None
#         else:
#             self.logger.info(f'open video {video}')
#             vsize = self.video_resolution
#             tsize = self.target_size
#             self.t = self.gen_thread()
#             return vsize, tsize
#
#     def clear(self):
#         self.q.queue.clear()
#
#     def gen_thread(self):
#         t = Thread(target=self._reader, daemon=True)
#         t.start()
#         return t
#
#     def is_alive(self):
#         return self.t.is_alive()
#
#     def is_empty(self):
#         return self.q.qsize() == 0
#
#     def read(self):
#         return self.q.get()
#
#     def transform(self, img):
#         """
#         Resize with right-bottom padding.
#         Returns transformed tensor as well as meta info.
#         """
#         ih, iw, ic = img.shape
#         tw, th = self.target_size
#
#         target_ratio = th/tw
#         img_ratio = ih/iw
#         if img_ratio <= target_ratio:
#             resized_h = round(tw/iw*ih)
#             resized_w = tw
#             pad_h = th - resized_h
#             pad_w = 0
#         else:
#             resized_h = th
#             resized_w = round(th/ih*iw)
#             pad_h = 0
#             pad_w = tw - resized_w
#         img = cv2.resize(img, (resized_w, resized_h))
#
#         img = torch.from_numpy(img).float()
#         img = img.permute(2, 0, 1)  # c, h, w
#         img = self.norm(img)
#         img = F.pad(img, (0, pad_w, 0, pad_h))
#         img = img.unsqueeze(0)  # n
#
#         meta = dict(pad=(pad_w, pad_h))
#
#         return img, meta
#
#
# if __name__ == '__main__':
#     import os
#
#     logger = logging.basicConfig(
#         format='%(asctime)s - %(levelname)s - %(message)s',
#         level=logging.DEBUG
#     )
#     reader = Reader(size=(216, 384), frame_interval=40, buffer_size=32)
#
#     date_dir = '/home/kuro/dev/db/dev/projects/TASED-Net/data/orig/zhiru2/videos'  # noqa: E501
#     for video_name in os.listdir(date_dir):
#         video_file = os.path.join(date_dir, video_name)
#
#         reader.set_video(video_file)
#         while reader.is_alive():
#             if not reader.is_empty():
#                 s = time.time()
#                 torch_frame, frame = reader.read()
#                 e = time.time()
#                 time.sleep(0.025)
#                 cv2.imshow('a', frame)
#                 cv2.waitKey(1)
