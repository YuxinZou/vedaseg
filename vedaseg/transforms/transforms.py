import random

import cv2
import decord
import torch
import numpy as np
import albumentations as albu
from albumentations import DualTransform
import albumentations.augmentations.functional as F

from .registry import TRANSFORMS


@TRANSFORMS.register_module
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, data):
        for t in self.transforms:
            data = t(data)
        return data


@TRANSFORMS.register_module
class VideoCropRawFrame:
    def __init__(self,
                 window_size=256,
                 fps=10,
                 size=(96, 96),
                 mode='train',
                 num_clip=6,
                 value=(123.675, 116.280, 103.530),
                 mask_value=255):
        self.window_size = window_size
        self.fps = fps
        self.size = size
        self.mode = mode
        self.num_clip = num_clip
        self.value = np.reshape(np.array(value), [1, 1, 3])
        self.mask_value = np.reshape(np.array(mask_value), [1])

    def gen_image(self, fnames, duration, start_idx, window_size):
        end_idx = min(duration, start_idx + window_size)
        images = []

        for i in range(start_idx, end_idx):
            img = cv2.imread(fnames[i])
            img = cv2.resize(img, self.size)
            images.append(img)
        images = np.array(images)

        if images.shape[0] < window_size:
            shape = (window_size - images.shape[0],) + self.size + (3,)
            pad_image = np.zeros(shape) + self.value
            images = np.concatenate((images, pad_image), axis=0)
        return images.astype(np.float)

    def gen_mask(self, mask, duration, start_idx, window_size):
        c, _ = mask.shape
        if start_idx + window_size > duration:
            shape = (c,) + (start_idx + window_size - duration,)
            pad_mask = np.zeros(shape) + self.mask_value
            mask = np.concatenate((mask, pad_mask), axis=1)
        mask = mask[:, start_idx:start_idx + window_size]
        return mask

    def __call__(self, data):
        image = data['image']
        mask = data['mask']
        duration = len(image)
        if self.mode == 'train':
            # sample_position = int(max(0, duration - self.window_size))
            # start_idx = 0 if sample_position == 0 else random.randint(
            #     0, sample_position)
            start_idx = 0
            image = self.gen_image(image, duration, start_idx, self.window_size)
            mask = self.gen_mask(mask, duration, start_idx, self.window_size)
        else:
            images, masks = [], []
            clip_len = self.window_size * self.num_clip
            num_clips = int(np.ceil(duration / clip_len))
            index = [i * clip_len for i in range(num_clips)]
            for inx in index:
                images.append(self.gen_image(image, duration, inx, clip_len))
                masks.append(self.gen_mask(mask, duration, inx, clip_len))
            image = np.array(images)
            mask = np.array(masks)
        return dict(image=image, mask=mask)


@TRANSFORMS.register_module
class VideoCrop:
    def __init__(self,
                 window_size=256,
                 fps=10,
                 size=(96, 96),
                 mode='train',
                 max_clip=8,
                 value=(123.675, 116.280, 103.530),
                 mask_value=255):
        self.window_size = window_size
        self.fps = fps
        self.size = size
        self.mode = mode
        self.max_clip = max_clip
        self.value = np.reshape(np.array(value), [1, 1, 3])
        self.mask_value = np.reshape(np.array(mask_value), [1])

        self.cap = cv2.VideoCapture()
        self.interval = int(self.window_size / self.fps)
        self.frame_interval = 1000 / self.fps

    def gen_image(self, fname, start_idx):
        start_idx = int(start_idx / self.fps)
        images = []
        self.cap.open(fname)
        for msec in range(0, int(self.interval * 1000),
                          int(self.frame_interval)):
            msec += start_idx * 1000
            self.cap.set(propId=cv2.CAP_PROP_POS_MSEC, value=msec)
            ret, frame = self.cap.read()
            if ret:
                frame = cv2.resize(frame, self.size)
                images.append(frame)
            else:
                break

        images = np.array(images)
        if images.shape[0] < self.window_size:
            shape = (self.window_size - images.shape[0],) + self.size + (3,)
            pad_image = np.zeros(shape) + self.value
            images = np.concatenate((images, pad_image), axis=0)
        return images.astype(np.float)

    def gen_mask(self, mask, start_idx):
        c, t = mask.shape
        if start_idx + self.window_size > t:
            shape = (c,) + (start_idx + self.window_size - t,)
            pad_mask = np.zeros(shape) + self.mask_value
            mask = np.concatenate((mask, pad_mask), axis=1)
        mask = mask[:, start_idx:start_idx + self.window_size]
        return mask

    def __call__(self, data):
        image = data['image']
        mask = data['mask']
        duration = data['duration']
        if self.mode == 'train':
            # sample_position = int(max(0, duration - self.window_size))
            # start_idx = 0 if sample_position == 0 else random.randint(
            #     0, sample_position)
            start_idx = 0
            image = self.gen_image(image, start_idx)
            mask = self.gen_mask(mask, start_idx)
        else:
            images, masks = [], []
            num_clips = int(np.ceil(duration / self.window_size))
            index = [i * self.window_size for i in range(num_clips)]
            for inx in index:
                images.append(self.gen_image(image, inx))
                masks.append(self.gen_mask(mask, inx))
            image = np.array(images)
            mask = np.array(masks)
        return dict(image=image, mask=mask)


@TRANSFORMS.register_module
class Normalize:
    def __init__(self, mean=(123.675, 116.280, 103.530),
                 std=(58.395, 57.120, 57.375)):
        self.mean = mean
        self.std = std
        self.channel = len(mean)

    def __call__(self, data):
        image = data['image']
        mask = data['mask']
        mean = np.reshape(np.array(self.mean, dtype=image.dtype),
                          [self.channel])
        std = np.reshape(np.array(self.std, dtype=image.dtype), [self.channel])
        denominator = np.reciprocal(std, dtype=image.dtype)

        new_image = (image - mean) * denominator
        new_mask = mask
        return dict(image=new_image, mask=new_mask)


@TRANSFORMS.register_module
class ToTensor:
    def __call__(self, data):
        image = data['image']
        mask = data['mask']
        if len(image.shape) == 4:
            image = torch.from_numpy(image).permute(3, 0, 1, 2)
        else:
            image = torch.from_numpy(image).permute(0, 4, 1, 2, 3)
        mask = torch.from_numpy(mask)

        return dict(image=image, mask=mask)


@TRANSFORMS.register_module
class FactorScale(DualTransform):
    def __init__(self, scale=1.0, interpolation=cv2.INTER_LINEAR,
                 always_apply=False,
                 p=1.0):
        super(FactorScale, self).__init__(always_apply, p)
        self.scale = scale
        self.interpolation = interpolation

    def apply(self, image, scale=1.0, **params):
        return F.scale(image, scale, interpolation=self.interpolation)

    def apply_to_mask(self, image, scale=1.0, **params):
        return F.scale(image, scale, interpolation=cv2.INTER_NEAREST)

    def get_params(self):
        return {'scale': self.scale}

    def get_transform_init_args_names(self):
        return ('scale',)


@TRANSFORMS.register_module
class LongestMaxSize(FactorScale):
    def __init__(self, h_max, w_max, interpolation=cv2.INTER_LINEAR,
                 always_apply=False, p=1.0):
        self.h_max = h_max
        self.w_max = w_max
        super(LongestMaxSize, self).__init__(interpolation=interpolation,
                                             always_apply=always_apply,
                                             p=p)

    def update_params(self, params, **kwargs):
        params = super(LongestMaxSize, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        scale_h = self.h_max / rows
        scale_w = self.w_max / cols
        scale = min(scale_h, scale_w)

        params.update({'scale': scale})
        return params

    def get_transform_init_args_names(self):
        return ('h_max', 'w_max',)


@TRANSFORMS.register_module
class RandomScale(FactorScale):
    def __init__(self, scale_limit=(0.5, 2), interpolation=cv2.INTER_LINEAR,
                 scale_step=None, always_apply=False, p=1.0):
        super(RandomScale, self).__init__(interpolation=interpolation,
                                          always_apply=always_apply,
                                          p=p)
        self.scale_limit = albu.to_tuple(scale_limit)
        self.scale_step = scale_step

    def get_params(self):
        if self.scale_step:
            num_steps = int((self.scale_limit[1] - self.scale_limit[
                0]) / self.scale_step + 1)
            scale_factors = np.linspace(self.scale_limit[0],
                                        self.scale_limit[1], num_steps)
            scale_factor = np.random.choice(scale_factors).item()
        else:
            scale_factor = random.uniform(self.scale_limit[0],
                                          self.scale_limit[1])

        return {'scale': scale_factor}

    def get_transform_init_args_names(self):
        return ('scale_limit', 'scale_step',)


@TRANSFORMS.register_module
class PadIfNeeded(albu.PadIfNeeded):
    def __init__(self, min_height, min_width, border_mode=cv2.BORDER_CONSTANT,
                 value=None, mask_value=None):
        super(PadIfNeeded, self).__init__(min_height=min_height,
                                          min_width=min_width,
                                          border_mode=border_mode,
                                          value=value,
                                          mask_value=mask_value)

    def update_params(self, params, **kwargs):
        params = super(PadIfNeeded, self).update_params(params, **kwargs)
        rows = params['rows']
        cols = params['cols']

        if rows < self.min_height:
            h_pad_bottom = self.min_height - rows
        else:
            h_pad_bottom = 0

        if cols < self.min_width:
            w_pad_right = self.min_width - cols
        else:
            w_pad_right = 0

        params.update({'pad_top': 0,
                       'pad_bottom': h_pad_bottom,
                       'pad_left': 0,
                       'pad_right': w_pad_right})
        return params

    def get_transform_init_args_names(self):
        return ('min_height', 'min_width',)

# @TRANSFORMS.register_module
# class ToTensor(DualTransform):
#     def __init__(self):
#         super(ToTensor, self).__init__(always_apply=True)
#
#     def apply(self, image, **params):
#         if isinstance(image, np.ndarray):
#             if image.ndim == 2:
#                 image = image[:, :, None]
#             image = torch.from_numpy(image).float()
#             image = image.permute(2, 0, 1)
#         else:
#             raise TypeError('img shoud be np.ndarray. Got {}'
#                             .format(type(image)))
#         return image
#
#     def apply_to_mask(self, image, **params):
#         image = torch.from_numpy(image)
#         return image
#
#     def apply_to_masks(self, masks, **params):
#         masks = [self.apply_to_mask(mask, **params) for mask in masks]
#         return torch.stack(masks, dim=0).squeeze()
#
#     def get_transform_init_args_names(self):
#         return ()
