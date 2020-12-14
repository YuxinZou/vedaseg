import cv2
import torch
import numpy as np

from vedaseg.datasets import build_dataset
from vedaseg.transforms import build_transform
from vedaseg.dataloaders import build_dataloader
from vedaseg.utils.config import Config


def tensor_to_img(t_img: torch.Tensor):
    t_img = t_img[0]
    if t_img.ndim == 3:
        t_img = t_img.permute(1, 2, 0).cpu().numpy()
    else:
        t_img = t_img.cpu().numpy()
    t_img = (t_img - np.min(t_img)) / (np.max(t_img) - np.min(t_img))
    t_img = (t_img * 255).clip(0, 255).astype(np.uint8)

    return t_img


def main():
    cfg = Config.fromfile('./configs/coco_unet_test.py')
    transforms = build_transform(cfg['train']['data']['train']['transforms'])

    dataset = build_dataset(cfg['train']['data']['train']['dataset'],
                            dict(transform=transforms))
    print(dataset)
    """data = datasets[0][0]
    for item in data['text_mask']:
        print(item.shape)
        cv2.imshow('text_mask', tensor_to_img(item))
        cv2.waitKey()
    for idx, item in enumerate(data['kernels_mask']):
        print(item.shape)
        cv2.imshow(f'kernel_mask_{idx}', tensor_to_img(item))
        cv2.waitKey()"""
    dataloader = build_dataloader(False,
                                  1,
                                  cfg['train']['data']['train']['dataloader'],
                                  dict(dataset=dataset,
                                       sampler=None))
    for idx, (img, mask) in enumerate(dataloader):
        # print(img.shape)
        cv2.imshow('a', tensor_to_img(img))
        # cv2.imshow('aa', tensor_to_img(kernels_mask[i]))
        cv2.waitKey()
        mask = mask.squeeze().numpy().astype(np.uint8)
        # print(np.unique(mask))
        cv2.imshow('ss', mask)
        cv2.waitKey()


if __name__ == '__main__':
    main()
