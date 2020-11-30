import albumentations as albu

from vedaseg.utils import build_from_cfg
from .registry import TRANSFORMS
from .transforms import Compose


# def build_transform(cfgs):
#     tfs = []
#     for cfg in cfgs:
#         if TRANSFORMS.get(cfg['type']):
#             tf = build_from_cfg(cfg, TRANSFORMS)
#         else:
#             tf = build_from_cfg(cfg, albu, mode='module')
#         tfs.append(tf)
#     aug = albu.Compose(tfs)
#
#     return aug

def build_transform(cfgs):
    tfs = []
    for cfg in cfgs:
        tf = build_from_cfg(cfg, TRANSFORMS)
        tfs.append(tf)
    aug = Compose(tfs)

    return aug
