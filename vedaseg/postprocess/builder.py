from ..utils import build_from_cfg
from .registry import POSTPROCESS


def build_postprocess(cfg):
    criterion = build_from_cfg(cfg, POSTPROCESS, mode='registry')
    return criterion
