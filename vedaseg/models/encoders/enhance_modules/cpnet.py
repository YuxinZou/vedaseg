import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import ENHANCE_MODULES
from ...utils.act import build_act_layer
from ...utils.norm import build_norm_layer
from ...weight_init import init_weights

logger = logging.getLogger()


class CPNetBaseConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, norm_layer,
                 act_layer):
        modules = [
            nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size,
                      stride=stride,
                      padding=dilation,
                      dilation=dilation,
                      bias=False),
            norm_layer(out_channels),
            act_layer(out_channels)
        ]
        super(CPNetBaseConv, self).__init__(*modules)


@ENHANCE_MODULES.register_module
class CPNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, from_layer='c_end', norm_cfg=None, act_cfg=None):
        super(CPNetConv, self).__init__()

        if norm_cfg is None:
            norm_cfg = dict(type='BN1d')
        norm_layer = partial(build_norm_layer, norm_cfg, layer_only=True)

        if act_cfg is None:
            act_cfg = dict(type='Relu', inplace=True)
        act_layer = partial(build_act_layer, act_cfg, layer_only=True)

        self.from_layer =from_layer

        self.layer1 = CPNetBaseConv(in_channels, out_channels, 3, 2, 1,
                                    norm_layer, act_layer)
        self.layer2 = CPNetBaseConv(out_channels, out_channels, 3, 2, 1,
                                    norm_layer, act_layer)
        self.layer3 = CPNetBaseConv(out_channels, out_channels, 3, 2, 1,
                                    norm_layer, act_layer)
        self.layer4 = CPNetBaseConv(out_channels, out_channels, 3, 2, 1,
                                    norm_layer, act_layer)

        init_weights(self.modules())

    def forward(self, feats):
        feats_ = feats.copy()
        x0 = feats_[self.from_layer]

        x1 = self.layer1(x0)  # 4
        feats_['c2'] = x1
        x2 = self.layer2(x1)  # 8
        feats_['c3'] = x2
        x3 = self.layer3(x2)  # 16
        feats_['c4'] = x3
        x4 = self.layer4(x3)  # 32
        feats_['c5'] = x4

        return feats_
