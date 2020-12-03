
import torch
import torch.nn as nn
import torch.nn.functional as F

from .registry import CRITERIA
# from .utils import weight_reduce_loss


@CRITERIA.register_module
class RectBCELoss(nn.Module):

    def __init__(self,
                 num_class=21,
                 reduction='mean',
                 ignore_index=255,
                 loss_weight=1.0):
        super(RectBCELoss, self).__init__()
        self.num_class = num_class
        self.reduction = reduction
        self.ignore_index = ignore_index
        self.loss_weight = loss_weight


    def forward(self,
                cls_score,
                label,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)

        weight = cls_score.new_ones(label.shape, dtype=torch.float)
        weight[label == 1] = 5

        mask = (label != self.ignore_index)

        # import pdb
        # pdb.set_trace()

        # loss_cls = self.loss_weight * F.binary_cross_entropy_with_logits(cls_score[mask], label.float()[mask])

        # loss_cls = self.loss_weight * F.binary_cross_entropy_with_logits(cls_score[mask], label.float()[mask], weight=weight[mask])

        loss_cls = self.loss_weight * F.binary_cross_entropy_with_logits(
            cls_score[mask], label.float()[mask], weight=weight[mask],
            reduction=reduction)

        return loss_cls