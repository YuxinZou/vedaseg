import numpy as np

from .base import BaseMetric
from .registry import METRICS


class Compose:
    def __init__(self, metrics):
        self.metrics = metrics

    def reset(self):
        for m in self.metrics:
            m.reset()

    def accumulate(self):
        res = dict()
        for m in self.metrics:
            mtc = m.accumulate()
            res.update(mtc)
        return res

    def __call__(self, pred, target):
        for m in self.metrics:
            m(pred, target)


class ConfusionMatrix(BaseMetric):
    """
    Calculate confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__()

    def reset(self):
        self.cfsmtx = np.zeros((self.num_classes,) * 2)

    def compute(self, pred, target):
        mask = (target >= 0) & (target < self.num_classes)

        self.current_state = np.bincount(
            self.num_classes * target[mask].astype('int') + pred[mask],
            minlength=self.num_classes ** 2
        ).reshape(self.num_classes, self.num_classes)
        return self.current_state

    def update(self, n=1):
        self.cfsmtx += self.current_state

    def accumulate(self):
        accumulate_state = {
            'confusion matrix': self.cfsmtx
        }
        return accumulate_state


class MultiLabelConfusionMatrix(BaseMetric):
    """
    Calculate confusion matrix for multi label segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        self.binary = 2
        self.current_state = np.zeros(
            (self.num_classes, self.binary, self.binary))
        super().__init__()

    @staticmethod
    def _check_match(pred, target):
        assert pred.shape == target.shape, \
            "pred should habe same shape with target"

    def reset(self):
        self.cfsmtx = np.zeros((self.num_classes, self.binary, self.binary))

    def compute(self, pred, target):
        mask = (target >= 0) & (target < self.binary)
        for i in range(self.num_classes):
            # pred_index_sub = pred[:, i, :, :]
            # target_sub = target[:, i, :, :]
            # mask_sub = mask[:, i, :, :]
            # self.current_state[i, :, :] = np.bincount(
            #     self.binary * target_sub[mask_sub].astype('int') +
            #     pred_index_sub[mask_sub], minlength=self.binary ** 2
            # ).reshape(self.binary, self.binary)
            pred_index_sub = pred[:, i, :]
            target_sub = target[:, i, :]
            mask_sub = mask[:, i, :]
            # print(pred_index_sub[mask_sub].max(), target_sub[mask_sub].max(), mask_sub.max())
            self.current_state[i, :, :] = np.bincount(
                self.binary * target_sub[mask_sub].astype('int') +
                pred_index_sub[mask_sub], minlength=self.binary ** 2
            ).reshape(self.binary, self.binary)
        return self.current_state

    def update(self, n=1):
        self.cfsmtx += self.current_state

    def accumulate(self):
        accumulate_state = {
            'confusion matrix': self.cfsmtx
        }
        return accumulate_state


@METRICS.register_module
class Accuracy(ConfusionMatrix):
    """
    Calculate accuracy based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
        average (str): {'pixel', 'class'}
            'pixel':
                calculate pixel wise average accuracy
            'class':
                calculate class wise average accuracy
    """

    def __init__(self, num_classes, average='pixel'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):

        assert self.average in ('pixel', 'class'), \
            'Accuracy only support "pixel" & "class" wise average'

        if self.average == 'pixel':
            accuracy = self.cfsmtx.diagonal().sum() / (
                    self.cfsmtx.sum() + 1e-15)

        elif self.average == 'class':
            accuracy_class = self.cfsmtx.diagonal() / self.cfsmtx.sum(axis=1)
            accuracy = np.nanmean(accuracy_class)

        accumulate_state = {
            'accuracy': accuracy
        }
        return accumulate_state


@METRICS.register_module
class MultiLabelAccuracy(MultiLabelConfusionMatrix):
    """
    Calculate accuracy based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
        average (str): {'pixel', 'class'}
            'pixel':
                calculate pixel wise average accuracy
            'class':
                calculate class wise average accuracy
    """

    def __init__(self, num_classes, average='pixel'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):

        assert self.average in ('pixel', 'class'), \
            'Accuracy only support "pixel" & "class" wise average'

        if self.average == 'pixel':
            accuracy = self.cfsmtx.diagonal().sum() / (
                    self.cfsmtx.sum() + 1e-15)

        elif self.average == 'class':
            accuracy_class = self.cfsmtx.diagonal() / self.cfsmtx.sum(axis=1)
            accuracy = np.nanmean(accuracy_class)

        accumulate_state = {
            'accuracy': accuracy
        }
        return accumulate_state


@METRICS.register_module
class MultiLabelIoU(MultiLabelConfusionMatrix):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def accumulate(self):
        ious = self.cfsmtx.diagonal(axis1=1, axis2=2) / (
                self.cfsmtx.sum(axis=1) + self.cfsmtx.sum(axis=2) -
                self.cfsmtx.diagonal(axis1=1, axis2=2) + np.finfo(
            np.float32).eps)

        accumulate_state = {
            'IoUs': ious[:, 1]
        }
        return accumulate_state


@METRICS.register_module
class MultiLabelMIoU(MultiLabelIoU):
    def __init__(self, num_classes):
        super().__init__(num_classes)

    def accumulate(self):
        ious = (super().accumulate())['IoUs']

        accumulate_state = {
                'mIoU': np.nanmean(ious[:-1])
        }
        return accumulate_state


@METRICS.register_module
class IoU(ConfusionMatrix):
    """
    Calculate IoU for each class based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        ious = self.cfsmtx.diagonal() / (
                self.cfsmtx.sum(axis=0) + self.cfsmtx.sum(axis=1) -
                self.cfsmtx.diagonal() + np.finfo(np.float32).eps)
        accumulate_state = {
            'IoUs': ious
        }
        return accumulate_state


@METRICS.register_module
class MIoU(IoU):
    """
    Calculate mIoU based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
        average (str): {'equal', 'frequency_weighted'}
            'equal':
                calculate mIoU in an equal class wise average manner
            'frequency_weighted':
                calculate mIoU in an frequency weighted class wise average manner
    """

    def __init__(self, num_classes, average='equal'):
        self.num_classes = num_classes
        self.average = average
        super().__init__(num_classes=self.num_classes)

    def accumulate(self):
        assert self.average in ('equal', 'frequency_weighted'), \
            'mIoU only support "equal" & "frequency_weighted" average'

        ious = (super().accumulate())['IoUs']

        if self.average == 'equal':
            miou = np.nanmean(ious)
        elif self.average == 'frequency_weighted':
            pos_freq = self.cfsmtx.sum(axis=1) / self.cfsmtx.sum()
            miou = (pos_freq[pos_freq > 0] * ious[pos_freq > 0]).sum()

        accumulate_state = {
            'mIoU': miou
        }
        return accumulate_state


class DiceScore(ConfusionMatrix):
    """
    Calculate dice score based on confusion matrix for segmentation
    Args:
        num_classes (int): number of classes.
    """

    def __init__(self, num_classes):
        self.num_classes = num_classes
        super().__init__(self.num_classes)

    def accumulate(self):
        dice_score = 2.0 * self.cfsmtx.diagonal() / (self.cfsmtx.sum(axis=1) +
                                                     self.cfsmtx.sum(axis=0) +
                                                     np.finfo(np.float32).eps)

        accumulate_state = {
            'dice_score': dice_score
        }
        return accumulate_state


def eval_ap(detections, gt_by_cls, iou_range):
    """Evaluate average precisions.

    Args:
        detections (dict): Results of detections.
        gt_by_cls (dict): Information of groudtruth.
        iou_range (list): Ranges of iou.

    Returns:
        list: Average precision values of classes at ious.
    """
    ap_values = np.zeros((len(detections), len(iou_range)))

    for iou_idx, min_overlap in enumerate(iou_range):
        for class_idx in range(len(detections)):
            ap = average_precision_at_temporal_iou(gt_by_cls[class_idx],
                                                   detections[class_idx],
                                                   [min_overlap])
            ap_values[class_idx, iou_idx] = ap

    return ap_values


def average_precision_at_temporal_iou(ground_truth,
                                      prediction,
                                      temporal_iou_thresholds=(np.linspace(
                                          0.5, 0.95, 10))):
    """Compute average precision (in detection task) between ground truth and
    predicted data frames. If multiple predictions match the same predicted
    segment, only the one with highest score is matched as true positive. This
    code is greatly inspired by Pascal VOC devkit.

    Args:
        ground_truth (dict): Dict containing the ground truth instances.
            Key: 'video_id'
            Value (np.ndarry): 1D array of 't-start' and 't-end'.
        proposals (np.ndarray): 2D array containing the information of proposal
            instances, including 'video_id', 'class_id', 't-start', 't-end' and
            'score'.
        temporal_iou_thresholds (np.ndarray): 1D array with temporal_iou
            thresholds. Default: np.linspace(0.5, 0.95, 10).

    Returns:
        np.ndarray: 1D array of average precision score.
    """
    ap = np.zeros(len(temporal_iou_thresholds), dtype=np.float32)
    if len(prediction) < 1:
        return ap

    num_gts = 0.
    lock_gt = dict()
    for key in ground_truth:
        lock_gt[key] = np.ones(
            (len(temporal_iou_thresholds), len(ground_truth[key]))) * -1
        num_gts += len(ground_truth[key])

    # Sort predictions by decreasing score order.
    prediction = np.array(prediction)
    scores = prediction[:, 4].astype(float)
    sort_idx = np.argsort(scores)[::-1]
    prediction = prediction[sort_idx]
    # Initialize true positive and false positive vectors.
    tp = np.zeros((len(temporal_iou_thresholds), len(prediction)),
                  dtype=np.int32)
    fp = np.zeros((len(temporal_iou_thresholds), len(prediction)),
                  dtype=np.int32)

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in enumerate(prediction):
        # print(this_pred)

        # Check if there is at least one ground truth in the video.
        if (this_pred[0] in ground_truth):
            this_gt = np.array(ground_truth[this_pred[0]], dtype=float)
        else:
            fp[:, idx] = 1
            continue

        # print(f'this gt{this_gt}')
        t_iou = pairwise_temporal_iou(this_pred[2:4].astype(float), this_gt)
        # We would like to retrieve the predictions with highest t_iou score.
        t_iou_sorted_idx = t_iou.argsort()[::-1]
        for t_idx, t_iou_threshold in enumerate(temporal_iou_thresholds):
            for jdx in t_iou_sorted_idx:
                if t_iou[jdx] < t_iou_threshold:
                    fp[t_idx, idx] = 1
                    break
                if lock_gt[this_pred[0]][t_idx, jdx] >= 0:
                    continue
                # Assign as true positive after the filters above.
                tp[t_idx, idx] = 1
                lock_gt[this_pred[0]][t_idx, jdx] = idx
                break

            if fp[t_idx, idx] == 0 and tp[t_idx, idx] == 0:
                fp[t_idx, idx] = 1
    tp_cumsum = np.cumsum(tp, axis=1).astype(np.float32)
    fp_cumsum = np.cumsum(fp, axis=1).astype(np.float32)
    recall_cumsum = tp_cumsum / num_gts

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    for t_idx in range(len(temporal_iou_thresholds)):
        ap[t_idx] = interpolated_precision_recall(precision_cumsum[t_idx, :],
                                                  recall_cumsum[t_idx, :])

    return ap


def interpolated_precision_recall(precision, recall):
    """Interpolated AP - VOCdevkit from VOC 2011.

    Args:
        precision (np.ndarray): The precision of different thresholds.
        recall (np.ndarray): The recall of different thresholds.

    Returns：
        float: Average precision score.
    """
    mprecision = np.hstack([[0], precision, [0]])
    mrecall = np.hstack([[0], recall, [1]])
    for i in range(len(mprecision) - 1)[::-1]:
        mprecision[i] = max(mprecision[i], mprecision[i + 1])
    idx = np.where(mrecall[1::] != mrecall[0:-1])[0] + 1
    ap = np.sum((mrecall[idx] - mrecall[idx - 1]) * mprecision[idx])
    return ap


def pairwise_temporal_iou(candidate_segments,
                          target_segments,
                          calculate_overlap_self=False):
    """Compute intersection over union between segments.
    Args:
        candidate_segments (np.ndarray): 1-dim/2-dim array in format
            ``[init, end]/[m x 2:=[init, end]]``.
        target_segments (np.ndarray): 2-dim array in format
            ``[n x 2:=[init, end]]``.
        calculate_overlap_self (bool): Whether to calculate overlap_self
            (union / candidate_length) or not. Default: False.
    Returns:
        t_iou (np.ndarray): 1-dim array [n] /
            2-dim array [n x m] with IoU ratio.
        t_overlap_self (np.ndarray, optional): 1-dim array [n] /
            2-dim array [n x m] with overlap_self, returns when
            calculate_overlap_self is True.
    """
    candidate_segments_ndim = candidate_segments.ndim
    if target_segments.ndim != 2 or candidate_segments_ndim not in [1, 2]:
        raise ValueError('Dimension of arguments is incorrect')

    if candidate_segments_ndim == 1:
        candidate_segments = candidate_segments[np.newaxis, :]

    n, m = target_segments.shape[0], candidate_segments.shape[0]
    t_iou = np.empty((n, m), dtype=np.float32)
    if calculate_overlap_self:
        t_overlap_self = np.empty((n, m), dtype=np.float32)

    for i in range(m):
        candidate_segment = candidate_segments[i, :]
        tt1 = np.maximum(candidate_segment[0], target_segments[:, 0])
        tt2 = np.minimum(candidate_segment[1], target_segments[:, 1])
        # Intersection including Non-negative overlap score.
        segments_intersection = (tt2 - tt1).clip(0)
        # Segment union.
        segments_union = ((target_segments[:, 1] - target_segments[:, 0]) +
                          (candidate_segment[1] - candidate_segment[0]) -
                          segments_intersection)
        # Compute overlap as the ratio of the intersection
        # over union of two segments.
        t_iou[:, i] = (segments_intersection.astype(float) / segments_union)
        if calculate_overlap_self:
            candidate_length = candidate_segment[1] - candidate_segment[0]
            t_overlap_self[:, i] = (
                    segments_intersection.astype(float) / candidate_length)

    if candidate_segments_ndim == 1:
        t_iou = np.squeeze(t_iou, axis=1)
    if calculate_overlap_self:
        if candidate_segments_ndim == 1:
            t_overlap_self = np.squeeze(t_overlap_self, axis=1)
        return t_iou, t_overlap_self

    return t_iou
