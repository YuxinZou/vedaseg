import json
import pickle
import random
import torch
import numpy as np
import torch.nn.functional as F

from .inference_runner import InferenceRunner
from ..utils import gather_tensor
from ..metrics import eval_ap


class TestRunner(InferenceRunner):
    def __init__(self, test_cfg, inference_cfg, base_cfg=None):
        super().__init__(inference_cfg, base_cfg)

        self.test_dataloader = self._build_dataloader(test_cfg['data'])
        extra_data = len(self.test_dataloader.dataset) % self.world_size
        self.test_exclude_num = self.world_size - extra_data if extra_data != 0 else 0

        self.tta = test_cfg.get('tta', False)
        self.CLASSES = self.test_dataloader.dataset.CLASSES

        self.all_gts = self.test_dataloader.dataset.get_all_gts()

    def __call__(self):
        self.metric.reset()
        self.model.eval()

        res = {}
        prediction = {}
        model_output = {}

        self.logger.info('Start testing')
        with torch.no_grad():
            for idx, (image, mask, fname) in enumerate(self.test_dataloader):
                if len(image.shape) == 6:
                    outputs = []
                    image = image.transpose(0, 1)
                    mask = mask.squeeze(0)
                    if self.use_gpu:
                        mask = mask.cuda()

                    for i in range(image.shape[0]):
                        img = image[i]
                        if self.use_gpu:
                            img = img.cuda()
                        output = self.model(img)
                        outputs.append(output)
                    output = torch.cat(outputs, 0)
                else:
                    if self.use_gpu:
                        image = image.cuda()
                        mask = mask.cuda()

                    output = self.model(image)

                print(output.shape)
                print(mask.shape)
                #pred, valid_output = self.postprocess(output, mask)
                #model_output.update({fname[0]: valid_output})
                prediction.update({fname[0]: [output.cpu(), mask.cpu()]})
                output = self.compute(output)
                output, shape_max = gather_tensor(output)
                mask, shape_max = gather_tensor(mask)

                if idx + 1 == len(
                        self.test_dataloader) and self.test_exclude_num > 0:
                    output = output[:-self.test_exclude_num * shape_max]
                    mask = mask[:-self.test_exclude_num * shape_max]
                self.metric(output.cpu().numpy(), mask.cpu().numpy())
                res = self.metric.accumulate()
                self.logger.info('Test, Iter {}, {}'.format(
                    idx + 1,
                    ', '.join(['{}: {}'.format(k, np.round(v, 4)) for k, v in
                               res.items()])))
        self.logger.info('Test Result: {}'.format(', '.join(
            ['{}: {}'.format(k, np.round(v, 4)) for k, v in res.items()])))
        
        self.save_prediction(prediction)
        #plain_detections = self.get_predictions(prediction)
        #self.evaluate(plain_detections)
        return res

    def save_prediction(self, predition):
        with open(self.pickle_save, 'wb') as pf:
            pickle.dump(predition, pf)

    def gt2pred(self):
        gt = self.all_gts
        num_classes = len(self.CLASSES)
        plain_detections = {}
        for class_idx in range(num_classes):
            detection_list = []
            for cls, anno in gt.items():
                if cls == class_idx:
                    for k, v in anno.items():
                        x = [[k, cls] + det + [random.random()] for det in v]
                        detection_list.extend(x)
                plain_detections[class_idx] = detection_list
        return plain_detections

    def get_predictions(self, result):
        num_classes = len(self.CLASSES)
        plain_detections = {}
        for class_idx in range(num_classes):
            detection_list = []
            for video, dets in result.items():
                x = [[video, class_idx] + det['segment'] + [det['score']] for
                     det in dets if det['label'] == class_idx]
                detection_list.extend(x)
            plain_detections[class_idx] = detection_list
        return plain_detections

    def _tta_compute(self, image):
        b, c, h, w = image.size()
        probs = []
        for scale, bias in zip(self.tta['scales'], self.tta['biases']):
            new_h, new_w = int(h * scale + bias), int(w * scale + bias)
            new_img = F.interpolate(image, size=(new_h, new_w),
                                    mode='bilinear', align_corners=True)
            output = self.model(new_img)
            probs.append(output)

            if self.tta['flip']:
                flip_img = new_img.flip(3)
                flip_output = self.model(flip_img)
                prob = flip_output.flip(3)
                probs.append(prob)

        if self.multi_label:
            prob = torch.stack(probs, dim=0).sigmoid().mean(dim=0)
            prob = torch.where(prob >= 0.5,
                               torch.full_like(prob, 1),
                               torch.full_like(prob, 0)).long()  # b c h w
        else:
            prob = torch.stack(probs, dim=0).softmax(dim=2).mean(dim=0)
            _, prob = torch.max(prob, dim=1)  # b h w
        return prob

    def evaluate(self, plain_detections):
        # get gts
        all_gts = self.all_gts
        for class_idx in range(len(self.CLASSES)):
            if class_idx not in all_gts:
                all_gts[class_idx] = dict()

        eval_results = {}
        # plain = self.gt2pred()
        iou_range = np.arange(0.1, 1.0, .1)
        ap_values = eval_ap(plain_detections, all_gts, iou_range)
        print(ap_values)
        map_ious = ap_values.mean(axis=0)
        self.logger.info('Evaluation finished')

        for iou, map_iou in zip(iou_range, map_ious):
            eval_results[f'mAP@{iou:.02f}'] = map_iou
        print(eval_results)
        return eval_results
