import torch
import numpy as np

from ..models import build_model
from ..utils import load_checkpoint
from .base import Common


class InferenceRunner(Common):
    def __init__(self, inference_cfg, base_cfg=None):
        inference_cfg = inference_cfg.copy()
        base_cfg = {} if base_cfg is None else base_cfg.copy()

        base_cfg['gpu_id'] = inference_cfg.pop('gpu_id')
        super().__init__(base_cfg)

        self.multi_label = inference_cfg.get('multi_label', False)
        self.thres = inference_cfg.get('threshold', 0.5)
        # build inference transform
        self.transform = self._build_transform(inference_cfg['transforms'])

        # build model
        self.model = self._build_model(inference_cfg['model'])
        self.model.eval()

        # build postprocess
        self.postprocess = self._build_postprocess(inference_cfg['postprocess']) if inference_cfg.get('postprocess') else None

    def load_checkpoint(self, filename, map_location='default', strict=True):
        self.logger.info('Load checkpoint from {}'.format(filename))

        if map_location == 'default':
            if self.use_gpu:
                device_id = torch.cuda.current_device()
                map_location = lambda storage, loc: storage.cuda(device_id)
            else:
                map_location = 'cpu'

        return load_checkpoint(self.model, filename, map_location, strict)

    def _build_model(self, cfg):
        self.logger.info('Build model')

        model = build_model(cfg)

        if torch.cuda.is_available():
            if self.distribute:
                model = torch.nn.parallel.DistributedDataParallel(
                    model.cuda(),
                    device_ids=[torch.cuda.current_device()],
                    broadcast_buffers=True,
                    find_unused_parameters=True,
                )
                self.logger.info('Using distributed training')
            else:
                if torch.cuda.device_count() > 1:
                    model = torch.nn.DataParallel(model)
                model.cuda()
        return model

    def compute(self, output):
        if self.multi_label:
            output = output.sigmoid()

            output[:,:-1,:] = output[:,:-1,:] * output[:,-1,:].unsqueeze(1)
            output = torch.where(output >= self.thres,
                                 torch.full_like(output, 1),
                                 torch.full_like(output, 0)).long()

        else:
            output = output.softmax(dim=1)
            _, output = torch.max(output, dim=1)
        return output

    def __call__(self, image, masks):
        with torch.no_grad():
            # image = self.transform(image=image, masks=masks)['image']
            if len(image.shape) == 5:
                image = image.unsqueeze(1)

            outputs = []
            for i in range(image.shape[0]):

                img = image[i]
                if self.use_gpu:
                    img = img.cuda()

                output = self.model(img)
                output = self.compute(output)
                output = output.squeeze().cpu().numpy()
                outputs.append(output)

        return np.array(outputs)
