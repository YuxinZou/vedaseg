import torch
import numpy as np

from .registry import POSTPROCESS


@POSTPROCESS.register_module
class SimplePostProcess:
    def __init__(self, threshold=0.5, mini_len=5, mini_merge=2,
                 ignore_label=255, fps=10):
        self.threshold = threshold
        self.mini_len = mini_len
        self.mini_merge = mini_merge
        self.ignore_label = ignore_label
        self.fps = fps

    def generate_sequence(self, label, data, fps=10):
        result = []
        start = 0
        idx = 0
        index = np.where(data == 1)[0]
        if len(index) == 0:
            return []
        for i in range(1, len(index)):
            idx += 1
            if index[i] > index[i - 1] + self.mini_merge:
                result.append(dict(segment=[float(index[start] / fps),
                                            float(index[idx-1] / fps)],
                                   label=label))
                start = i
        result.append(dict(segment=[float(index[start] / fps),
                                    float(index[idx] / fps)],
                           label=label))
        return result

    def __call__(self, output, mask, classes):
        res = []
        output = output.sigmoid()
        output = output.cpu().numpy()
        mask = mask.cpu().numpy()
        output = np.hstack(output)
        mask = np.hstack(mask)
        valid = mask[-1] != 255
        valid_output = output[:, valid]
        valid_output = valid_output * valid_output[-1]
        valid_output = valid_output[:-1, :]
        valid_output = np.where(valid_output >= self.threshold, 1, 0)
        for i in range(valid_output.shape[0]):
            res.extend(self.generate_sequence(classes[i], valid_output[i]))
        return res
