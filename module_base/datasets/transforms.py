import torch
import numpy as np
from .registry import PIPELINES


def to_tensor(data):
    if isinstance(data, torch.Tensor):
        return data
    elif isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, (list, tuple)):
        return torch.tensor(data)
    elif isinstance(data, int):
        return torch.LongTensor([data])
    elif isinstance(data, float):
        return torch.FloatTensor([data])
    else:
        raise TypeError('type {} cannot be converted to tensor.'.format(type(data)))


@PIPELINES.register_module
class ToTensor(object):

    def __init__(self, keys):
        self.keys = keys

    def __call__(self, results):
        for key in self.keys:
            results[key] = to_tensor(results[key])
        return results

    def __repr__(self):
        return self.__class__.__name__ + '(keys={})'.format(self.keys)


class RandomCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        data = results['data']

        assert 0 < self.crop_size <= min(data.shape)
        crop_y0 = np.random.randint(0, data.shape[0] - self.crop_size + 1)
        crop_x0 = np.random.randint(0, data.shape[1] - self.crop_size + 1)
        patch = np.array([crop_x0, crop_y0, crop_x0 + self.crop_size, crop_y0 + self.crop_size])

        # crop the image
        data = data[patch[1]:patch[3], patch[0]:patch[2]]
        results['data'] = data

        # adjust boxes
        if 'boxes' in results:
            boxes = results['boxes']
            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
            boxes -= np.tile(patch[:2], 2)

            valid_inds = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not np.any(valid_inds):
                return None
            results['boxes'] = boxes[valid_inds, :]

        # adjust masks
        if 'masks' in results:
            valid_masks = []
            for i in np.where(valid_inds)[0]:
                valid_masks.append(results['masks'][i][patch[1]:patch[3], patch[0]:patch[2]])
            results['gt_masks'] = valid_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(self.crop_size)
