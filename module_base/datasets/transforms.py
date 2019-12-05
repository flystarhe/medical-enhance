import numpy as np
from .registry import PIPELINES

'''
sitk image: (width, height, depth)
sitk ndarray: (depth, height, width)
numpy ndarray: (height, width, channel)
torch tensor shape: (channel, height, width)
'''


@PIPELINES.register_module
class RandomCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        data = results['data']

        assert 0 < self.crop_size <= min(data.shape)
        crop_y = np.random.randint(0, data.shape[0] - self.crop_size + 1)
        crop_x = np.random.randint(0, data.shape[1] - self.crop_size + 1)
        patch = np.array([crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size])

        # crop the image
        data = data[patch[1]:patch[3], patch[0]:patch[2]]
        results['data'] = data
        results['ori_shape'] = data.shape

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


@PIPELINES.register_module
class NormalizeCustomize(object):

    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, results):
        data = results['data']

        a, b = data.min(), data.max()
        mean = (a + b) / 2
        std = (b - a) / 2

        data = (data - mean) / (std + self.eps)
        results['data'] = data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class NormalizeInstance(object):

    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, results):
        data = results['data']

        mean = data.mean()
        std = data.std()

        data = (data - mean) / (std + self.eps)
        results['data'] = data

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class Pad(object):

    def __init__(self, size_divisor=32, fill_value=0):
        self.size_divisor = size_divisor
        self.fill_value = fill_value

    def __call__(self, results):
        data = results['data']

        new_shape = tuple(int(np.ceil(v / self.size_divisor)) * self.size_divisor for v in data.shape)
        pad_data = np.empty(new_shape, dtype=data.dtype)
        pad_data[...] = self.fill_value

        pad_data[:data.shape[0], :data.shape[1]] = data
        results['data'] = pad_data
        results['pad_shape'] = pad_data.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(size_divisor={}, fill_value={})'.format(self.size_divisor, self.fill_value)
