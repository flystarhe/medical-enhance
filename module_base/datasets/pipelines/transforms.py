import numpy as np
from ..registry import PIPELINES

'''
sitk image: (width, height, depth)
sitk ndarray: (depth, height, width)
numpy ndarray: (height, width, channel)
torch tensor shape: (channel, height, width)
'''


def pad2d(data, shape, fill_value):
    padded_data = np.empty(shape, dtype=data.dtype)

    padded_data[...] = fill_value
    padded_data[:data.shape[0], :data.shape[1]] = data
    return padded_data


@PIPELINES.register_module
class NormalizeCustomize(object):

    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, results):
        input_data = results['input']

        a, b = input_data.min(), input_data.max()
        mean = (a + b) / 2
        std = (b - a) / 2

        input_data = (input_data - mean) / (std + self.eps)
        results['input'] = input_data
        results['norm_cfg'] = dict(mean=mean, std=std)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class NormalizeInstance(object):

    def __init__(self, eps=0.):
        self.eps = eps

    def __call__(self, results):
        input_data = results['input']

        mean = input_data.mean()
        std = input_data.std()

        input_data = (input_data - mean) / (std + self.eps)
        results['input'] = input_data
        results['norm_cfg'] = dict(mean=mean, std=std)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class RandomCrop(object):

    def __init__(self, crop_size):
        self.crop_size = crop_size

    def __call__(self, results):
        input_data = results['input']

        assert 0 < self.crop_size <= min(input_data.shape)
        crop_y = np.random.randint(0, input_data.shape[0] - self.crop_size + 1)
        crop_x = np.random.randint(0, input_data.shape[1] - self.crop_size + 1)
        patch = np.array([crop_x, crop_y, crop_x + self.crop_size, crop_y + self.crop_size])

        # crop the image
        input_data = input_data[patch[1]:patch[3], patch[0]:patch[2]]
        results['input'] = input_data
        results['ori_shape'] = input_data.shape

        # adjust boxes
        if 'gt_boxes' in results:
            boxes = results['gt_boxes']
            boxes[:, 2:] = boxes[:, 2:].clip(max=patch[2:])
            boxes[:, :2] = boxes[:, :2].clip(min=patch[:2])
            boxes -= np.tile(patch[:2], 2)

            valid_inds = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            if not np.any(valid_inds):
                return None

            results['gt_boxes'] = boxes[valid_inds, :]

        # adjust masks
        if 'gt_masks' in results:
            valid_masks = []
            for i in np.where(valid_inds)[0]:
                valid_masks.append(results['gt_masks'][i][patch[1]:patch[3], patch[0]:patch[2]])
            results['gt_masks'] = valid_masks

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(crop_size={})'.format(self.crop_size)


@PIPELINES.register_module
class Pad(object):

    def __init__(self, size_divisor=32, fill_value=0):
        self.size_divisor = size_divisor
        self.fill_value = fill_value

    def __call__(self, results):
        input_data = results['input']

        pad_shape = tuple(int(np.ceil(v / self.size_divisor)) * self.size_divisor for v in input_data.shape)
        padded_data = pad2d(input_data, pad_shape, self.fill_value)
        results['input'] = padded_data
        results['pad_shape'] = padded_data.shape

        # adjust masks
        if 'gt_masks' in results:
            pad_shape = results['pad_shape']
            padded_masks = [pad2d(mask, pad_shape, 0) for mask in results['gt_masks']]
            results['gt_masks'] = np.stack(padded_masks, axis=0)

        return results

    def __repr__(self):
        return self.__class__.__name__ + '(size_divisor={}, fill_value={})'.format(self.size_divisor, self.fill_value)
