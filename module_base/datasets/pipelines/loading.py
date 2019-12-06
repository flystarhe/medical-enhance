import os.path as osp
import SimpleITK as sitk
import pycocotools.mask as maskUtils
from ..registry import PIPELINES

'''
pip install SimpleITK

sitk image: (width, height, depth)
sitk ndarray: (depth, height, width)
'''


@PIPELINES.register_module
class LoadDicomFromFile(object):

    def __init__(self):
        pass

    def __call__(self, results):
        if results['data_root'] is not None:
            filename = osp.join(results['data_root'], results['img_info']['filename'])
        else:
            filename = results['img_info']['filename']

        itk_img = sitk.ReadImage(filename, sitk.sitkFloat32)
        input_data = sitk.GetArrayFromImage(itk_img)[0]  # (height, width)

        results['filename'] = filename
        results['input'] = input_data
        results['ori_shape'] = input_data.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'


@PIPELINES.register_module
class LoadAnnotations(object):

    def __init__(self, with_bbox=False, with_mask=False, poly2mask=True):
        self.with_bbox = with_bbox
        self.with_mask = with_mask
        self.poly2mask = poly2mask

    def _load_boxes(self, results):
        results['gt_boxes'] = results['ann_info']['boxes']
        return results

    def _poly2mask(self, mask_ann, img_h, img_w):
        if isinstance(mask_ann, list):
            # polygon -- a single object might consist of multiple parts
            # we merge all parts into one mask rle code
            rles = maskUtils.frPyObjects(mask_ann, img_h, img_w)
            rle = maskUtils.merge(rles)
        elif isinstance(mask_ann['counts'], list):
            # uncompressed RLE
            rle = maskUtils.frPyObjects(mask_ann, img_h, img_w)
        else:
            # rle
            rle = mask_ann
        mask = maskUtils.decode(rle)
        return mask

    def _load_masks(self, results):
        h, w = results['ori_shape']
        gt_masks = results['ann_info']['masks']

        if self.poly2mask:
            gt_masks = [self._poly2mask(mask, h, w) for mask in gt_masks]
        results['gt_masks'] = gt_masks

        return results

    def __call__(self, results):
        if self.with_bbox:
            results = self._load_bboxes(results)

        if self.with_mask:
            results = self._load_masks(results)

        return results

    def __repr__(self):
        return self.__class__.__name__
