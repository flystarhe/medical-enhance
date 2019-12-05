import os.path as osp
import SimpleITK as sitk
from .registry import PIPELINES

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
            filename = osp.join(results['data_root'], results['filename'])
        else:
            filename = results['filename']

        itk_img = sitk.ReadImage(filename, sitk.sitkFloat32)
        data = sitk.GetArrayFromImage(itk_img)[0]  # (height, width)

        results['filename'] = filename
        results['data'] = data
        results['ori_shape'] = data.shape

        return results

    def __repr__(self):
        return self.__class__.__name__ + '()'
