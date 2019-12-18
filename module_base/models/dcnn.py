import torch.nn as nn
from .registry import MODELS


@MODELS.register_module
class DCNN(object):

    def __init__(self,
                 type='ResNet',
                 depth=9,
                 in_channels=1,
                 out_channels=1,
                 padding_type='reflect',
                 upsampling='bilinear'):
        super(DCNN, self).__init__()
