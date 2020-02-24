import torch.nn as nn
from .registry import MODELS


@MODELS.register_module
class DCNN(object):

    def __init__(self,
                 depth=9,
                 block=None,
                 in_channels=1,
                 out_channels=1,
                 padding_type='reflect',
                 downsampling='max_pool',
                 upsampling='bilinear'):
        super(DCNN, self).__init__()

        nf = 64
        layers = list()
        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(in_channels, nf, kernel_size=7, stride=1, padding=0, bias=True))
        layers.append(nn.InstanceNorm2d(nf))
        layers.append(nn.ReLU(True))

        for i in range(2):
            layers.append(nn.Conv2d(nf, nf * 2, kernel_size=4, stride=2, padding=1, bias=True))
            layers.append(nn.InstanceNorm2d(nf * 2))
            layers.append(nn.ReLU(True))
            nf = nf * 2

        for i in range(depth):
            layers.append(ResBlock(nf, padding_type, 0, True))

        for i in range(2):
            layers.append(nn.ConvTranspose2d(nf, nf // 2, kernel_size=4, stride=2, padding=1, bias=True))
            layers.append(nn.InstanceNorm2d(nf // 2))
            layers.append(nn.ReLU(True))
            nf = nf // 2

        layers.append(nn.ReflectionPad2d(3))
        layers.append(nn.Conv2d(nf, out_channels, kernel_size=7, stride=1, padding=0))
        layers.append(nn.Tanh())

        self.model = nn.Sequential(*layers)


def down_sample_layers(in_channels, out_channels, kernel_size, padding_type='reflect'):
    pass


def up_sample_layers(in_channels, out_channels, kernel_size, padding_type='reflect', upsampling='bilinear'):
    layers = []

    p = 0
    if padding_type == "reflect":
        layers.append(nn.ReflectionPad2d(1))
    elif padding_type == "replicate":
        layers.append(nn.ReplicationPad2d(1))
    elif padding_type == "zero":
        p = 1
    else:
        raise NotImplementedError("padding [%s] is not implemented" % padding_type)

    if upsampling is not None:
        layers.append(nn.Upsample(scale_factor=2, mode=upsampling))
        layers.append(nn.ReflectionPad2d(1))


class ResBlock(nn.Module):
    def __init__(self, dim, padding_type, drop_prob, use_bias):
        super(ResBlock, self).__init__()
        conv_block = list()

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias))
        conv_block.append(nn.InstanceNorm2d(dim))
        conv_block.append(nn.ReLU(True))

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias))
        conv_block.append(nn.InstanceNorm2d(dim))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        y = x + self.conv_block(x)
        return y


class NeXtBlock(nn.Module):
    def __init__(self, dim, padding_type, drop_prob, use_bias):
        super(NeXtBlock, self).__init__()
        conv_block = list()

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias))
        conv_block.append(nn.InstanceNorm2d(dim))
        conv_block.append(nn.ReLU(True))

        if drop_prob:
            conv_block.append(nn.Dropout(drop_prob))

        p = 0
        if padding_type == "reflect":
            conv_block.append(nn.ReflectionPad2d(1))
        elif padding_type == "replicate":
            conv_block.append(nn.ReplicationPad2d(1))
        elif padding_type == "zero":
            p = 1
        else:
            raise NotImplementedError("padding [%s] is not implemented" % padding_type)

        conv_block.append(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=p, bias=use_bias))
        conv_block.append(nn.InstanceNorm2d(dim))

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        y = x + self.conv_block(x)
        return y
