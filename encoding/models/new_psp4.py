###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Module, Sequential, Conv2d, ReLU, AdaptiveAvgPool2d, BCELoss, CrossEntropyLoss
from torch.nn.functional import upsample

from .base import BaseNet
from .fcn import FCNHead
# from ..nn import PyramidPooling

class new_psp4(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_psp4, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = new_psp4Head(2048, nclass, norm_layer, se_loss, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        # x[1] = F.interpolate(x[1], (h, w), **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class new_psp4Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, up_kwargs):
        super(new_psp4Head, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = PyramidPooling(in_channels, inter_channels, norm_layer, up_kwargs)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))


    def forward(self, x):
        outputs = [self.conv6(self.conv5(x))]
        return tuple(outputs)

def get_new_psp4(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = new_psp4(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('new_psp4_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_new_psp4_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""new_psp4 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_new_psp4_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_new_psp4('ade20k', 'resnet50', pretrained, root=root, **kwargs)

class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool1 = AdaptiveAvgPool2d(1)
        self.pool2 = AdaptiveAvgPool2d(2)
        self.pool3 = AdaptiveAvgPool2d(3)
        self.pool4 = AdaptiveAvgPool2d(6)
        self.pool5 = AdaptiveAvgPool2d(12)


        # out_channels = int(in_channels/4)
        self.conv1 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv2 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv3 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv4 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))

        self.conv5 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.conv6 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
        self.psaa = Psaa_Module(out_channels, norm_layer)


        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        feat5 = F.upsample(self.conv4(self.pool5(x)), (h, w), **self._up_kwargs)
        feat6 = self.conv6(x)
        y1 = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), 1)
        y = torch.stack((feat1, feat2, feat3, feat4, feat5, feat6), dim=-1)
        out = self.psaa(y1, y)
        return out


class Psaa_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, out_channels, norm_layer, scales=6):
        super(Psaa_Module, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(scales * out_channels, scales, 1, bias=True))

        self.fuse_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU(True))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, cat, stack):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        n, c, h, w, s = stack.size()

        energy = self.project(cat)
        attention = torch.softmax(energy, dim=1)
        yv = stack.view(n, c, h * w, s).permute(0, 2, 1, 3)
        out = torch.matmul(yv, attention.view(n, s, h * w).permute(0, 2, 1).unsqueeze(dim=3))

        energy = torch.matmul(yv.permute(0, 1, 3, 2), out)
        attention = torch.softmax(energy, dim=2)
        out2 = torch.matmul(yv, attention)

        out = self.gamma * out2 + out
        out = out.squeeze(dim=3).permute(0, 2, 1).view(n, c, h, w)
        out = self.fuse_conv(out)
        return out