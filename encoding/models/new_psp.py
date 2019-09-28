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

class new_psp(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_psp, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = new_pspHead(2048, nclass, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = self.head(c4)
        x = upsample(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h,w), **self._up_kwargs)
            outputs.append(auxout)
        return tuple(outputs)


class new_pspHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(new_pspHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidPooling(in_channels, inter_channels, norm_layer, up_kwargs),
                                   nn.Conv2d(inter_channels * 6, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

def get_new_psp(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = new_psp(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('new_psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_new_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""new_psp model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_new_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_new_psp('ade20k', 'resnet50', pretrained, root=root, **kwargs)

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
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        feat5 = F.upsample(self.conv5(self.pool5(x)), (h, w), **self._up_kwargs)
        feat6 = self.conv6(x)

        return torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), 1)