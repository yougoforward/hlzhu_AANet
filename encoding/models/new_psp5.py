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

class new_psp5(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_psp5, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = new_psp5Head(2048, nclass, norm_layer, se_loss, self._up_kwargs)
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


class new_psp5Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, up_kwargs):
        super(new_psp5Head, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = PyramidPooling(in_channels, inter_channels, norm_layer, up_kwargs)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))


    def forward(self, x):
        outputs = [self.conv6(self.conv5(x))]
        return tuple(outputs)

def get_new_psp5(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = new_psp5(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('new_psp5_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_new_psp5_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""new_psp5 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_new_psp5_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_new_psp5('ade20k', 'resnet50', pretrained, root=root, **kwargs)

class PyramidPooling(Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PyramidPooling, self).__init__()
        self.pool0 = AdaptiveAvgPool2d(1)
        self.pool1 = AdaptiveAvgPool2d(2)
        self.pool2 = AdaptiveAvgPool2d(3)
        self.pool3 = AdaptiveAvgPool2d(6)
        self.pool4 = AdaptiveAvgPool2d(12)
        self.pool5 = AdaptiveAvgPool2d(24)


        # out_channels = int(in_channels/4)
        self.conv0 = Sequential(Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                ReLU(True))
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

        # self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+7*out_channels, out_channels, 1, padding=0, bias=False),
        #                             norm_layer(out_channels),
        #                             nn.ReLU(True),
        #                             nn.Conv2d(out_channels, 7, 1, bias=True))
        self.project = nn.Sequential(nn.Conv2d(in_channels=7*out_channels, out_channels=out_channels,
                      kernel_size=3, stride=1, padding=1, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs

    def forward(self, x):
        _, _, h, w = x.size()
        feat0 = F.upsample(self.conv0(self.pool0(x)), (h, w), **self._up_kwargs)
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        feat5 = F.upsample(self.conv5(self.pool5(x)), (h, w), **self._up_kwargs)

        feat6 = self.conv6(x)
        # psaa
        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4, feat5, feat6), 1)
        # psaa_feat = self.psaa_conv(torch.cat([x,y1], dim=1))
        # psaa_att = torch.sigmoid(psaa_feat)
        # psaa_att_list = torch.split(psaa_att, 1, dim=1)
        # 
        # y2 = torch.cat((psaa_att_list[0]*feat0, psaa_att_list[1]*feat1, psaa_att_list[2]*feat2, psaa_att_list[3]*feat3, psaa_att_list[4]*feat4, psaa_att_list[5]*feat5, psaa_att_list[6]*feat6), 1)
        out = self.project(y1)
        return out


