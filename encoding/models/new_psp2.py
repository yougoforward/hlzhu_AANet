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

class new_psp2(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_psp2, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = new_psp2Head(2048, nclass, norm_layer, se_loss, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        x[1] = F.interpolate(x[1], (h, w), **self._up_kwargs)

        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class new_psp2Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, up_kwargs):
        super(new_psp2Head, self).__init__()
        inter_channels = in_channels // 4
        self.aa_psaa3 = PyramidPooling(in_channels, inter_channels, norm_layer, up_kwargs)
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels + in_channels, inter_channels, 1, padding=0, bias=False),
                                    norm_layer(inter_channels), nn.ReLU(True))

        self.guide_pred = nn.Sequential(nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        psaa3_feat, guide = self.aa_psaa3(x)
        feat_cat = torch.cat([psaa3_feat, x], dim=1)
        feat_sum = self.conv52(feat_cat)
        guide_pred = self.guide_pred(guide)
        outputs = [self.conv8(feat_sum)]
        outputs.append(guide_pred)
        return tuple(outputs)

def get_new_psp2(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = new_psp2(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('new_psp2_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_new_psp2_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""new_psp2 model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_new_psp2_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_new_psp2('ade20k', 'resnet50', pretrained, root=root, **kwargs)

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

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
    def forward(self, x):
        _, _, h, w = x.size()
        feat0 = F.upsample(self.conv0(self.pool1(x)), (h, w), **self._up_kwargs)
        feat1 = F.upsample(self.conv1(self.pool2(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool3(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool4(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool5(x)), (h, w), **self._up_kwargs)
        feat5 = self.conv5(x)


        guide = self.conv6(x)
        y = torch.stack((feat0, feat1, feat2, feat3, feat4, feat5), 1)

        m_batchsize, C, height, width = guide.size()
        proj_query = guide.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        proj_key = y.view(m_batchsize, 6, C, -1).permute(0, 3, 2, 1).contiguous().view(-1, C, 6)
        energy = torch.bmm(proj_query.view(-1, 1, C), proj_key)
        attention = self.softmax(energy)
        proj_value = proj_key.permute(0, 2, 1)

        out = torch.bmm(attention, proj_value)
        out = self.gamma * out.view(m_batchsize, height, width, C).permute(0, 3, 1, 2) + guide
        return out, guide


class SE_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super(SE_Module, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(in_dim, in_dim // 8, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_dim // 8, out_dim, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        out = self.se(x)
        return out