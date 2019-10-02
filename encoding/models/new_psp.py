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
        self.head = new_pspHead(2048, nclass, norm_layer, se_loss, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        x = list(self.head(c4))
        x[0] = upsample(x[0], (h,w), **self._up_kwargs)
        # outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = upsample(auxout, (h,w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class new_pspHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, up_kwargs):
        super(new_pspHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = PyramidPooling(in_channels, inter_channels, norm_layer, up_kwargs)
        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

        self.se_loss = se_loss
        if self.se_loss:
            # self.selayer1 = nn.Linear(inter_channels, out_channels)
            self.selayer2 = nn.Linear(inter_channels, out_channels)


    def forward(self, x):
        if self.se_loss:
            out, global_cont1, global_cont2=self.conv5(x)
            outputs = [self.conv6(out)]
            # outputs.append(self.selayer1(torch.squeeze(global_cont1))+self.selayer2(torch.squeeze(global_cont2)))
            outputs.append(self.selayer2(torch.squeeze(global_cont2)))

        else:
            outputs = [self.conv6(self.conv5(x)[0])]
        return tuple(outputs)

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

        self.project = nn.Sequential(
            nn.Conv2d(6 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False))

        self.global_cont1 = psaa2Pooling(out_channels, out_channels, norm_layer, up_kwargs)
        self.global_cont2 = psaa2Pooling(out_channels, out_channels, norm_layer, up_kwargs)
        self.softmax = nn.Softmax(dim=-1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.se = SE_Module(out_channels, out_channels)
        self.relu = nn.ReLU()
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 1),
            nn.Sigmoid())

    def forward(self, x):
        _, _, h, w = x.size()
        feat1 = F.upsample(self.conv1(self.pool1(x)), (h, w), **self._up_kwargs)
        feat2 = F.upsample(self.conv2(self.pool2(x)), (h, w), **self._up_kwargs)
        feat3 = F.upsample(self.conv3(self.pool3(x)), (h, w), **self._up_kwargs)
        feat4 = F.upsample(self.conv4(self.pool4(x)), (h, w), **self._up_kwargs)
        feat5 = F.upsample(self.conv5(self.pool5(x)), (h, w), **self._up_kwargs)
        feat6 = self.conv6(x)

        y1 = torch.cat((feat1, feat2, feat3, feat4, feat5, feat6), 1)
        y1 = self.project(y1)

        y = torch.stack((feat1, feat2, feat3, feat4, feat5, feat6), 1)

        global_cont1 = self.global_cont1(y1)
        query = global_cont1+y1
        # query = y1
        m_batchsize, C, height, width = query.size()
        proj_query = query.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        proj_key = y.view(m_batchsize, 6, C, -1).permute(0, 3, 2, 1).contiguous().view(-1, C, 6)
        energy = torch.bmm(proj_query.view(-1, 1, C), proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = proj_key.permute(0, 2, 1)

        out = torch.bmm(attention, proj_value)
        out = self.gamma * out.view(m_batchsize, height, width, C).permute(0, 3, 1, 2) + query
        global_cont2 = self.global_cont2(out)
        out = self.relu(out+self.fc(global_cont2)*out)
        # out = self.relu(out + self.se(out) * out)
        return out, global_cont1, global_cont2

class psaa2Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa2Pooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)
        return pool
        # return F.interpolate(pool, (h, w), **self._up_kwargs)
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