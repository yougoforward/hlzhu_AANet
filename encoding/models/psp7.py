###########################################################################
# Created by: Hang Zhang
# Email: zhang.hang@rutgers.edu
# Copyright (c) 2017
###########################################################################
from __future__ import division

import torch
import torch.nn as nn

from torch.nn.functional import upsample

from .base import BaseNet
from .fcn import FCNHead

class PSP7(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(PSP7, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        self.head = PSP7Head(2048, nclass, norm_layer, self._up_kwargs)
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


class PSP7Head(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(PSP7Head, self).__init__()
        inter_channels = in_channels // 4
        self.conv5 = nn.Sequential(PyramidContext(in_channels, norm_layer),
                                   nn.Conv2d(in_channels+inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU(True),
                                   nn.Dropout2d(0.1, False),
                                   nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        return self.conv5(x)

def get_PSP7(dataset='pascal_voc', backbone='resnet50', pretrained=False,
            root='~/.encoding/models', **kwargs):
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'ade20k': 'ade',
    }
    # infer number of classes
    from ..datasets import datasets
    model = PSP7(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('psp_%s_%s'%(backbone, acronyms[dataset]), root=root)))
    return model

def get_psp_resnet50_ade(pretrained=False, root='~/.encoding/models', **kwargs):
    r"""PSP model from the paper `"Context Encoding for Semantic Segmentation"
    <https://arxiv.org/pdf/1803.08904.pdf>`_

    Parameters
    ----------
    pretrained : bool, default False
        Whether to load the pretrained weights for model.
    root : str, default '~/.encoding/models'
        Location for keeping the model parameters.


    Examples
    --------
    >>> model = get_psp_resnet50_ade(pretrained=True)
    >>> print(model)
    """
    return get_PSP7('ade20k', 'resnet50', pretrained, root=root, **kwargs)

class PyramidContext(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, norm_layer, up_kwargs):
        super(PyramidContext, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)
        self.cont_dim =50 

        self.out_channels = int(in_channels/4)
        self.conv0 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                   norm_layer(self.out_channels),
                                   nn.ReLU(True))

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                   norm_layer(self.out_channels),
                                   nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                   norm_layer(self.out_channels),
                                   nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                   norm_layer(self.out_channels),
                                   nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, self.out_channels, 1, bias=False),
                                   norm_layer(self.out_channels),
                                   nn.ReLU(True))

        # self.conv_aff = nn.Sequential(nn.Conv2d(self.out_channels, self.cont_dim, 1, bias=True),
        #                            nn.Softmax(dim=1))
        # bilinear upsample options
        self._up_kwargs = up_kwargs
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        bs, c, h, w = x.size()
        feat1 = self.conv1(self.pool1(x))
        feat2 = self.conv2(self.pool2(x))
        feat3 = self.conv3(self.pool3(x))
        feat4 = self.conv4(self.pool4(x))


        local_context = self.conv0(x)
        local_global_context = self.pool1(local_context)+local_context
        # aff = self.conv_aff(local_global_context).view(bs,-1,h*w)
        py_context = torch.cat([feat1.view(bs,self.out_channels,-1),feat1.view(bs,self.out_channels,-1),\
        feat1.view(bs,self.out_channels,-1),feat1.view(bs,self.out_channels,-1)], dim=2)
        # out = torch.bmm(py_context,aff).view(bs,self.out_channels,h,w)+local_context

        ch_energy = torch.bmm(py_context,py_context.permute(0,2,1))
        ch_energy_new = torch.max(ch_energy, -1, keepdim=True)[0].expand_as(ch_energy) - ch_energy
        ch_attention = self.softmax(ch_energy_new)
        py_context = torch.bmm(ch_attention, py_context)

        query = local_global_context.view(bs, self.out_channels,-1).permute(0,2,1)
        energy = torch.bmm(query,py_context)
        ps_attention = self.softmax(energy)
        ps_agg = torch.bmm(py_context,ps_attention.permute(0,2,1)).view(bs,self.out_channels,h,w)

        

        out = ch_agg + local_context
        return torch.cat((x, out), 1)