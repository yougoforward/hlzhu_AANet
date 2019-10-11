from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaa3Net', 'get_psaa3net']


class psaa3Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa3Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa3NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class psaa3NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaa3NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 8

        self.aa_psaa3 = psaa3_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
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


def psaa3Conv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class psaa3Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa3Pooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), **self._up_kwargs)


class psaa3_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa3_Module, self).__init__()
        # out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa3Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa3Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa3Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa3Pooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.conv6 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False))

        self.global_cont = psaa3Pooling(out_channels, out_channels, norm_layer, up_kwargs)
        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        # self.se = SE_Module(out_channels, out_channels)
        self.relu = nn.ReLU()
        # self.project2 = nn.Sequential(
        #     nn.Conv2d(out_channels, out_channels, 1, bias=False),
        #     norm_layer(out_channels),
        #     nn.ReLU(True),
        #     nn.Dropout2d(0.1, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        guide = self.conv6(x)
        y = torch.stack((feat0, feat1, feat2, feat3, feat4), 1)

        m_batchsize, C, height, width = guide.size()
        proj_query = guide.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        proj_key = y.view(m_batchsize, 5, C, -1).permute(0, 3, 2, 1).contiguous().view(-1, C, 5)
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


def get_psaa3net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa3Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model



