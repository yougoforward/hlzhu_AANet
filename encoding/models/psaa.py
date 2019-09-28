from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaaNet', 'get_psaanet']


class psaaNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaaNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaaNetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class psaaNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaaNetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aa_psaa = psaa_Module(in_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=0, bias=False),
                                    norm_layer(inter_channels), nn.ReLU(True))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, out_channels, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1),
            nn.Sigmoid())

        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):

        psaa_feat = self.aa_psaa(x)
        psaa_conv = self.conv52(psaa_feat)
        feat_sum = psaa_conv

        if self.se_loss:
            gap_feat = self.gap(feat_sum)
            gamma = self.fc(gap_feat)
            outputs = [self.conv8(F.relu_(feat_sum + feat_sum * gamma))]
            outputs.append(self.selayer(torch.squeeze(gap_feat)))
        else:
            outputs = [self.conv8(feat_sum)]

        return tuple(outputs)


def psaaConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class psaaPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaaPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), **self._up_kwargs)


class psaa_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaaConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaaConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaaConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaaPooling(in_channels, out_channels, norm_layer, up_kwargs)
        
        
        self.global_cont = psaaPooling(out_channels, out_channels, norm_layer, up_kwargs)
        self.softmax = nn.Softmax(dim=-1)

        self.project = nn.Sequential(
            nn.Conv2d(out_channels, 512, 1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        # y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        y = torch.stack((feat0, feat1, feat2, feat3, feat4), 1)
        
        query = self.global_cont(feat0)+feat0
        m_batchsize, C, height, width = query.size()
        proj_query = query.view(m_batchsize, C, -1).permute(0,2,1).contiguous()
        proj_key = y.view(m_batchsize, 5, C, -1).permute(0, 3, 2, 1).contiguous().view(-1,C,5)
        energy = torch.bmm(proj_query.view(-1,1, C), proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = proj_key.permute(0,2,1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, height, width, C).permute(0,3,1,2)+query
        
        return self.project(out)


def get_psaanet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaaNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


