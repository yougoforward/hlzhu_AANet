from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['asp_pgfNet', 'get_asp_pgfnet']


class asp_pgfNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(asp_pgfNet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = asp_pgfNetHead(2048, nclass, se_loss, norm_layer, self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)

        outputs = []
        # x = self.head(c4)
        x = list(self.head(c4))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        # x = F.interpolate(x, (h,w), **self._up_kwargs)
        outputs.append(x)
        if self.aux:
            auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            outputs.append(auxout)

        return tuple(outputs)


class asp_pgfNetHead(nn.Module):
    def __init__(self, in_channels, out_channels, se_loss, norm_layer, up_kwargs, atrous_rates=(12, 24, 36)):
        super(asp_pgfNetHead, self).__init__()
        inter_channels = in_channels // 8

        self.aspp = asp_pgf_Module(in_channels, atrous_rates, se_loss, norm_layer, up_kwargs)

        self.block = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
            norm_layer(inter_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False),
            nn.Conv2d(inter_channels, out_channels, 1))
        self.se_loss = se_loss

        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):
        x = self.aspp(x)
        outputs = [self.block(x[0])]

        if self.se_loss:
            outputs.append(self.selayer(torch.squeeze(x[1])))

        return tuple(outputs)


class GuidedFusion(nn.Module):
    """
    exploit self-attentin for  adjacent scale fusion
    """

    def __init__(self, in_channels, query_dim):
        super(GuidedFusion, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_channels, out_channels=query_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_channels, out_channels=query_dim, kernel_size=1)

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, low_level, high_level):
        m_batchsize, C, hl, wl = low_level.size()
        m_batchsize, C, hh, wh = high_level.size()

        # query = low_level.view(m_batchsize, C, hl * wl).permute(0, 2, 1)  # m, hl*wl, c
        # key = high_level.view(m_batchsize, C, hh * wh)  # m, c, hh*wh

        query = self.query_conv(low_level).view(m_batchsize, -1, hl * wl).permute(0, 2, 1)  # m, hl*wl, c
        key = self.key_conv(high_level).view(m_batchsize, -1, hh * wh)  # m, c, hh*wh
        energy = torch.bmm(query, key)  # C, hl*wl,hh*wh
        attention = self.softmax(energy)
        value = high_level
        out = torch.bmm(value.view(m_batchsize, C, hh * wh), attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, hl, wl)

        out = self.gamma * out + low_level
        return out


class PyramidGuidedFusion(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, se_loss, norm_layer):
        super(PyramidGuidedFusion, self).__init__()

        self.pool2 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.pool3 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))
        self.pool4 = nn.Sequential(nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=2, padding=1, bias=False),
                                norm_layer(in_channels),
                                nn.ReLU(True))

        self.gf2 = GuidedFusion(in_channels, in_channels//2)
        self.gf3 = GuidedFusion(in_channels, in_channels//2)
        self.gf4 = GuidedFusion(in_channels, in_channels//2)

        self.se_loss = se_loss
        if self.se_loss:
            self.gamma = nn.Parameter(torch.zeros(1))
            self.gap = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Sequential(
                nn.Conv2d(in_channels, in_channels, 1),
                nn.Sigmoid())


    def forward(self, x):
        _, _, h, w = x.size()
        d1 = x
        d2=self.pool2(d1)
        d3=self.pool3(d2)
        d4=self.pool4(d3)

        if self.se_loss:
            gap_feat = self.gap(d4)
            gamma = self.fc(gap_feat)
            d4 = F.relu(d4 + d4 * gamma)

        u3 = self.gf4(d3, d4)
        u2 = self.gf3(d2, u3)
        u1 = self.gf2(d1, u2)
        outputs= [u1]

        if self.se_loss:
            outputs.append(gap_feat)
        return outputs



def ASPPConv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class AsppPooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(AsppPooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), **self._up_kwargs)


class asp_pgf_Module(nn.Module):
    def __init__(self, in_channels, atrous_rates, se_loss, norm_layer, up_kwargs):
        super(asp_pgf_Module, self).__init__()
        out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.pgf = PyramidGuidedFusion(out_channels, se_loss, norm_layer)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = ASPPConv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = ASPPConv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = ASPPConv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = AsppPooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.project = nn.Sequential(
            nn.Conv2d(4 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.5, False))

        self.se_loss = se_loss

    def forward(self, x):
        feat0 = self.pgf(self.b0(x))
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        # feat4 = self.b4(x)

        # y = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        y = torch.cat((feat0[0], feat1, feat2, feat3), 1)
        outputs = [self.project(y)]
        if self.se_loss:
            outputs.append(feat0[1])

        return tuple(outputs)


def get_asp_pgfnet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = asp_pgfNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model
