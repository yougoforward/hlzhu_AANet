from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['ACA2Net', 'get_aca2net']


class ACA2Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(ACA2Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = ACA2NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class ACA2NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(ACA2NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                    norm_layer(512),
                                    nn.ReLU(inplace=True))
        self.sec = guided_SE_CAM_Module(inter_channels, 256, 256, norm_layer)
        self.conv5e = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU(True))

        self.conv5c2 = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 1, bias=False),
                                    norm_layer(512),
                                    nn.ReLU(inplace=True))
        self.sec2 = guided_SE_CAM_Module(inter_channels, 256, 256, norm_layer)
        self.conv5e2 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                    norm_layer(inter_channels), nn.ReLU(True))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(512, out_channels, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1),
            nn.Sigmoid())

        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):

        # sec
        feat = self.conv5c(x)
        sec_feat = self.sec(feat)
        sec_feat = self.conv5e(sec_feat)

        feat2 = self.conv5c(x)
        sec_feat2 = self.sec(feat2)
        sec_feat2 = self.conv5e(sec_feat2)

        feat_sum = sec_feat + sec_feat2

        if self.se_loss:
            gap_feat = self.gap(feat_sum)
            gamma = self.fc(gap_feat)
            outputs = [self.conv8(F.relu_(feat_sum + feat_sum * gamma))]
            outputs.append(self.selayer(torch.squeeze(gap_feat)))
        else:
            outputs = [self.conv8(feat_sum)]

        return tuple(outputs)



def get_aca2net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
              root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = ACA2Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


class guided_CAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, query_dim, out_dim):
        super(guided_CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_dim = query_dim
        self.chanel_out = out_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.query_conv_c = nn.Sequential(
            nn.Conv2d(in_channels=in_dim, out_channels=query_dim, kernel_size=1, bias=False), nn.BatchNorm2d(query_dim),
            nn.ReLU(), nn.Dropout2d(0.1))

    def forward(self, x):
        """
            inputs :
                x=[x1,x2]
                x1 : input feature maps( B X C*5 X H X W)
                x2 : input deature maps (BxCxHxW)
            returns :
                out : output feature maps( B X C X H X W)
        """

        m_batchsize, C, height, width = x.size()
        proj_c_query = self.query_conv_c(x)

        proj_c_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_c_query.view(m_batchsize, self.query_dim, -1), proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out_c = torch.bmm(attention, x.view(m_batchsize, -1, width * height))
        out_c = out_c.view(m_batchsize, -1, height, width)
        out_c = self.gamma * out_c + proj_c_query
        return out_c


class SE_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, out_dim):
        super(SE_Module, self).__init__()
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
                                nn.Conv2d(in_dim, in_dim // 16, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.ReLU(),
                                nn.Conv2d(in_dim // 16, out_dim, kernel_size=1, padding=0, dilation=1,
                                          bias=True),
                                nn.Sigmoid()
                                )

    def forward(self, x):
        out = self.se(x)
        return out


class guided_SE_CAM_Module(nn.Module):
    """ Channel attention module"""

    def __init__(self, in_dim, query_dim, out_dim, norm_layer):
        super(guided_SE_CAM_Module, self).__init__()
        self.guided_cam = guided_CAM_Module(in_dim, query_dim, out_dim)
        self.project = nn.Sequential(
            nn.Conv2d(in_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            norm_layer(out_dim), nn.ReLU(True),
            nn.Dropout2d(0.1)
        )
        self.se = SE_Module(in_dim, out_dim)

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        gcam = self.guided_cam(x)

        bottle = self.project(x)
        se_x = self.se(x)
        se_bottle = se_x * bottle + bottle
        out = torch.cat([gcam, se_bottle], dim=1)
        return out




