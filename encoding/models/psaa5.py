from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaa5Net', 'get_psaa5net']


class psaa5Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa5Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa5NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class psaa5NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaa5NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 8

        self.aa_psaa5 = psaa5_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        # self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=0, bias=False),
        #                             norm_layer(inter_channels), nn.ReLU(True))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(inter_channels, inter_channels, 1),
            nn.Sigmoid())

        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):

        psaa5_feat = self.aa_psaa5(x)
        # psaa5_conv = self.conv52(psaa5_feat)
        feat_sum = psaa5_feat

        if self.se_loss:
            gap_feat = self.gap(feat_sum)
            gamma = self.fc(gap_feat)
            outputs = [self.conv8(F.relu_(feat_sum + feat_sum * gamma))]
            outputs.append(self.selayer(torch.squeeze(gap_feat)))
        else:
            outputs = [self.conv8(feat_sum)]

        return tuple(outputs)


def psaa5Conv(in_channels, out_channels, atrous_rate, norm_layer):
    block = nn.Sequential(
        nn.Conv2d(in_channels, 512, 1, padding=0,
                  dilation=1, bias=False),
        norm_layer(512),
        nn.ReLU(True),
        nn.Conv2d(512, out_channels, 3, padding=atrous_rate,
                  dilation=atrous_rate, bias=False),
        norm_layer(out_channels),
        nn.ReLU(True))
    return block


class psaa5Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa5Pooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        return F.interpolate(pool, (h, w), **self._up_kwargs)


class psaa5_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa5_Module, self).__init__()
        # out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Conv2d(512, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa5Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa5Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa5Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa5Pooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Sequential(
            nn.Conv2d(5 * out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True),
            nn.Dropout2d(0.1, False))

        self.softmax = nn.Softmax(dim=-1)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.se = SE_Module(out_channels, out_channels)
        # self.relu = nn.ReLU()
        self.pam = PAM_Module(out_channels, out_channels//4, out_channels)
        self.cam = CAM_Module(out_channels)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)

        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        query = self.project(y1)

        y = torch.stack((feat0, feat1, feat2, feat3, feat4), 1)

        m_batchsize, C, height, width = query.size()
        proj_query = query.view(m_batchsize, C, -1).permute(0, 2, 1).contiguous()
        proj_key = y.view(m_batchsize, 5, C, -1).permute(0, 3, 2, 1).contiguous().view(-1, C, 5)
        energy = torch.bmm(proj_query.view(-1, 1, C), proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)
        proj_value = proj_key.permute(0, 2, 1)

        out = torch.bmm(attention, proj_value)
        # out = self.gamma * out.view(m_batchsize, height, width, C).permute(0, 3, 1, 2) + self.se(query) * query
        out = self.gamma*out.view(m_batchsize, height, width, C).permute(0,3,1,2)+query
        # out = self.relu(out+self.se(out)*out)
        out = self.pam(out)
        out = self.cam(out)
        return out


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


def get_psaa5net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa5Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model



class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, out_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=out_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim


        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax  = nn.Softmax(dim=-1)
    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width = x.size()
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
        return out