from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaa8Net', 'get_psaa8net']


class psaa8Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa8Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa8NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class psaa8NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaa8NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 8

        self.aa_psaa8 = psaa8_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        psaa8_feat = self.aa_psaa8(x)
        feat_sum = psaa8_feat
        outputs = [self.conv8(feat_sum)]
        return tuple(outputs)


def psaa8Conv(in_channels, out_channels, atrous_rate, norm_layer):
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


class psaa8Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa8Pooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

        self.out_chs = out_channels

    def forward(self, x):
        bs, _, h, w = x.size()
        pool = self.gap(x)

        # return F.interpolate(pool, (h, w), **self._up_kwargs)
        # return pool.repeat(1,1,h,w)
        return pool.expand(bs, self.out_chs, h, w)

class psaa8_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa8_Module, self).__init__()
        # out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa8Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa8Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa8Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa8Pooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.gap = psaa8Pooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.se = SE_Module(out_channels, out_channels)
        self.reduce_conv = nn.Sequential(
            nn.Conv2d(2 * out_channels, out_channels, 1, padding=0, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.guided_cam_fuse = guided_CAM_Module(5*out_channels, out_channels, out_channels, norm_layer)
        self.psaa = Psaa_Module(out_channels, norm_layer)
        self.project = nn.Sequential(
            nn.Conv2d(5*out_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        n, c, h, w = feat0.size()

        # psaa
        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        y = torch.stack((feat0, feat1, feat2, feat3, feat4), dim=-1)
        out = self.psaa(y1, y)
        # guided fuse channel
        query = self.project(y1)
        out2 = self.guided_cam_fuse(y1, query)
        #gp
        # gap = self.gap(x)
        out = self.reduce_conv(torch.cat([out, out2], dim=1))
        # out = self.reduce_conv(torch.cat([gap, out, out2], dim=1))

        # se
        # out = out + self.se(out) * out
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


def get_psaa8net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa8Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model


class PAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=key_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=value_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        self.fuse_conv = nn.Sequential(nn.Conv2d(value_dim, out_dim, 1, bias=False),
                                       norm_layer(out_dim),
                                       nn.ReLU(True))

    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma * out + x
        out = self.fuse_conv(out)
        return out


class guided_CAM_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_dim, query_dim, out_dim, norm_layer):
        super(guided_CAM_Module, self).__init__()
        self.chanel_in = in_dim
        self.query_dim = query_dim
        self.chanel_out = out_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)
        self.fuse_conv = nn.Sequential(nn.Conv2d(query_dim, out_dim, 1, padding=0, bias=False),
                                       norm_layer(out_dim),
                                       nn.ReLU(True))

    def forward(self, x, query):
        """
            inputs :
                x=[x1,x2]
                x1 : input feature maps( B X C*5 X H X W)
                x2 : input deature maps (BxCxHxW)
            returns :
                out : output feature maps( B X C X H X W)
        """

        m_batchsize, C, height, width = x.size()
        proj_c_query = query

        proj_c_key = x.view(m_batchsize, C, -1).permute(0, 2, 1)
        energy = torch.bmm(proj_c_query.view(m_batchsize, self.query_dim, -1), proj_c_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        out_c = torch.bmm(attention, x.view(m_batchsize, -1, width * height))
        out_c = out_c.view(m_batchsize, -1, height, width)
        out_c = self.gamma * out_c + proj_c_query
        out_c = self.fuse_conv(out_c)
        return out_c

# class Psaa_Module(nn.Module):
#     """ Position attention module"""
#
#     # Ref from SAGAN
#     def __init__(self, out_channels, norm_layer):
#         super(Psaa_Module, self).__init__()
#         self.project = nn.Sequential(nn.Conv2d(5 * out_channels, 5, 1, bias=True))
#
#         self.fuse_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=False),
#                                        norm_layer(out_channels),
#                                        nn.ReLU(True))
#         self.gamma = nn.Parameter(torch.zeros(1))
#         self.beta = nn.Parameter(torch.zeros(1))
#
#
#         self.fuse_project = nn.Sequential(
#             nn.Conv2d(5 * out_channels, out_channels, 1, padding=0, bias=False),
#             norm_layer(out_channels),
#             nn.ReLU(True))
#
#     def forward(self, cat, stack):
#         """
#             inputs :
#                 x : input feature maps( B X C X H X W)
#             returns :
#                 out : attention value + input feature
#                 attention: B X (HxW) X (HxW)
#         """
#         n, c, h, w, s = stack.size()
#
#         # learned psaa
#         energy = self.project(cat)
#         attention = torch.softmax(energy, dim=1)
#         yv = stack.view(n, c, h * w, 5).permute(0, 2, 1, 3) # n ,hw, c, 5
#         out = torch.matmul(yv, attention.view(n, 5, h * w).permute(0, 2, 1).unsqueeze(dim=3)) # n, hw, c, 1
#         # out = out.squeeze(dim=3).permute(0, 2, 1).view(n, c, h, w)
#
#         # guided psaa
#         query = self.fuse_project(cat)
#         energy = torch.matmul(yv.permute(0, 1, 3, 2), query.view(n, -1, h*w).permute(0, 2, 1).unsqueeze(dim=3)) # n, hw, 5, 1
#         attention = torch.softmax(energy, dim=2)
#         out2 = torch.matmul(yv, attention)# n, hw, c, 1
#         # out2 = out2.squeeze(dim=3).permute(0, 2, 1).view(n, c, h, w)
#
#         out = self.beta * out2 + self.gamma*out
#         out = out.squeeze(dim=3).permute(0, 2, 1).view(n, c, h, w) + query
#         out = self.fuse_conv(out)
#         return out


class Psaa_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, out_channels, norm_layer):
        super(Psaa_Module, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, 5, 1, bias=True))

        self.fuse_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels, 1, padding=0, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU(True))
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, cat, stack):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        n, c, h, w, s = stack.size()

        energy = self.project(cat)
        attention = torch.softmax(energy, dim=1)
        yv = stack.view(n, c, h * w, 5).permute(0, 2, 1, 3)
        out = torch.matmul(yv, attention.view(n, 5, h * w).permute(0, 2, 1).unsqueeze(dim=3))

        energy = torch.matmul(yv.permute(0, 1, 3, 2), out)
        attention = torch.softmax(energy, dim=2)
        out2 = torch.matmul(yv, attention)

        out = self.gamma * out2 + out
        out = out.squeeze(dim=3).permute(0, 2, 1).view(n, c, h, w)
        out = self.fuse_conv(out)

        return out