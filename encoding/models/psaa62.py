from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaa62Net', 'get_psaa62net']


class psaa62Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa62Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa62NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class psaa62NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaa62NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 8
        self.cls = out_channels
        self.inter = inter_channels

        self.aa_psaa62 = psaa62_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)

        self.conv7 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))


        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(3*inter_channels, out_channels, 1))

        self.reduce_conv = nn.Sequential(nn.Conv2d(2 * inter_channels, inter_channels, 1, padding=0, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))
        self.fuse_conv = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 1, padding=0, bias=False),
                                         norm_layer(inter_channels),
                                         nn.ReLU(True))

    def forward(self, x):
        n ,c, h, w = x.size()
        psaa62_feat = self.aa_psaa62(x)
        coarse = self.conv7(psaa62_feat)
        psaa62_feat_reduce = self.reduce_conv(psaa62_feat)
        p_att_list = torch.split(torch.softmax(coarse, dim=1), 1, dim=1)
        center_list = [torch.sum(psaa62_feat_reduce * p_att_list[i], (2, 3), keepdim=False) / torch.sum(p_att_list[i], (2, 3), keepdim=False) for i in range(self.cls)]
        center_fea = torch.stack(center_list, dim=-1)  # n x in_dim x cls_p
        center_energy = torch.bmm(psaa62_feat_reduce.view(n , self.inter, h*w).permute(0,2,1), center_fea)
        center_att = torch.softmax(center_energy, dim=-1)
        center_agg = torch.bmm(center_fea, center_att.permute(0,2,1)).view(n, self.inter, h, w)


        fine = self.conv8(torch.cat([self.fuse_conv(center_agg), psaa62_feat], dim=1))
        outputs = [fine, coarse]
        return tuple(outputs)


def psaa62Conv(in_channels, out_channels, atrous_rate, norm_layer):
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


class psaa62Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa62Pooling, self).__init__()
        self._up_kwargs = up_kwargs
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                                 nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                 norm_layer(out_channels),
                                 nn.ReLU(True))

    def forward(self, x):
        _, _, h, w = x.size()
        pool = self.gap(x)

        # return F.interpolate(pool, (h, w), **self._up_kwargs)
        return pool.repeat(1,1,h,w)

class psaa62_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa62_Module, self).__init__()
        # out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, 512, 1, bias=False),
            norm_layer(512),
            nn.ReLU(True),
            nn.Conv2d(512, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa62Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa62Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa62Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa62Pooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, 5, 1, bias=True))

        self.softmax = nn.Softmax(dim=1)
        self.se = SE_Module(out_channels, out_channels)
        self.pam = PAM_Module(out_channels, out_channels//4, out_channels, out_channels, norm_layer)

        self.skip_conv = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, padding=0, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU(True))
        self.reduce_conv = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, 1, padding=0, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU(True))
        self.guided_cam = guided_CAM_Module(out_channels, out_channels, out_channels, norm_layer)
        self.gap = psaa62Pooling(in_channels, out_channels, norm_layer, up_kwargs)



    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        n, c, h, w = feat0.size()

        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        energy = self.project(y1)
        attention = self.softmax(energy)
        y = torch.stack((feat0, feat1, feat2, feat3, feat4), dim=-1)
        out = torch.matmul(y.view(n, c, h*w, 5).permute(0,2,1,3), attention.view(n, 5, h*w).permute(0,2,1).unsqueeze(dim=3))
        out = out.squeeze(dim=3).permute(0,2,1).view(n,c,h,w)
        out = self.pam(out)


        # out = self.guided_cam(self.skip_conv(x), out)
        # out = self.reduce_conv(torch.cat([x, out], dim=1))

        # gcam
        gap =self.gap(x)
        # out = self.guided_cam(self.skip_conv(x), out)
        # out = self.reduce_conv(torch.cat([gap, out], dim=1))
        # out = out+self.se(out)*out

        out = torch.cat([gap, out], dim=1)

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


def get_psaa62net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa62Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model



class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
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
        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width*height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = self.gamma*out + x
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