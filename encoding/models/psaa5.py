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
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat_sum = self.aa_psaa5(x)
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

        self.out_chs = out_channels

    def forward(self, x):
        bs, _, h, w = x.size()
        pool = self.gap(x)

        # return F.interpolate(pool, (h, w), **self._up_kwargs)
        # return pool.repeat(1,1,h,w)
        return pool.expand(bs, self.out_chs, h, w)

class psaa5_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa5_Module, self).__init__()
        # out_channels = in_channels // 8
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa5Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa5Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa5Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa5Pooling(in_channels, out_channels, norm_layer, up_kwargs)
        self.psaa = Psaa_Module(in_channels, out_channels, norm_layer)

        self.pool = nn.MaxPool2d(kernel_size=2)
        self.f_key = nn.Sequential(
            nn.Conv1d(in_channels=out_channels, out_channels=out_channels//4,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.f_query = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels//4,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.f_value = nn.Conv1d(in_channels=out_channels, out_channels=out_channels,
                                 kernel_size=1, stride=1, padding=0, bias=True)
        self.W = nn.Sequential(
            nn.Conv2d(in_channels=2*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True)
        )
        self._up_kwargs = up_kwargs

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
        out = self.psaa(x, y1, y)

        # guided fuse scales and positions
        # pool0 = self.pool(feat0)
        # pool1 = self.pool(feat1)
        # pool2 = self.pool(feat2)
        # pool3 = self.pool(feat3)
        # pool4 = self.pool(feat4)

        _,_,hp,wp = pool0.size()
        out_pool = self.pool(out)
        # y2 = torch.stack([pool0, pool1, pool2, pool3, pool4], dim=-1).view(n,c,-1) # n, c, hws/4
        y2 = y.view(n,c,-1) # n, c, hws
        query = self.f_query(out_pool).view(n,c//4,-1).permute(0,2,1) # n, hw/4, c
        key = self.f_key(y2)
        value = self.f_value(y2).permute(0, 2, 1)
        sim_map = torch.bmm(query, key)
        sim_map = (c//4 ** -.5) * sim_map
        sim_map = torch.softmax(sim_map, dim=-1) # n, hw/4, hws

        context = torch.bmm(sim_map, value)
        context = context.permute(0, 2, 1).contiguous().view(n,c,hp,wp)
        context = F.interpolate(context, (h, w), **self._up_kwargs)
        out = self.W(torch.cat([context, out], dim=1))
        return out


class Psaa_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, in_channels, out_channels, norm_layer):
        super(Psaa_Module, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(in_channels+5*out_channels, out_channels, 1, padding=0, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_channels, 5, 1, bias=True))

    def forward(self, x, cat, stack):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        n, c, h, w, s = stack.size()

        energy = self.project(torch.cat([x, cat], dim=1))
        # attention = torch.softmax(energy, dim=1)
        attention = torch.sigmoid(energy)
        yv = stack.view(n, c, h * w, 5).permute(0, 2, 1, 3)
        out = torch.matmul(yv, attention.view(n, 5, h * w).permute(0, 2, 1).unsqueeze(dim=3)) # n ,hw, c, 1
        out = out.squeeze(3).permute(0,2,1).view(n,c,h,w)
        return out

def get_psaa5net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa5Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model