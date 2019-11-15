from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaa34Net', 'get_psaa34net']


class psaa34Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa34Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa34NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class psaa34NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaa34NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aa_psaa34 = psaa34_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x):
        feat_sum = self.aa_psaa34(x)
        outputs = [self.conv8(feat_sum)]
        return tuple(outputs)


def psaa34Conv(in_channels, out_channels, atrous_rate, norm_layer):
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


class psaa34Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa34Pooling, self).__init__()
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


class psaa34_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa34_Module, self).__init__()
        # out_channels = in_channels // 4
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa34Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa34Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa34Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa34Pooling(in_channels, out_channels, norm_layer, up_kwargs)

        self._up_kwargs = up_kwargs
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+5*out_channels, 5, 1, bias=True))
        self.project = nn.Sequential(nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))

        self.query_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        self.key_conv0 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        self.key_conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        self.key_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        self.key_conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        self.key_conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        self.fuse_conv = nn.Sequential(nn.Conv2d(2 * out_channels, out_channels, 1, padding=0, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU(True))
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        feat4 = self.b4(x)
        n, c, h, w = feat0.size()

        # psaa
        y1 = torch.cat((feat0, feat1, feat2, feat3, feat4), 1)
        fea_stack = torch.stack((feat0, feat1, feat2, feat3, feat4), dim=-1)
        psaa_feat = self.psaa_conv(torch.cat([x, y1], dim=1))
        psaa_att = torch.sigmoid(psaa_feat)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
                        psaa_att_list[3] * feat3, psaa_att_list[4] * feat4), 1)
        out = self.project(y2)

        #scale spatial guided attention aggregation

        query = self.query_conv(self.pool(out)) # n, c//8, hp, wp
        feat0_p = self.pool(feat0)
        feat1_p = self.pool(feat1)
        feat2_p = self.pool(feat2)
        feat3_p = self.pool(feat3)
        feat4_p = self.pool(feat4)
        fea_p_stack = torch.stack((feat0_p, feat1_p, feat2_p, feat3_p, feat4_p), dim=-1)
        key0 = self.key_conv0(feat0_p) # n, c//8, hp, wp
        key1 = self.key_conv1(feat1_p)
        key2 = self.key_conv2(feat2_p)
        key3 = self.key_conv3(feat3_p)
        key4 = self.key_conv4(feat4_p)

        key_stack = torch.stack((key0, key1, key2, key3, key4), dim=-1) #n, c//8, hp, wp, s
        out = self.scale_spatial_agg(query, out, key_stack, fea_stack)

        n, c_key, hp, wp, s = key_stack.size()
        energy = torch.bmm(query.view(n, c_key, -1).permute(0, 2, 1),key_stack.view(n, c_key, -1)) # n, hw/4, hws/4
        attention = torch.softmax(energy, -1)
        ps_agg = torch.bmm(fea_p_stack.view(n, c, -1), attention.permute(0, 2, 1))
        ps_agg = F.interpolate(ps_agg, (h, w), mode="bilinear", align_corners=True)
        out = self.fuse_conv(torch.cat([out, ps_agg], dim=1))
        return out

def get_psaa34net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa34Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model