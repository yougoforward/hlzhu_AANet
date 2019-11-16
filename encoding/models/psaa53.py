from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psaa53Net', 'get_psaa53net']


class psaa53Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psaa53Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psaa53NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class psaa53NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psaa53NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aa_psaa53 = psaa53_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(3 * inter_channels, out_channels, 1))

    def forward(self, x):
        feat_sum = self.aa_psaa53(x)
        outputs = [self.conv8(feat_sum)]
        return tuple(outputs)


def psaa53Conv(in_channels, out_channels, atrous_rate, norm_layer):
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


class psaa53Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(psaa53Pooling, self).__init__()
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


class psaa53_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(psaa53_Module, self).__init__()
        # out_channels = in_channels // 4
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = psaa53Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = psaa53Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = psaa53Conv(in_channels, out_channels, rate3, norm_layer)
        self.b4 = psaa53Pooling(in_channels, out_channels, norm_layer, up_kwargs)

        self._up_kwargs = up_kwargs
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+5*out_channels, 5, 1, bias=True))
        self.project = nn.Sequential(nn.Conv2d(in_channels=5*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))

        # self.query_conv = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        # self.key_conv0 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        # self.key_conv1 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        # self.key_conv2 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        # self.key_conv3 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        # self.key_conv4 = nn.Conv2d(in_channels=out_channels, out_channels=out_channels//8, kernel_size=1, padding=0)
        # self.scale_spatial_agg = ss_Module(out_channels, norm_layer)
        self.gap = psaa53Pooling(in_channels, out_channels, norm_layer, up_kwargs)

        self.pam = PAM_Module(in_dim=out_channels, key_dim=out_channels//8,value_dim=out_channels,out_dim=out_channels,norm_layer=norm_layer)

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

        out2 = self.pam(out)
        # #scale spatial guided attention aggregation

        # query = self.query_conv(out) # n, c//4, h, w
        # key0 = self.key_conv0(feat0) # n, c//4, h, w
        # key1 = self.key_conv1(feat1)
        # key2 = self.key_conv2(feat2)
        # key3 = self.key_conv3(feat3)
        # key4 = self.key_conv4(feat4)

        # key_stack = torch.stack((key0, key1, key2, key3, key4), dim=-1)
        # out = self.scale_spatial_agg(query, out, key_stack, fea_stack)

        #gp
        gp = self.gap(x)
        out = torch.cat([out, out2, gp], dim=1)

        return out

class ss_Module(nn.Module):
    """ Position attention module"""

    # Ref from SAGAN
    def __init__(self, out_channels, norm_layer):
        super(ss_Module, self).__init__()
        self.project = nn.Sequential(nn.Conv2d(5 * out_channels, 5, 1, bias=True))

        self.fuse_conv = nn.Sequential(nn.Conv2d(2*out_channels, out_channels, 1, padding=0, bias=False),
                                       norm_layer(out_channels),
                                       nn.ReLU(True))
        self.key_conv = nn.Sequential(nn.Conv2d(out_channels, out_channels//8, 1, padding=0, bias=True))

    def forward(self, query, fea, key_stack, fea_stack):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        n, c, h, w, s = fea_stack.size()
        key1 = key_stack.view(n, -1, h * w, s).permute(0, 2, 3, 1) # n, h*w, s, c//4
        query1 = query.view(n, -1, h*w, 1).permute(0, 2, 1, 3) # n, h*w, c//4, 1

        energy = torch.matmul(key1, query1) #n, hw, s, 1
        attention1 = torch.softmax(energy, dim=2)
        out2 = torch.matmul(fea_stack.view(n, -1, h*w, s).permute(0, 2, 1, 3), attention1) # n, hw, c, 1
        out2 = out2.squeeze(dim=3).permute(0, 2, 1).view(n, -1, h, w)

        key2 = self.key_conv(out2) # n, c//4, h, w
        key2 = key2.view(n, -1, h*w).permute(0, 2, 1)
        query2 = query.view(n, -1, h*w)
        energy = torch.bmm(key2, query2) # n, hw, hw

        attention2 = torch.softmax(energy, dim=1)
        out3 = torch.bmm(out2.view(n, -1, h*w), attention2).view(n, -1, h, w)

        out = torch.cat([out3, fea], dim=1)
        out = self.fuse_conv(out)

        return out

def get_psaa53net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psaa53Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
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
        # self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=value_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)
        # self.fuse_conv = nn.Sequential(nn.Conv2d(value_dim, out_dim, 1, bias=False),
        #                                norm_layer(out_dim),
        #                                nn.ReLU(True))
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
        # proj_value = x.view(m_batchsize, -1, width*height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        # out = self.gamma*out + x
        # out = self.fuse_conv(out)
        return out