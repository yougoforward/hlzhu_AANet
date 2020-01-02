from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['new_psp5Net', 'get_new_psp5net']


class new_psp5Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(new_psp5Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = new_psp5NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
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


class new_psp5NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(new_psp5NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4

        self.aa_new_psp5 = new_psp5_Module(in_channels, inter_channels, atrous_rates, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(2*inter_channels, out_channels, 1))
        if self.se_loss:
            self.selayer = nn.Linear(inter_channels, out_channels)

    def forward(self, x):
        feat_sum, gap_feat = self.aa_new_psp5(x)
        outputs = [self.conv8(feat_sum)]
        if self.se_loss:
            outputs.append(self.selayer(torch.squeeze(gap_feat)))

        return tuple(outputs)


def new_psp5Conv(in_channels, out_channels, atrous_rate, norm_layer):
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


class new_psp5Pooling(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, up_kwargs):
        super(new_psp5Pooling, self).__init__()
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


class new_psp5_Module(nn.Module):
    def __init__(self, in_channels, out_channels, atrous_rates, norm_layer, up_kwargs):
        super(new_psp5_Module, self).__init__()
        # out_channels = in_channels // 4
        rate1, rate2, rate3 = tuple(atrous_rates)
        self.b0 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            norm_layer(out_channels),
            nn.ReLU(True))
        self.b1 = new_psp5Conv(in_channels, out_channels, rate1, norm_layer)
        self.b2 = new_psp5Conv(in_channels, out_channels, rate2, norm_layer)
        self.b3 = new_psp5Conv(in_channels, out_channels, rate3, norm_layer)

        self._up_kwargs = up_kwargs
        self.psaa_conv = nn.Sequential(nn.Conv2d(in_channels+4*out_channels, out_channels, 1, padding=0, bias=False),
                                    norm_layer(out_channels),
                                    nn.ReLU(True),
                                    nn.Conv2d(out_channels, 4, 1, padding=0, bias=True))  

        self.project = nn.Sequential(nn.Conv2d(in_channels=4*out_channels, out_channels=out_channels,
                      kernel_size=1, stride=1, padding=0, bias=False),
                      norm_layer(out_channels),
                      nn.ReLU(True))


        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(in_channels, out_channels, 1, bias=False),
                            norm_layer(out_channels),
                            nn.ReLU(True))
        self.se = nn.Sequential(
                            nn.Conv2d(out_channels, out_channels, 1, bias=True),
                            nn.Sigmoid())


        self.pam0 = PAM_Module(in_dim=out_channels, key_dim=out_channels//8,value_dim=out_channels,out_dim=out_channels,norm_layer=norm_layer)

    def forward(self, x):
        feat0 = self.b0(x)
        feat1 = self.b1(x)
        feat2 = self.b2(x)
        feat3 = self.b3(x)
        n, c, h, w = feat0.size()

        # psaa
        psaa_feat = self.psaa_conv(torch.cat([x, feat0, feat1, feat2, feat3], dim=1))
        psaa_att = torch.sigmoid(psaa_feat)
        psaa_att_list = torch.split(psaa_att, 1, dim=1)

        y2 = torch.cat((psaa_att_list[0] * feat0, psaa_att_list[1] * feat1, psaa_att_list[2] * feat2,
                        psaa_att_list[3] * feat3), 1)
        out = self.project(y2)
        
        #gp
        gp = self.gap(x)
        se = self.se(gp)
        out = torch.cat([self.pam0(out+se*out), gp.expand(n, c, h, w)], dim=1)
        return out, gp

def get_new_psp5net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                 root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = new_psp5Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model

class PyramidPooling(nn.Module):
    """
    Reference:
        Zhao, Hengshuang, et al. *"Pyramid scene parsing network."*
    """
    def __init__(self, in_channels, out_channels, norm_layer):
        super(PyramidPooling, self).__init__()
        self.pool1 = nn.AdaptiveAvgPool2d(1)
        self.pool2 = nn.AdaptiveAvgPool2d(2)
        self.pool3 = nn.AdaptiveAvgPool2d(3)
        self.pool4 = nn.AdaptiveAvgPool2d(6)

        self.out_chs = out_channels


        # out_channels = int(in_channels/4)
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1, bias=False),
                                norm_layer(out_channels),
                                nn.ReLU(True))

    def forward(self, x):
        n, c, h, w = x.size()
        feat1 = self.conv1(self.pool1(x)).view(n, self.out_chs, -1)
        feat2 = self.conv2(self.pool2(x)).view(n, self.out_chs, -1)
        feat3 = self.conv3(self.pool3(x)).view(n, self.out_chs, -1)
        feat4 = self.conv4(self.pool4(x)).view(n, self.out_chs, -1)

        y1 = torch.cat((feat1, feat2, feat3, feat4), 2) # 1+4+9+36=50
        return y1


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim, key_dim, value_dim, out_dim, norm_layer):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.spp = PyramidPooling(in_dim, in_dim, norm_layer)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=value_dim, out_channels=value_dim, kernel_size=1)
        self.gamma = nn.Sequential(nn.Conv2d(in_channels=in_dim, out_channels=1, kernel_size=1, bias=True), nn.Sigmoid())

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        proj_key = self.spp(x)
        m_batchsize, C, height, width = x.size()
        m_batchsize, C, dspp = proj_key.size()

        proj_query = self.query_conv(x).view(m_batchsize, -1, width*height).permute(0, 2, 1)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        out = torch.bmm(proj_key, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        gamma = self.gamma(x)
        out = (1-gamma)*out + gamma*self.value_conv(x)
        return out