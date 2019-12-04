from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from .mask_softmax import Mask_Softmax

from .fcn import FCNHead
from .base import BaseNet

__all__ = ['psp4Net', 'get_psp4net']


class psp4Net(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(psp4Net, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)

        self.head = psp4NetHead(2048, nclass, norm_layer, se_loss, jpu=kwargs['jpu'], up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer)

    def forward(self, x):
        _, _, h, w = x.size()
        _, _, c3, c4 = self.base_forward(x)
        if self.aux:
            auxout = self.auxlayer(c3)
        x = list(self.head(c4, auxout))
        x[0] = F.interpolate(x[0], (h, w), **self._up_kwargs)
        if self.aux:
            # auxout = self.auxlayer(c3)
            auxout = F.interpolate(auxout, (h, w), **self._up_kwargs)
            x.append(auxout)
        return tuple(x)


class psp4NetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer, se_loss, jpu=False, up_kwargs=None,
                 atrous_rates=(12, 24, 36)):
        super(psp4NetHead, self).__init__()
        self.se_loss = se_loss
        inter_channels = in_channels // 4
        self.project = nn.Sequential(nn.Conv2d(in_channels, out_channels, 3, padding=1,
                  dilation=1, bias=False), norm_layer(out_channels), nn.ReLU(True))

        self.ocr = OCR_Module(inter_channels, inter_channels, out_channels, norm_layer, up_kwargs)
        self.conv8 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(inter_channels, out_channels, 1))

    def forward(self, x, coarse_seg):
        x = self.project(x)
        ocr_feat = self.ocr(x, coarse_seg)
        outputs = [self.conv8(ocr_feat)]
        return tuple(outputs)


class OCR_Module(nn.Module):
    def __init__(self, in_channels, out_channels, nclass, norm_layer, up_kwargs):
        super(OCR_Module, self).__init__()
        self.classes = nclass
        self. key_dim = 256
        self.query_conv = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=self.key_dim, kernel_size=1),
                                        norm_layer(self.key_dim), nn.ReLU(True))
        self.key_conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=self.key_dim, kernel_size=1),
                                        norm_layer(self.key_dim), nn.ReLU(True))

        self.value_conv = nn.Sequential(nn.Conv1d(in_channels=in_channels, out_channels=self.key_dim, kernel_size=1),
                                        norm_layer(self.key_dim), nn.ReLU(True))
        self.weights = nn.Sequential(nn.Conv1d(in_channels=self.key_dim, out_channels=out_channels, kernel_size=1),
                                        norm_layer(out_channels), nn.ReLU(True))

        self.project = nn.Sequential(nn.Conv1d(in_channels=2*out_channels, out_channels=out_channels, kernel_size=1),
                                        norm_layer(out_channels), nn.ReLU(True))

    def forward(self, x, coarse_seg):
        # object region representation or object center feature
        n,c,h,w = x.size()
        cls_att_sum = torch.sum(coarse_seg, dim=(2,3), keepdim=False) # nxN
        cls_center = torch.bmm(coarse_seg.view(n, self.classes, -1), x.view(n, c, -1).permute(0,2,1))
        norm_cls_center = cls_center/cls_att_sum.unsqueeze(2)
        norm_cls_center = norm_cls_center.permute(0, 2, 1)
        # self-attention based pixel-region relation

        query = self.query_conv(x) # n, c, h, w
        key = self.key_conv(norm_cls_center) # n, c, N
        value = self.value_conv(norm_cls_center) #n, c, N

        sim_map = torch.bmm(query.view(n, -1, h*w).permute(0, 2, 1), key)
        sim_map = (self.key_dim ** -.5) * sim_map
        sim_map = torch.softmax(sim_map, dim=-1) # n, hw, N

        value = torch.bmm(value, sim_map.permute(0,2,1)).view(n, self.key_dim, h, w)
        ocr_feat = self.weights(value)

        ocr_aug = self.project(torch.cat([ocr_feat, x], dim=1))

        return ocr_aug

def get_psp4net(dataset='pascal_voc', backbone='resnet50', pretrained=False,
                    root='~/.encoding/models', **kwargs):
    # infer number of classes
    from ..datasets import datasets
    model = psp4Net(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)
    if pretrained:
        raise NotImplementedError

    return model