# -*- coding: utf-8 -*-
# @File             : mt_ocr.py
# @Author           : zhaoHL
# @Contact          : huilin16@qq.com
# @Time Create First: 2023/4/4 14:21
# @Contributor      : zhaoHL
# @Time Modify Last : 2023/4/4 14:21
'''
@File Description:

'''
from __future__ import print_function
from __future__ import division



import os

import numpy as np

import torch
import torch.nn as nn
import torch._utils
import torch.nn.functional as F
from timm import create_model


ALIGN_CORNERS = True
BN_MOMENTUM = 0.1


class ModuleHelper:

    @staticmethod
    def BNReLU(num_features, bn_type=None, **kwargs):
        return nn.Sequential(
            nn.BatchNorm2d(num_features, **kwargs),
            nn.ReLU()
        )

    @staticmethod
    def BatchNorm2d(*args, **kwargs):
        return nn.BatchNorm2d


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class SpatialGather_Module(nn.Module):
    """
        Aggregate the context features according to the initial
        predicted probability distribution.
        Employ the soft-weighted method to aggregate the context.
    """
    def __init__(self, cls_num=0, scale=1):
        super(SpatialGather_Module, self).__init__()
        self.cls_num = cls_num
        self.scale = scale

    def forward(self, feats, probs):
        batch_size, c, h, w = probs.shape #probs.size(0), probs.size(1), probs.size(2), probs.size(3)
        probs = probs.view(batch_size, c, -1)
        feats = feats.view(batch_size, feats.shape[1], -1)
        feats = feats.permute(0, 2, 1) # batch * hw * c
        probs = F.softmax(self.scale * probs, dim=2)# batch * k * hw
        ocr_context = torch.matmul(probs, feats).permute(0, 2, 1).unsqueeze(3)# batch * c * k * 1
        return ocr_context


class _ObjectAttentionBlock(nn.Module):
    '''
    The basic implementation for object context block
    Input:
        N X C X H X W
    Parameters:
        in_channels       : the dimension of the input feature map
        key_channels      : the dimension after the key/query transform
        scale             : choose the scale to downsample the input feature maps (save memory cost)
        bn_type           : specify the bn type
    Return:
        N X C X H X W
    '''
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(_ObjectAttentionBlock, self).__init__()
        self.scale = scale
        self.in_channels = in_channels
        self.key_channels = key_channels
        self.pool = nn.MaxPool2d(kernel_size=(scale, scale))
        self.f_pixel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_object = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_down = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels, out_channels=self.key_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.key_channels, bn_type=bn_type),
        )
        self.f_up = nn.Sequential(
            nn.Conv2d(in_channels=self.key_channels, out_channels=self.in_channels,
                kernel_size=1, stride=1, padding=0, bias=False),
            ModuleHelper.BNReLU(self.in_channels, bn_type=bn_type),
        )

    def forward(self, x, proxy):
        batch_size, h, w = x.size(0), x.size(2), x.size(3)
        if self.scale > 1:
            x = self.pool(x)

        query = self.f_pixel(x).view(batch_size, self.key_channels, -1)
        query = query.permute(0, 2, 1)
        key = self.f_object(proxy).view(batch_size, self.key_channels, -1)
        value = self.f_down(proxy).view(batch_size, self.key_channels, -1)
        value = value.permute(0, 2, 1)

        sim_map = torch.matmul(query, key)
        sim_map = (self.key_channels**-.5) * sim_map
        sim_map = F.softmax(sim_map, dim=-1)

        # add bg context ...
        context = torch.matmul(sim_map, value)
        context = context.permute(0, 2, 1).contiguous()
        context = context.view(batch_size, self.key_channels, *x.size()[2:])
        context = self.f_up(context)
        if self.scale > 1:
            context = F.interpolate(input=context, size=(h, w), mode='bilinear', align_corners=ALIGN_CORNERS)

        return context


class ObjectAttentionBlock2D(_ObjectAttentionBlock):
    def __init__(self,
                 in_channels,
                 key_channels,
                 scale=1,
                 bn_type=None):
        super(ObjectAttentionBlock2D, self).__init__(in_channels,
                                                     key_channels,
                                                     scale,
                                                     bn_type=bn_type)


class SpatialOCR_Module(nn.Module):
    """
    Implementation of the OCR module:
    We aggregate the global object representation to update the representation for each pixel.
    """
    def __init__(self,
                 in_channels,
                 key_channels,
                 out_channels,
                 scale=1,
                 dropout=0.1,
                 bn_type=None):
        super(SpatialOCR_Module, self).__init__()
        self.object_context_block = ObjectAttentionBlock2D(in_channels,
                                                           key_channels,
                                                           scale,
                                                           bn_type)
        _in_channels = 2 * in_channels

        self.conv_bn_dropout = nn.Sequential(
            nn.Conv2d(_in_channels, out_channels, kernel_size=1, padding=0, bias=False),
            ModuleHelper.BNReLU(out_channels, bn_type=bn_type),
            nn.Dropout2d(dropout)
        )

    def forward(self, feats, proxy_feats):
        context = self.object_context_block(feats, proxy_feats)

        output = self.conv_bn_dropout(torch.cat([context, feats], 1))

        return output

class OCRNet(nn.Module):
    def __init__(self, input_chs, inter_chs=128):
        super(OCRNet, self).__init__()
        self.class_num_seg = 47
        self.inter_chs = inter_chs
        # self.att_module = NonLocalAttn(inter_chs)
        self.seg_head_layer = nn.Conv2d(self.inter_chs, self.class_num_seg, kernel_size=1)
        self.backbone = create_model('resnet34', pretrained=True)

        self.ocr_gather_head = SpatialGather_Module(self.class_num_seg)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.inter_chs,
                                                 key_channels=self.inter_chs,
                                                 out_channels=self.inter_chs,
                                                 scale=1,
                                                 dropout=0.05,
                                                 )

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.inter_chs, self.inter_chs,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inter_chs),
            nn.ReLU(inplace=True),
        )
        self.aux_head = nn.Sequential(
            nn.Conv2d(self.inter_chs, self.inter_chs,
                      kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(self.inter_chs),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.inter_chs, self.class_num_seg,
                      kernel_size=1, stride=1, padding=0, bias=True)
        )

    def forward(self, imgs):
        feature = self.feature_forward(imgs)

        feature_ocr = self.conv3x3_ocr(feature)
        out_aux = self.aux_head(feature_ocr)

        context = self.ocr_gather_head(feature_ocr, out_aux)
        feats = self.ocr_distri_head(feature_ocr, context)

        result_seg = self.seg_head_layer(feats)
        result_seg = F.interpolate(result_seg, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        return None, None, None, result_seg

    def feature_forward(self, imgs):
        feature = self.backbone.conv1(imgs)
        feature = self.backbone.bn1(feature)
        feature = self.backbone.act1(feature)
        feature = self.backbone.layer1(feature)
        feature = self.backbone.layer2(feature)
        return feature

class OCRNet2(OCRNet):
    def __init__(self, input_chs, inter_chs=128):
        super(OCRNet, self).__init__()
        self.class_num_seg = 47
        self.inter_chs = inter_chs
        # self.att_module = NonLocalAttn(inter_chs)
        self.seg_head_layer = nn.Conv2d(self.inter_chs, self.class_num_seg, kernel_size=1)
        # self.backbone = create_model('resnet34', pretrained=True)
        self.backbone = create_model('hrnet_w18', pretrained=True, features_only=True)

        self.ocr_gather_head = SpatialGather_Module(self.class_num_seg)

        self.ocr_distri_head = SpatialOCR_Module(in_channels=self.inter_chs,
                                                 key_channels=self.inter_chs,
                                                 out_channels=self.inter_chs,
                                                 scale=1,
                                                 dropout=0.5,
                                                 )

        self.conv3x3_ocr = nn.Sequential(
            nn.Conv2d(self.inter_chs, self.inter_chs,
                      kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(self.inter_chs),
            nn.ReLU(inplace=True),
        )
        self.aux_head = nn.Conv2d(self.inter_chs, self.class_num_seg, kernel_size=1)

    def forward(self, imgs):
        feature = self.backbone(imgs)[1]

        feature_ocr = self.conv3x3_ocr(feature)
        # feature_ocr = feature
        out_aux = self.aux_head(feature_ocr)

        context = self.ocr_gather_head(feature_ocr, out_aux)
        feats = self.ocr_distri_head(feature_ocr, context)

        result_seg = self.seg_head_layer(feats)
        result_seg = F.interpolate(result_seg, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        out_aux = F.interpolate(out_aux, size=imgs.shape[-2:], mode='bilinear', align_corners=False)
        if self.training:
            return None, None, None, [result_seg, out_aux]
        else:
            return None, None, None, result_seg

if __name__ == '__main__':
    pass

    ocr_net = OCRNet2(3).cuda()

    imgs_shape = (2, 3, 224, 224)
    imgs = torch.rand(imgs_shape).cuda()
    result = ocr_net(imgs)
    print(result.shape)
