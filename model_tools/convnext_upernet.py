import torch
import torch.nn as nn
import torch.nn.functional as F

def conv2d(chIn, chOut, kernel_size, stride, padding, bias=True, norm=True, relu=False, inplace=True):
    layers = []
    layers.append(nn.Conv2d(chIn, chOut, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias))
    if norm:
        layers.append(nn.BatchNorm2d(chOut, affine=bias))
    if relu:
        layers.append(nn.ReLU(inplace=inplace))
    return nn.Sequential(*layers)

class PPM(nn.ModuleList):
    """Pooling Pyramid Module used in PSPNet.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module.
        in_channels (int): Input channels.
        channels (int): Channels after modules, before conv_seg.
        conv_cfg (dict|None): Config of conv layers.
        norm_cfg (dict|None): Config of norm layers.
        act_cfg (dict): Config of activation layers.
        align_corners (bool): align_corners argument of F.interpolate.
    """

    def __init__(self, pool_scales, in_channels, channels, conv_cfg, norm_cfg,
                 act_cfg, align_corners):
        super(PPM, self).__init__()
        self.pool_scales = pool_scales
        self.align_corners = align_corners
        self.in_channels = in_channels
        self.channels = channels
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        for pool_scale in pool_scales:
            self.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(pool_scale),
                    conv2d(self.in_channels, self.channels,
                           kernel_size=1, stride=1, padding=0, bias=True, norm=False, relu=True)))

    def forward(self, x):
        """Forward function."""
        ppm_outs = []
        for ppm in self:
            ppm_out = ppm(x)
            upsampled_ppm_out = F.interpolate(ppm_out, size=x.size()[2:],
                                              mode='bilinear', align_corners=self.align_corners)
            ppm_outs.append(upsampled_ppm_out)
        return ppm_outs

class UPerHead(nn.Module):
    """Unified Perceptual Parsing for Scene Understanding.
    This head is the implementation of `UPerNet
    <https://arxiv.org/abs/1807.10221>`_.
    Args:
        pool_scales (tuple[int]): Pooling scales used in Pooling Pyramid
            Module applied on the last feature. Default: (1, 2, 3, 6).
    """

    def __init__(self, in_channels=[96, 192, 384, 768], out_channels=2, out_shape=(256, 256), up_layers=2,
                 in_index=[0, 1, 2, 3], pool_scales=(1, 2, 3, 6), channels=512, **kwargs):
        super(UPerHead, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.out_shape = out_shape
        self.in_index = in_index
        self.pool_scales = pool_scales
        self.channels = channels
        self.conv_cfg = None
        self.norm_cfg = None
        self.act_cfg = None
        self.align_corners = False
        # PSP Module
        self.psp_modules = PPM(
            pool_scales,
            self.in_channels[-1],
            self.channels,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg,
            align_corners=self.align_corners)
        self.bottleneck = conv2d(self.in_channels[-1] + len(pool_scales) * self.channels, self.channels,
                                 kernel_size=3, stride=1, padding=1, bias=True, norm=True, relu=True)
        # FPN Module
        self.lateral_convs = nn.ModuleList()
        self.fpn_convs = nn.ModuleList()
        for in_channels in self.in_channels[:-1]:  # skip the top layer
            l_conv = conv2d(in_channels, self.channels,
                            kernel_size=1, stride=1, padding=0, bias=True, norm=True, relu=True, inplace=False)
            fpn_conv = conv2d(self.channels, self.channels,
                              kernel_size=3, stride=1, padding=1, bias=True, norm=True, relu=True, inplace=False)
            self.lateral_convs.append(l_conv)
            self.fpn_convs.append(fpn_conv)

        self.fpn_bottleneck = conv2d(len(self.in_channels) * self.channels, self.channels,
                                     kernel_size=3, stride=1, padding=1, bias=True, norm=True, relu=True)

        self.upscale = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up_layers = up_layers
        self.tails = nn.ModuleList()
        for idx in range(self.up_layers):
            self.tails.append(conv2d(self.channels, self.channels, kernel_size=3, stride=1,
                                     padding=1, bias=True, norm=True, relu=True))
        self.cls_seg = conv2d(self.channels, self.out_channels,
                              kernel_size=3, stride=1, padding=1, bias=True, norm=False, relu=False)

    def psp_forward(self, inputs):
        """Forward function of PSP module."""
        x = inputs[-1]
        psp_outs = [x]
        psp_outs.extend(self.psp_modules(x))
        psp_outs = torch.cat(psp_outs, dim=1)
        output = self.bottleneck(psp_outs)

        return output

    def _transform_inputs(self, inputs):
        return inputs

    def forward(self, inputs):
        """Forward function."""

        inputs = self._transform_inputs(inputs)

        # build laterals
        laterals = [
            lateral_conv(inputs[i])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        laterals.append(self.psp_forward(inputs))

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            prev_shape = laterals[i - 1].shape[2:]
            laterals[i - 1] = laterals[i - 1] + F.interpolate(laterals[i], size=prev_shape,
                                                              mode='bilinear', align_corners=self.align_corners)

        # build outputs
        fpn_outs = [
            self.fpn_convs[i](laterals[i])
            for i in range(used_backbone_levels - 1)
        ]
        # append psp feature
        fpn_outs.append(laterals[-1])

        for i in range(used_backbone_levels - 1, 0, -1):
            fpn_outs[i] = F.interpolate(fpn_outs[i], size=fpn_outs[0].shape[2:],
                                        mode='bilinear', align_corners=self.align_corners)
        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn = self.fpn_bottleneck(fpn_outs)
        tail = fpn
        scale = 2
        for layer in self.tails:
            tail = layer(tail)
            tail = self.upscale(tail) + F.interpolate(fpn, scale_factor=scale, mode='bilinear',
                                                      align_corners=self.align_corners)
            scale *= 2
        output = self.cls_seg(tail)
        output = F.interpolate(output, self.out_shape, mode='bilinear', align_corners=self.align_corners)
        return output

from timm import create_model
class ConvNext_UperNet(nn.Module):
    def __init__(self, in_channels, **kwargs):
        super(ConvNext_UperNet, self).__init__()
        self.class_num_seg = 47
        self.backbone = create_model('convnext_tiny', pretrained=True)
        self.uperhead = UPerHead(in_channels=[96, 192, 384, 768],
                                 out_channels=self.class_num_seg ,
                                 out_shape=(224, 224),
                                 in_index=(0, 1, 2, 3),
                                 pool_scales=(1, 2, 3, 6),
                                 channels=512,
                                 num_classes=self.class_num_seg,
                                 )


    def feature_forward(self, inputs):
        features = self.backbone.stem(inputs)
        features0 = self.backbone.stages[0](features)
        features1 = self.backbone.stages[1](features0)
        features2 = self.backbone.stages[2](features1)
        features3 = self.backbone.stages[3](features2)
        # features = self.backbone.norm_pre(features3)
        # features = self.backbone.head(features)
        return features0, features1, features2, features3

    def forward(self, x):
        x = self.feature_forward(x)
        y = self.uperhead(x)
        return y

if __name__ == '__main__':
    model = ConvNext_UperNet(3)
    data = torch.rand(1, 3, 224, 224)
    print(data.shape)
    result = model(data)
    print(result.shape)

