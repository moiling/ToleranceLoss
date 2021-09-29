import torch
from torch import nn

import torch.nn.functional as F

from networks.backbones.mobilenetv2 import MobileNetV2
from networks.ops.ops import SEBlock, Conv2dIBNormRelu


class MobileNetWrapper(nn.Module):

    def __init__(self, pretrain=False):
        super().__init__()
        self.model = MobileNetV2(in_channels=3, pretrain=pretrain)
        enc_channels = [16, 24, 32, 96, 1280]

        self.se_block = SEBlock(enc_channels[4], enc_channels[4], reduction=4)
        self.conv_lr16x = Conv2dIBNormRelu(enc_channels[4], enc_channels[3], 5, stride=1, padding=2)
        self.conv_lr8x = Conv2dIBNormRelu(enc_channels[3], enc_channels[2], 5, stride=1, padding=2)
        self.conv_lr = Conv2dIBNormRelu(enc_channels[2], 3, kernel_size=3, stride=2, padding=1, with_ibn=False, with_relu=False)

    def forward(self, x):
        _, _, h, w = x.size()
        enc32x = self.model(x)

        enc32x = self.se_block(enc32x)
        lr16x = F.interpolate(enc32x, scale_factor=2, mode='bilinear', align_corners=False)
        lr16x = self.conv_lr16x(lr16x)
        lr8x = F.interpolate(lr16x, scale_factor=2, mode='bilinear', align_corners=False)
        lr8x = self.conv_lr8x(lr8x)
        lr = self.conv_lr(lr8x)  # 16x

        lr = F.interpolate(lr, size=(h, w), mode="bilinear", align_corners=False)

        return lr
