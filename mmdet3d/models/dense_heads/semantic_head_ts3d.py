import numpy as np

try:
    import MinkowskiEngine as ME
except ImportError:
    import warnings
    warnings.warn(
        'Please follow `getting_started.md` to install MinkowskiEngine.`')

import torch
from mmcv.runner import BaseModule
from torch import nn

from mmdet3d.core.bbox.structures import rotation_3d_in_axis
from mmdet3d.models.builder import HEADS, build_loss, build_backbone

from icecream import ic
import pdb

@HEADS.register_module()
class SemanticHead_TS3D(BaseModule):
    def __init__(self,
                 in_channels,
                 out_channels,
                 n_feat,
                 train_cfg=None,
                 test_cfg=None):
        super(SemanticHead_TS3D, self).__init__()
        self.n_feat = n_feat
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self._init_layers(in_channels, out_channels)
    @staticmethod
    def make_block(in_channels, out_channels, kernel_size=3):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels,
                                    kernel_size=kernel_size, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    @staticmethod
    def make_down_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolution(in_channels, out_channels, kernel_size=3,
                                    stride=2, dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    @staticmethod
    def make_up_block(in_channels, out_channels):
        return nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                in_channels,
                out_channels,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(out_channels),
            ME.MinkowskiReLU(inplace=True))

    def _init_layers(self, in_channels, out_channels):
        self.upsample_st_2 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                128,
                64,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))
        self.upsample_st_1 = nn.Sequential(
            ME.MinkowskiConvolutionTranspose(
                64,
                64,
                kernel_size=3,
                stride=2,
                dimension=3),
            ME.MinkowskiBatchNorm(64),
            ME.MinkowskiReLU(inplace=True))
        self.conv_32_ch = nn.Sequential(
            ME.MinkowskiConvolution(
                64,
                32,
                kernel_size=3,
                stride=1,
                dimension=3),
            ME.MinkowskiBatchNorm(32),
            ME.MinkowskiReLU(inplace=True))

        self.final = ME.MinkowskiConvolution(
            32,
            self.n_feat,
            kernel_size=1,
            dimension=3)

        for i in range(len(in_channels)):
            if i > 0:
                self.__setattr__(
                    f'up_block_{i}',
                    self.make_up_block(in_channels[i], in_channels[i - 1]))
            if i < len(in_channels) - 1:
                self.__setattr__(
                    f'lateral_block_{i}',
                    self.make_block(in_channels[i], in_channels[i]))
            if i == 0:
                self.__setattr__(
                    f'out_block_{i}',
                    self.make_block(in_channels[i], out_channels))

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, ME.MinkowskiConvolution):
                ME.utils.kaiming_normal_(
                    m.kernel, mode='fan_out', nonlinearity='relu')

            if isinstance(m, ME.MinkowskiBatchNorm):
                nn.init.constant_(m.bn.weight, 1)
                nn.init.constant_(m.bn.bias, 0)

    def _forward_semantic(self, x):
        feats_st_2 = x[0]
        inputs = x[1:]
        x = inputs[-1]
        for i in range(len(inputs) - 1, -1, -1):
            if i < len(inputs) - 1:
                x = self.__getattr__(f'up_block_{i + 1}')(x)
                x = inputs[i] + x
                x = self.__getattr__(f'lateral_block_{i}')(x)
            if i == 0:
                out = self.__getattr__(f'out_block_{i}')(x)
        seg_feats = self.upsample_st_1(self.upsample_st_2(out) + feats_st_2)
        seg_feats = self.conv_32_ch(seg_feats)
        return seg_feats

    def forward(self, batch_dict):
        x = batch_dict['backbone_feat']
        seg_feats = self._forward_semantic(x)
        return self.final(seg_feats), seg_feats
