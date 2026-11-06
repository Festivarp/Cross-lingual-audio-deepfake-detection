# -*- coding: utf-8 -*-
"""Single-branch ResNeXt stack, following Tahaoglu et al. [11], Table 1, PDF p. 7.

[11]'s Table 1 (layer-wise configuration)
-----------------------------------------
    Layer               Filter size/Filters/Stride     Output
    Input               -                              (1, 60, 750)
    Conv0               Conv2d (3 x 3, 64, 1)          (64, 60, 750)
    BN0                 BN (64)                        (64, 60, 750)
    Pool0               MaxPool2d (3 x 3, 2)           (64, 30, 375)
    ResNeXtBlock1       [Conv2d(1x1, 64, 1);           (128, 30, 375)
    ResNeXtBlock2        BN & ReLU;                    (256, 15, 188)
    ResNeXtBlock3        GroupConv2d(3x3, 64, G);      (512,  8,  94)
    ResNeXtBlock4        BN & ReLU;                    (1024, 4,  47)
    AdaptiveAvgPool2d    Conv2d(1x1, 64, 1); BN]       (1024, 1,   1)
                                                       (1024)

What is taken from [11] vs. chosen here
---------------------------------------
PAPER-SPECIFIED:
  * Conv0 = 3x3, 64 filters, stride 1, then BN + ReLU              ([11] p.6, Table 1)
  * Pool0 = MaxPool2d 3x3, stride 2                                (Table 1)
  * Four ResNeXt blocks with output channels 128 -> 256 -> 512 -> 1024  (Table 1)
  * Bottleneck form 1x1 -> grouped 3x3 -> 1x1, each followed by BN  (Table 1)
  * Grouped convolution implements split-transform-merge; the group
    count IS the cardinality                                        ([11] p.6)
  * AdaptiveAvgPool2d to a 1024-d embedding                         (Table 1)

OUR DEFAULTS (not stated unambiguously in [11]):
  * CARDINALITY = 32. Table 1 renders the grouped conv as
    "GroupConv2d(3 x 3, 64, 2)"; the surrounding text (p.6) says the
    parenthesised values are "the number of input channels, the size of the
    convolution kernel, and the number of convolution kernels", which does not
    match a 3-tuple containing a kernel size, so the final value is ambiguous
    between stride and group count. We use 32, the ResNeXt default, and set the
    downsampling stride separately from the block's spatial output shape in
    Table 1 (blocks 2-4 halve resolution, block 1 does not).
  * Bottleneck width = out_channels // 2, the standard ResNeXt ratio.
  * Projection shortcut (1x1 conv + BN) whenever shape changes, per ResNet/ResNeXt
    convention; [11] does not describe the shortcut explicitly.

INPUT-SHAPE DEVIATION (deliberate, documented)
----------------------------------------------
[11] feeds (1, 60, 750): 60 coefficients x 750 frames. This project feeds
(1, 13, 398): 13 coefficients x 398 frames (4.0 s at a 10 ms hop). The 13-
coefficient front-end is fixed by the rest of this project -- both notebooks and
the paper's Table III use it -- so changing it here would break comparability
with our own published feature analysis. The consequence is a much thinner
frequency axis: after Pool0 and three stride-2 blocks the frequency dimension
runs 13 -> 6 -> 6 -> 3 -> 2 -> 1, so the deeper blocks see almost no frequency
extent. This is a real capacity reduction relative to [11] and is reported as a
limitation rather than hidden.
"""
import torch
import torch.nn as nn

CARDINALITY = 32          # our default; see module docstring


class ResNeXtBlock(nn.Module):
    """Bottleneck 1x1 -> grouped 3x3 -> 1x1 with a projection shortcut."""

    def __init__(self, in_ch, out_ch, stride=1, cardinality=CARDINALITY):
        super().__init__()
        width = max(cardinality, (out_ch // 2) // cardinality * cardinality)

        self.conv1 = nn.Conv2d(in_ch, width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.conv2 = nn.Conv2d(width, width, kernel_size=3, stride=stride,
                               padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(width)
        self.conv3 = nn.Conv2d(width, out_ch, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return self.relu(out + self.shortcut(x))


class ResNeXtBranch(nn.Module):
    """One feature family in, one `embed_dim`-d embedding out.

    `width_mult` scales every channel count. The paper's configuration is
    width_mult=1.0 (64/128/256/512/1024). A smaller multiplier is available
    because this project trains on CPU; whatever value is used is recorded in
    the run's README and results JSON.

    `in_channels` is 1 for every per-family branch. The early-fusion baseline
    (`models/early_fusion.py`) passes 3, stacking MFCC/LFCC/CQCC as channels.
    """

    def __init__(self, width_mult=1.0, cardinality=CARDINALITY, in_channels=1):
        super().__init__()
        c = [int(64 * width_mult), int(128 * width_mult), int(256 * width_mult),
             int(512 * width_mult), int(1024 * width_mult)]
        # keep every channel count divisible by the cardinality
        c = [max(cardinality, v // cardinality * cardinality) for v in c]
        self.embed_dim = c[4]

        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, c[0], kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(c[0]),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )
        # Strides follow Table 1's output shapes: block 1 preserves resolution,
        # blocks 2-4 halve it.
        self.block1 = ResNeXtBlock(c[0], c[1], stride=1, cardinality=cardinality)
        self.block2 = ResNeXtBlock(c[1], c[2], stride=2, cardinality=cardinality)
        self.block3 = ResNeXtBlock(c[2], c[3], stride=2, cardinality=cardinality)
        self.block4 = ResNeXtBlock(c[3], c[4], stride=2, cardinality=cardinality)
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):                      # x: (B, in_channels, 13, 398)
        x = self.stem(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return torch.flatten(self.pool(x), 1)  # (B, embed_dim)
