# -*- coding: utf-8 -*-
"""Early-fusion baseline: MFCC, LFCC and CQCC stacked as 3 input channels.

    EarlyFusionDetector()     # one ResNeXt branch over a (3, 13, 398) input

Why this exists
---------------
[11] compares three ways of combining spectral features (Table "Results of
different fusion techniques", ASVspoof 2019 LA, LFCC + MFCC):

    early feature concatenation        3.23 % EER
    classification score-level fusion  1.15 % EER
    feature embedding concatenation    0.72 % EER   <- TripleBranchResNeXt

The triple-branch model reproduces the last row, and the E2 single-branch
detectors make the score-level row available. This module supplies the first
row so that all three fusion strategies are compared on this project's data,
under one protocol.

Input
-----
The three families share one shape (13 x 398) and `FeatureDataset` z-normalises
each with its own train-split mean/std, so they are on a common scale and can
be stacked channel-wise without further alignment.

Provenance
----------
PAPER-SPECIFIED ([11]): early concatenation of features before a single network.
OUR DEFAULTS: the branch is `ResNeXtBranch` unchanged apart from
`in_channels=3`, and the head (embed_dim -> 64 -> 1) is the one used by
`SingleBranchDetector`. Parameter count therefore matches an E2 detector to
within the extra 2 x 3 x 3 x 64 stem weights, so the comparison with E2 is
like-for-like in capacity and the comparison with the triple model isolates
the fusion strategy at a third of the parameters.
"""
import torch
import torch.nn as nn

from .resnext_branch import CARDINALITY, ResNeXtBranch


class EarlyFusionDetector(nn.Module):
    def __init__(self, width_mult=1.0, cardinality=CARDINALITY,
                 hidden=64, dropout=0.3):
        super().__init__()
        self.branch = ResNeXtBranch(width_mult, cardinality, in_channels=3)
        self.head = nn.Sequential(
            nn.Linear(self.branch.embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, mfcc, lfcc, cqcc):
        x = torch.cat([mfcc, lfcc, cqcc], dim=1)            # (B, 3, 13, 398)
        return self.head(self.branch(x)).squeeze(1)         # (B,) raw fake logit

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": total, "head": head, "branch": total - head}
