# -*- coding: utf-8 -*-
"""Triple-Branch ResNeXt with MLP fusion (Tahaoglu et al. [11]).

Three parallel ResNeXt branches -- one each for MFCC, LFCC and CQCC -- produce
independent embeddings that an MLP head fuses into a single real/fake logit.

Provenance
----------
PAPER-SPECIFIED ([11]):
  * Three spectral feature streams processed separately, fused by an MLP
    ("a Multi-Layer Perceptron (MLP)-based fusion technique is employed",
     abstract, PDF p. 1; contributions, PDF p. 2; "MLP-based classification of
     concatenated feature embeddings", PDF p. 14).
  * Per-branch architecture: see `resnext_branch.py`.

OUR DEFAULTS (not stated in [11]):
  * MLP head shape: concat(3 x embed_dim) -> 256 -> 64 -> 1, ReLU + dropout 0.3.
    [11] states that an MLP fuses the embeddings but does not give its layer
    sizes.
  * Single logit with BCEWithLogitsLoss. [11] does not name its loss function;
    binary cross-entropy on one logit is the standard choice for this task and
    is equivalent, up to parameterisation, to a two-class softmax.

Why the fusion head is separated out
------------------------------------
`fusion_parameters()` and `freeze_branches()` exist for Phase 4, where only the
fusion layer is adapted to the target language while the three convolutional
branches stay frozen. That is the paper's "lightweight adaptation" strategy, and
it is viable precisely because [11] places a learned combiner at this point in
the network.
"""
import torch
import torch.nn as nn

from .resnext_branch import CARDINALITY, ResNeXtBranch


class TripleBranchResNeXt(nn.Module):
    def __init__(self, width_mult=1.0, cardinality=CARDINALITY,
                 hidden=(256, 64), dropout=0.3):
        super().__init__()
        self.mfcc_branch = ResNeXtBranch(width_mult, cardinality)
        self.lfcc_branch = ResNeXtBranch(width_mult, cardinality)
        self.cqcc_branch = ResNeXtBranch(width_mult, cardinality)

        d = self.mfcc_branch.embed_dim * 3
        layers, prev = [], d
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.fusion = nn.Sequential(*layers)

    def forward(self, mfcc, lfcc, cqcc):
        e = torch.cat([self.mfcc_branch(mfcc),
                       self.lfcc_branch(lfcc),
                       self.cqcc_branch(cqcc)], dim=1)
        return self.fusion(e).squeeze(1)          # (B,) raw logit

    # ---- Phase 4 helpers -------------------------------------------------
    def freeze_branches(self):
        """Freeze the three ResNeXt branches; leave the fusion head trainable."""
        for br in (self.mfcc_branch, self.lfcc_branch, self.cqcc_branch):
            for p in br.parameters():
                p.requires_grad = False
            br.eval()                 # keep BatchNorm running stats fixed too
        return self

    def fusion_parameters(self):
        return self.fusion.parameters()

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        fusion = sum(p.numel() for p in self.fusion.parameters())
        return {"total": total, "fusion": fusion, "branches": total - fusion}
