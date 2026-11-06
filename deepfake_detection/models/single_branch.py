# -*- coding: utf-8 -*-
"""One feature family, one ResNeXt branch, one logit.

    SingleBranchDetector("mfcc")      # MFCC only
    SingleBranchDetector("cqcc")      # CQCC only

Why this exists
---------------
E2 of the cost study: establish what each cepstral family is worth *on its own*,
so the cascade's ordering policy can be derived from measured accuracy-per-cost
rather than assumed. The triple-branch model of [11] cannot answer this, because
its MLP fusion head consumes all three embeddings at once.

Interface note
--------------
`forward` deliberately accepts all three feature tensors and uses only its own,
mirroring `TripleBranchResNeXt.forward`. That keeps `train.train_model` and
`evaluate.score_model` -- both of which call `model(mfcc, lfcc, cqcc)` -- working
unchanged. Nothing in the training or evaluation path had to be forked for this.

Provenance
----------
The branch itself is `resnext_branch.ResNeXtBranch`, unchanged, so single-branch
results remain directly comparable to the triple-branch baseline.

OUR DEFAULT (not from [11], which has no single-branch configuration):
  * Head shape embed_dim -> 64 -> 1, ReLU + dropout 0.3. Chosen to mirror the
    second half of the triple-branch fusion MLP (256 -> 64 -> 1) so that the
    single-branch model is not handicapped by a differently shaped classifier.
"""
import torch.nn as nn

from .resnext_branch import CARDINALITY, ResNeXtBranch

FAMILIES = ("mfcc", "lfcc", "cqcc")


class SingleBranchDetector(nn.Module):
    def __init__(self, family, width_mult=1.0, cardinality=CARDINALITY,
                 hidden=64, dropout=0.3):
        super().__init__()
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
        self.family = family
        self.index = FAMILIES.index(family)
        self.branch = ResNeXtBranch(width_mult, cardinality)
        self.head = nn.Sequential(
            nn.Linear(self.branch.embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, mfcc, lfcc, cqcc):
        x = (mfcc, lfcc, cqcc)[self.index]
        return self.head(self.branch(x)).squeeze(1)      # (B,) raw fake logit

    def n_params(self):
        total = sum(p.numel() for p in self.parameters())
        head = sum(p.numel() for p in self.head.parameters())
        return {"total": total, "head": head, "branch": total - head}
