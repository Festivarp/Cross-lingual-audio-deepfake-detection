# -*- coding: utf-8 -*-
"""Scoring: Equal Error Rate, and a note on min t-DCF.

EER convention follows Tahaoglu et al. [11], PDF p. 8, Eqs. (23)-(25): scores are
countermeasure outputs where a *higher* score means more likely bonafide; the
threshold is the point at which the false-acceptance and false-rejection rates
are equal, and EER is that common value.

min t-DCF is deliberately NOT computed. [11] Eq. (26) (PDF p. 8) defines it over
the *combined* ASV + CM system, requiring the miss and false-alarm rates of a
fixed automatic speaker verification back-end on the same trials. No ASV
back-end is available in this project, and the ASVspoof mirror used here ships
no ASV scores. Computing a "min t-DCF" from countermeasure scores alone would
produce a number that is not comparable to any published t-DCF, so this module
returns None for it and the results table records it as not computed.
"""
import numpy as np
import torch


def compute_eer(scores, labels):
    """EER for scores where higher = more likely REAL, labels 0=real, 1=fake.

    Returns (eer, threshold). Implemented by sweeping every candidate threshold
    given by the sorted scores, which is exact for finite samples.
    """
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels).astype(int)

    real = scores[labels == 0]
    fake = scores[labels == 1]
    if len(real) == 0 or len(fake) == 0:
        return float("nan"), float("nan")

    order = np.argsort(scores)
    thresholds = np.concatenate([[-np.inf], scores[order], [np.inf]])

    # FRR: real clips scored below threshold (wrongly rejected)
    # FAR: fake clips scored at/above threshold (wrongly accepted)
    frr = np.searchsorted(np.sort(real), thresholds, side="left") / len(real)
    far = 1.0 - np.searchsorted(np.sort(fake), thresholds, side="left") / len(fake)

    i = int(np.nanargmin(np.abs(frr - far)))
    return float((frr[i] + far[i]) / 2.0), float(thresholds[i])


def compute_min_tdcf(*_args, **_kwargs):
    """Not computable here -- see module docstring. Returns None by design."""
    return None


@torch.no_grad()
def score_model(model, loader, device="cpu"):
    """Returns (scores, labels). Score = logit for REAL, i.e. negated fake logit."""
    model.eval()
    scores, labels = [], []
    for mfcc, lfcc, cqcc, y in loader:
        logit = model(mfcc.to(device), lfcc.to(device), cqcc.to(device))
        # model emits a "fake" logit; EER convention wants higher = more real
        scores.append((-logit).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(scores), np.concatenate(labels)


def evaluate(model, loader, device="cpu"):
    scores, labels = score_model(model, loader, device)
    eer, thr = compute_eer(scores, labels)
    return {"eer": eer, "eer_percent": 100.0 * eer, "threshold": thr,
            "n": int(len(labels)), "n_real": int((labels == 0).sum()),
            "n_fake": int((labels == 1).sum()), "min_tdcf": compute_min_tdcf()}
