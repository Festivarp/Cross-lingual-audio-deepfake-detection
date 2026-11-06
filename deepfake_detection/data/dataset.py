# -*- coding: utf-8 -*-
"""PyTorch Dataset over cached feature tensors, plus the caching driver.

Features are extracted once and cached to .npz, because the expensive part of
this pipeline is network I/O, not computation: ASVspoof streams at roughly
7 rows/s. Caching means Phases 2-4 can be re-run, and epochs iterated, without
touching the network again.

Each cache file holds:
    mfcc  (N, 13, 398) float32
    lfcc  (N, 13, 398) float32
    cqcc  (N, 13, 398) float32
    label (N,)         int64      0 = real/bonafide, 1 = fake/spoof
"""
import os
import time

import numpy as np
import torch
from torch.utils.data import Dataset

from ..features.extract import extract_features

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__)))), "cache")


def build_cache(records, name, cache_dir=CACHE_DIR, verbose=True):
    """Extract features for `records` ({'audio','label'}) and save to `name`.npz."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{name}.npz")
    if os.path.exists(path):
        if verbose:
            print(f"  cache hit: {path}", flush=True)
        return path

    t0 = time.time()
    mf, lf, cq, ys, failed = [], [], [], [], 0
    # optional per-clip metadata (speaker-matched Indic sets carry it, so that
    # later train/test splits can be made speaker-disjoint)
    meta_keys = [k for k in ("speaker", "source") if records and all(k in r for r in records)]
    meta = {k: [] for k in meta_keys}
    for i, r in enumerate(records):
        try:
            a, b, c = extract_features(r["audio"])
        except Exception as e:                      # a corrupt clip must not kill the run
            failed += 1
            if failed <= 3:
                print(f"    skip {i}: {type(e).__name__}: {e}", flush=True)
            continue
        mf.append(a); lf.append(b); cq.append(c); ys.append(int(r["label"]))
        for k in meta_keys:
            meta[k].append(str(r[k]))
        if verbose and (i + 1) % 200 == 0:
            print(f"    {i + 1}/{len(records)} ({time.time() - t0:.0f}s)", flush=True)

    np.savez_compressed(path, mfcc=np.stack(mf), lfcc=np.stack(lf),
                        cqcc=np.stack(cq), label=np.array(ys, dtype="int64"),
                        **{k: np.array(v) for k, v in meta.items()})
    if verbose:
        n = len(ys)
        print(f"  wrote {path}: {n} clips "
              f"({int(sum(1 for y in ys if y == 0))} real / "
              f"{int(sum(ys))} fake), {failed} failed, {time.time() - t0:.0f}s",
              flush=True)
    return path


class FeatureDataset(Dataset):
    """Serves (mfcc, lfcc, cqcc, label) from one or more cache files.

    Per-feature standardisation uses statistics computed on the *training* cache
    and passed in for val/eval, so no evaluation statistic leaks into training.
    """

    def __init__(self, paths, stats=None):
        if isinstance(paths, (str, os.PathLike)):
            paths = [paths]
        mf, lf, cq, ys = [], [], [], []
        for p in paths:
            d = np.load(p)
            mf.append(d["mfcc"]); lf.append(d["lfcc"])
            cq.append(d["cqcc"]); ys.append(d["label"])
        self.mfcc = torch.from_numpy(np.concatenate(mf))
        self.lfcc = torch.from_numpy(np.concatenate(lf))
        self.cqcc = torch.from_numpy(np.concatenate(cq))
        self.label = torch.from_numpy(np.concatenate(ys)).float()

        if stats is None:
            stats = {}
            for k, t in (("mfcc", self.mfcc), ("lfcc", self.lfcc), ("cqcc", self.cqcc)):
                stats[k] = (float(t.mean()), float(t.std()) or 1.0)
        self.stats = stats
        for k, t in (("mfcc", self.mfcc), ("lfcc", self.lfcc), ("cqcc", self.cqcc)):
            m, s = stats[k]
            t.sub_(m).div_(s if s else 1.0)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, i):
        return (self.mfcc[i].unsqueeze(0),      # (1, 13, 398) - single-channel image
                self.lfcc[i].unsqueeze(0),
                self.cqcc[i].unsqueeze(0),
                self.label[i])

    def class_counts(self):
        n_fake = int(self.label.sum().item())
        return {"real": len(self) - n_fake, "fake": n_fake}
