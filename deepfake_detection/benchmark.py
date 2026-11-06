# -*- coding: utf-8 -*-
"""E1: end-to-end cost accounting for tri-feature audio deepfake detection.

    python -m deepfake_detection.benchmark --components

Why this exists
---------------
Every "lightweight detector" claim in the audio deepfake detection literature
counts model parameters and reports inference FLOPs. None of them report what
the *front end* costs. That omission matters here because the three cepstral
families in [11] are not equally cheap to compute:

  * MFCC and LFCC go through torchaudio's FFT-based filterbanks.
  * CQCC goes through `librosa.cqt` (features/extract.py:124), a Constant-Q
    Transform, which is a substantially heavier computation.

A detector that runs all three branches on every clip pays the CQT toll on
every clip, whether or not the CQCC branch changes the verdict. This module
measures that toll so the cascade's ordering policy can be derived from real
numbers rather than assumed ones.

Method
------
Median over `--reps` timed calls after `--warmup` untimed ones. Median, not
mean, because a background process on a laptop produces occasional large
outliers that a mean would absorb and misreport. Disk I/O is excluded: the
waveform is loaded and decoded once, before timing starts.
"""
import argparse
import json
import os
import platform
import statistics
import time

import librosa
import numpy as np
import torch
from scipy.fftpack import dct

from .features.extract import (HOP_LENGTH, N_COEFF, N_FRAMES, TARGET_SR,
                               fix_duration, lfcc_transform, load_waveform,
                               mfcc_transform)
from .models.triple_branch import TripleBranchResNeXt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
DEMO = os.path.join(ROOT, "demo_audio", "analysed_4s")


def time_call(fn, reps, warmup):
    """Median milliseconds per call."""
    for _ in range(warmup):
        fn()
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return {
        "median_ms": round(statistics.median(samples), 3),
        "mean_ms": round(statistics.mean(samples), 3),
        "min_ms": round(min(samples), 3),
        "max_ms": round(max(samples), 3),
    }


# ---- the three feature front ends, timed separately ---------------------
# These replicate features/extract.py exactly. They are duplicated rather than
# imported because extract_features() computes all three together and we need
# each one's cost in isolation.

def _mfcc(waveform):
    return mfcc_transform(waveform).squeeze(0).numpy()[:, :N_FRAMES]


def _lfcc(waveform):
    return lfcc_transform(waveform).squeeze(0).numpy()[:, :N_FRAMES]


def _cqcc(waveform):
    waveform_np = waveform.squeeze().numpy()
    cqt = np.abs(librosa.cqt(waveform_np, sr=TARGET_SR, hop_length=HOP_LENGTH))
    log_cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    return dct(log_cqt, type=2, axis=0, norm="ortho")[:N_COEFF][:, :N_FRAMES]


def load_probe_clip():
    """One real 4 s clip, decoded before timing begins."""
    if os.path.isdir(DEMO):
        wavs = sorted(f for f in os.listdir(DEMO) if f.endswith(".wav"))
        if wavs:
            path = os.path.join(DEMO, wavs[0])
            return fix_duration(load_waveform(path)), wavs[0]
    # Fall back to noise: timing does not depend on content, only on shape.
    torch.manual_seed(0)
    return torch.randn(1, int(4.0 * TARGET_SR)), "synthetic noise (4.0 s)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--components", action="store_true",
                    help="time feature extraction and model components")
    ap.add_argument("--reps", type=int, default=30)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--batch", type=int, default=32,
                    help="batch size for the throughput measurement")
    args = ap.parse_args()

    os.makedirs(RESULTS, exist_ok=True)
    torch.set_num_threads(torch.get_num_threads())   # record, do not override

    waveform, clip_name = load_probe_clip()
    print(f"probe clip: {clip_name}")
    print(f"cpu: {platform.processor()}")
    print(f"torch threads: {torch.get_num_threads()}\n")

    # ---- feature extraction ---------------------------------------------
    feat = {
        "mfcc": time_call(lambda: _mfcc(waveform), args.reps, args.warmup),
        "lfcc": time_call(lambda: _lfcc(waveform), args.reps, args.warmup),
        "cqcc": time_call(lambda: _cqcc(waveform), args.reps, args.warmup),
    }

    print("=" * 64)
    print("FEATURE EXTRACTION  (per 4.0 s clip)")
    print("=" * 64)
    print(f"  {'family':<10}{'median ms':>12}{'min':>10}{'max':>10}{'x MFCC':>10}")
    base = feat["mfcc"]["median_ms"]
    for k in ("mfcc", "lfcc", "cqcc"):
        m = feat[k]
        print(f"  {k.upper():<10}{m['median_ms']:>12.2f}{m['min_ms']:>10.2f}"
              f"{m['max_ms']:>10.2f}{m['median_ms'] / base:>9.1f}x")
    total_feat = sum(feat[k]["median_ms"] for k in feat)
    cqcc_share = 100.0 * feat["cqcc"]["median_ms"] / total_feat
    print(f"\n  all three: {total_feat:.2f} ms")
    print(f"  CQCC is {cqcc_share:.1f}% of total feature-extraction cost")

    # ---- model forward ---------------------------------------------------
    model = TripleBranchResNeXt(width_mult=1.0).eval()
    n = model.n_params()

    x1 = torch.randn(1, 1, N_COEFF, N_FRAMES)
    xb = torch.randn(args.batch, 1, N_COEFF, N_FRAMES)

    with torch.no_grad():
        one_branch = time_call(lambda: model.mfcc_branch(x1),
                               args.reps, args.warmup)
        full_1 = time_call(lambda: model(x1, x1, x1), args.reps, args.warmup)
        full_b = time_call(lambda: model(xb, xb, xb),
                           max(5, args.reps // 3), args.warmup)

    per_clip_batched = full_b["median_ms"] / args.batch

    print("\n" + "=" * 64)
    print("MODEL FORWARD")
    print("=" * 64)
    print(f"  one ResNeXt branch, batch 1    {one_branch['median_ms']:>9.2f} ms")
    print(f"  full triple model, batch 1     {full_1['median_ms']:>9.2f} ms")
    print(f"  full triple model, batch {args.batch:<3}   "
          f"{full_b['median_ms']:>9.2f} ms  ({per_clip_batched:.2f} ms/clip)")
    print(f"\n  parameters: {n['total']:,} total "
          f"({n['branches']:,} branches + {n['fusion']:,} fusion)")

    # ---- what the cascade is competing against ---------------------------
    baseline_total = total_feat + full_1["median_ms"]
    mfcc_only = feat["mfcc"]["median_ms"] + one_branch["median_ms"]

    print("\n" + "=" * 64)
    print("END-TO-END, BATCH 1  (what a deployed detector actually pays)")
    print("=" * 64)
    print(f"  baseline [11], all three branches   {baseline_total:>9.2f} ms")
    print(f"  MFCC branch alone                   {mfcc_only:>9.2f} ms")
    print(f"  ratio                               {baseline_total / mfcc_only:>9.2f}x")
    print(f"\n  feature extraction is "
          f"{100.0 * total_feat / baseline_total:.1f}% of end-to-end cost")

    out = {
        "experiment": "E1 - end-to-end cost accounting",
        "environment": {
            "cpu": platform.processor(),
            "platform": platform.platform(),
            "torch_threads": torch.get_num_threads(),
            "torch_version": torch.__version__,
            "probe_clip": clip_name,
            "reps": args.reps,
            "warmup": args.warmup,
        },
        "feature_extraction_ms": feat,
        "feature_extraction_total_ms": round(total_feat, 3),
        "cqcc_share_of_features_percent": round(cqcc_share, 2),
        "model": {
            "one_branch_batch1": one_branch,
            "full_batch1": full_1,
            f"full_batch{args.batch}": full_b,
            "per_clip_batched_ms": round(per_clip_batched, 3),
            "params": n,
        },
        "end_to_end_batch1_ms": {
            "baseline_all_three": round(baseline_total, 3),
            "mfcc_branch_only": round(mfcc_only, 3),
            "ratio": round(baseline_total / mfcc_only, 3),
            "feature_share_percent": round(100.0 * total_feat / baseline_total, 2),
        },
    }
    path = os.path.join(RESULTS, "e1_cost_accounting.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
