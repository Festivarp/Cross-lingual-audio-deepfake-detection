# -*- coding: utf-8 -*-
"""E2 (per-family branch ablation) and E3 (width sweep), both multi-seed.

    python -m deepfake_detection.run_ablation --e2            # 3 families x 3 seeds
    python -m deepfake_detection.run_ablation --e3            # 3 widths  x 3 seeds
    python -m deepfake_detection.run_ablation --e2 --e3
    python -m deepfake_detection.run_ablation --early         # early fusion x 3 seeds
    python -m deepfake_detection.run_ablation --triple        # triple-branch x 3 seeds

What these answer
-----------------
E2: what is each cepstral family worth on its own? The cascade orders branches by
    measured separability per unit cost, and E1 supplied the cost. This supplies
    the accuracy. It also answers, directly, whether every branch earns its keep.

E3: how much accuracy is lost by shrinking the network? `ResNeXtBranch` already
    takes a `width_mult`, so this is a sweep over existing machinery, not new
    code. Produces the accuracy-versus-parameters half of the Pareto curve.

Fusion comparison ([11]'s early / score / embedding fusion, on our data):
    --early   one branch over MFCC/LFCC/CQCC stacked as 3 channels
              (models/early_fusion.py) -- [11]'s "early feature concatenation".
    --triple  the triple-branch embedding-fusion model, re-trained under this
              protocol on seeds 0-2. The Phase-2 15.00 % is a single run; this
              gives it the same multi-seed footing as E2 and early fusion.
    Score-level fusion needs no training: it is computed from the saved E2
    per-clip scores.

Protocol
--------
* 3 seeds per configuration, reported as mean +/- range. A single unseeded run is
  not a claim; a repeat of the Gujarati evaluation in this project moved by
  0.37 pp, so differences below ~1 pp are noise.
* Identical data, epochs, optimiser and batch size across every configuration,
  so that only the varied factor differs.
* Per-clip scores are saved for every run, because E4's cascade is an
  inference-time policy over these same models and must not retrain anything.
* Resumable: a run whose checkpoint already exists is skipped. Earlier long runs
  in this project were lost to session teardown.
"""
import argparse
import json
import os
import platform
import time

import numpy as np
import torch

from .data.dataset import CACHE_DIR, FeatureDataset
from .evaluate import evaluate, score_model
from .models.early_fusion import EarlyFusionDetector
from .models.single_branch import FAMILIES, SingleBranchDetector
from .models.triple_branch import TripleBranchResNeXt
from .train import BATCH_SIZE, LR, load_checkpoint, make_loader, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints", "ablation")
SCORES = os.path.join(RESULTS, "scores")

EPOCHS = 20                    # same budget as the Phase-2 baseline
SEEDS = (0, 1, 2)
WIDTHS = (0.25, 0.5, 1.0)


def cache(name):
    return os.path.join(CACHE_DIR, f"{name}.npz")


def load_data():
    train_ds = FeatureDataset(cache("asvspoof_train"))
    stats = train_ds.stats                              # train-only statistics
    data = {
        "train": train_ds,
        "val": FeatureDataset(cache("asvspoof_validation"), stats),
        "english": FeatureDataset(cache("asvspoof_test"), stats),
        "gujarati": FeatureDataset(cache("indic_gujarati"), stats),
        "hindi": FeatureDataset(cache("indic_hindi"), stats),
    }
    # speaker-matched Indic sets (Kathbath real, same speakers as the fakes),
    # scored alongside the FLEURS-paired ones when they have been built
    for lang in ("gujarati", "hindi"):
        if os.path.exists(cache(f"indic_{lang}_kb")):
            data[f"{lang}_kb"] = FeatureDataset(cache(f"indic_{lang}_kb"), stats)
    return data


EVAL_SETS = ("english", "gujarati", "hindi", "gujarati_kb", "hindi_kb")


def evaluate_all(model, data, device):
    """EER on each held-out set, plus the per-clip scores E4 will reuse."""
    out, raw = {}, {}
    for name in (n for n in EVAL_SETS if n in data):
        loader = make_loader(data[name])
        out[name] = evaluate(model, loader, device)["eer_percent"]
        s, y = score_model(model, loader, device)
        raw[name] = {"scores": s.tolist(), "labels": y.tolist()}
    # validation scores are what the cascade calibrates its threshold on
    vl = make_loader(data["val"])
    s, y = score_model(model, vl, device)
    raw["validation"] = {"scores": s.tolist(), "labels": y.tolist()}
    out["validation"] = evaluate(model, vl, device)["eer_percent"]
    return out, raw


def run_one(tag, build_model, data, device):
    """Train (or resume) one configuration and return its per-set EERs."""
    ck_path = os.path.join(CKPT, f"{tag}.pt")
    score_path = os.path.join(SCORES, f"{tag}.json")
    model = build_model()

    if os.path.exists(ck_path) and os.path.exists(score_path):
        print(f"  [{tag}] checkpoint present, reusing", flush=True)
        model, _ = load_checkpoint(model, ck_path, device)
        with open(score_path, encoding="utf-8") as fh:
            saved = json.load(fh)
        return saved["eer"], saved.get("wall_clock_minutes")

    t0 = time.time()
    train_model(model, data["train"], data["val"], EPOCHS, ck_path,
                os.path.join(RESULTS, "logs", f"{tag}.csv"),
                device=device, tag=tag)
    model, _ = load_checkpoint(model, ck_path, device)
    eer, raw = evaluate_all(model, data, device)
    minutes = round((time.time() - t0) / 60.0, 2)

    os.makedirs(SCORES, exist_ok=True)
    with open(score_path, "w", encoding="utf-8") as fh:
        json.dump({"tag": tag, "eer": eer, "wall_clock_minutes": minutes,
                   "raw": raw}, fh)
    print(f"  [{tag}] EN {eer['english']:.2f}%  GU {eer['gujarati']:.2f}%  "
          f"HI {eer['hindi']:.2f}%  ({minutes:.1f} min)", flush=True)
    return eer, minutes


def aggregate(runs):
    """mean and range across seeds, per evaluation set."""
    keys = runs[0].keys()
    return {k: {"mean": round(float(np.mean([r[k] for r in runs])), 3),
                "min": round(float(min(r[k] for r in runs)), 3),
                "max": round(float(max(r[k] for r in runs)), 3),
                "range": round(float(max(r[k] for r in runs)
                                     - min(r[k] for r in runs)), 3)}
            for k in keys}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--e2", action="store_true", help="per-family branch ablation")
    ap.add_argument("--e3", action="store_true", help="width sweep")
    ap.add_argument("--early", action="store_true", help="early-fusion baseline")
    ap.add_argument("--triple", action="store_true",
                    help="triple-branch embedding fusion, multi-seed")
    ap.add_argument("--seeds", type=int, default=len(SEEDS))
    args = ap.parse_args()
    if not (args.e2 or args.e3 or args.early or args.triple):
        ap.error("pick at least one of --e2 / --e3 / --early / --triple")

    os.makedirs(CKPT, exist_ok=True)
    os.makedirs(SCORES, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    seeds = SEEDS[:args.seeds]

    data = load_data()
    print(f"device {device} | threads {torch.get_num_threads()} | "
          f"train {len(data['train'])} clips | seeds {list(seeds)}\n", flush=True)

    out = {"environment": {"device": device, "cpu": platform.processor(),
                           "torch": torch.__version__,
                           "threads": torch.get_num_threads(),
                           "epochs": EPOCHS, "lr": LR, "batch_size": BATCH_SIZE,
                           "seeds": list(seeds)}}

    if args.e2:
        print("=" * 66)
        print("E2 - per-family branch ablation")
        print("=" * 66, flush=True)
        e2 = {}
        for fam in FAMILIES:
            runs, mins = [], []
            for sd in seeds:
                torch.manual_seed(sd); np.random.seed(sd)
                eer, m = run_one(
                    f"e2_{fam}_seed{sd}",
                    lambda f=fam: SingleBranchDetector(f, width_mult=1.0),
                    data, device)
                runs.append(eer); mins.append(m or 0)
            n = SingleBranchDetector(fam).n_params()
            e2[fam] = {"per_seed": runs, "aggregate": aggregate(runs),
                       "params": n, "mean_minutes": round(float(np.mean(mins)), 2)}
            a = e2[fam]["aggregate"]
            print(f"  -> {fam.upper()}: EN {a['english']['mean']:.2f}% "
                  f"(range {a['english']['range']:.2f})  "
                  f"params {n['total']:,}\n", flush=True)
        out["e2_branch_ablation"] = e2

    if args.e3:
        print("=" * 66)
        print("E3 - width sweep, full triple-branch model")
        print("=" * 66, flush=True)
        e3 = {}
        for w in WIDTHS:
            runs, mins = [], []
            for sd in seeds:
                torch.manual_seed(sd); np.random.seed(sd)
                eer, m = run_one(
                    f"e3_w{w}_seed{sd}",
                    lambda ww=w: TripleBranchResNeXt(width_mult=ww),
                    data, device)
                runs.append(eer); mins.append(m or 0)
            n = TripleBranchResNeXt(width_mult=w).n_params()
            e3[str(w)] = {"per_seed": runs, "aggregate": aggregate(runs),
                          "params": n, "mean_minutes": round(float(np.mean(mins)), 2)}
            a = e3[str(w)]["aggregate"]
            print(f"  -> width {w}: EN {a['english']['mean']:.2f}% "
                  f"(range {a['english']['range']:.2f})  "
                  f"params {n['total']:,}\n", flush=True)
        out["e3_width_sweep"] = e3

    fusion = [(flag, key, tag, build) for flag, key, tag, build in (
        (args.early, "early_fusion", "early", lambda: EarlyFusionDetector(width_mult=1.0)),
        (args.triple, "triple_embedding_fusion", "triple",
         lambda: TripleBranchResNeXt(width_mult=1.0)),
    ) if flag]
    for _, key, tag, build in fusion:
        print("=" * 66)
        print(f"Fusion comparison - {key}")
        print("=" * 66, flush=True)
        runs, mins = [], []
        for sd in seeds:
            torch.manual_seed(sd); np.random.seed(sd)
            eer, m = run_one(f"{tag}_seed{sd}", build, data, device)
            runs.append(eer); mins.append(m or 0)
        n = build().n_params()
        out[key] = {"per_seed": runs, "aggregate": aggregate(runs),
                    "params": n, "mean_minutes": round(float(np.mean(mins)), 2)}
        a = out[key]["aggregate"]
        print(f"  -> {key}: EN {a['english']['mean']:.2f}% "
              f"(range {a['english']['range']:.2f})  "
              f"params {n['total']:,}\n", flush=True)

    path = os.path.join(RESULTS, "ablation_results.json")
    prev = {}
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            prev = json.load(fh)
    prev.update(out)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(prev, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
