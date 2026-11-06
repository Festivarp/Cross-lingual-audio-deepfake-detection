# -*- coding: utf-8 -*-
"""Phases 2-4 driver. Run from d:\\RMS after cache_data has completed:

    python -m deepfake_detection.run_phases

Phase 2  base training on ASVspoof 2019 LA (English source domain) -> baseline EER
Phase 3  zero-shot evaluation on Gujarati and Hindi -> the generalization gap
Phase 4  fine-tune the MLP fusion layer only, on 10% of each Indic set

Writes results/table_v_results.json, results/table_v_results.md,
results/training_log*.csv and checkpoints/*.pt.

NO-FABRICATION: every number written by this script is produced by code in this
script during the run. Where a quantity cannot be computed (min t-DCF, which
needs an ASV back-end -- see evaluate.py) it is recorded as null, never guessed.
"""
import json
import os
import platform
import time

import numpy as np
import torch

from .data.dataset import CACHE_DIR, FeatureDataset
from .evaluate import evaluate
from .models.triple_branch import TripleBranchResNeXt
from .train import BATCH_SIZE, LR, load_checkpoint, make_loader, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")

WIDTH_MULT = 1.0        # [11] Table 1 channel widths exactly (64..1024)
EPOCHS_BASE = 20        # [11] p.8 uses 100 on an RTX 3090; this is a CPU run
EPOCHS_FT = 15          # fusion-only adaptation, far cheaper per epoch
FT_FRACTION = 0.10      # "10% of the Indic datasets", per the roadmap
SEED = 42


def cache(name):
    return os.path.join(CACHE_DIR, f"{name}.npz")


def stratified_split(ds, fraction, seed=SEED):
    """Split indices into (fine-tune, held-out), stratified by label."""
    y = ds.label.numpy().astype(int)
    rng = np.random.default_rng(seed)
    ft, ho = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        k = max(1, int(round(fraction * len(idx))))
        ft += list(idx[:k]); ho += list(idx[k:])
    return sorted(ft), sorted(ho)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    os.makedirs(os.path.join(RESULTS, "figures"), exist_ok=True)
    os.makedirs(CKPT, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    R = {"run": {
        "device": device,
        "torch": torch.__version__,
        "cpu": platform.processor(),
        "threads": torch.get_num_threads(),
        "width_mult": WIDTH_MULT,
        "epochs_base": EPOCHS_BASE,
        "epochs_finetune": EPOCHS_FT,
        "lr": LR, "batch_size": BATCH_SIZE,
        "seed": SEED, "single_run_no_seed_averaging": True,
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }}

    # ---------------- data -------------------------------------------------
    train_ds = FeatureDataset(cache("asvspoof_train"))
    stats = train_ds.stats                                   # train-only statistics
    val_ds = FeatureDataset(cache("asvspoof_validation"), stats)
    test_ds = FeatureDataset(cache("asvspoof_test"), stats)
    guj_ds = FeatureDataset(cache("indic_gujarati"), stats)
    hin_ds = FeatureDataset(cache("indic_hindi"), stats)

    R["data"] = {k: v.class_counts() for k, v in
                 (("asvspoof_train", train_ds), ("asvspoof_val", val_ds),
                  ("asvspoof_test", test_ds), ("gujarati", guj_ds), ("hindi", hin_ds))}

    # An EER needs both classes present in usable numbers. A badly skewed split
    # still yields a number, but one whose FRR resolution is 1/n_real -- which is
    # how a 66-real test set produced a misleadingly smooth-looking EER earlier.
    # Fail loudly rather than reporting it.
    for name, counts in R["data"].items():
        if not isinstance(counts, dict) or "real" not in counts:
            continue
        lo = min(counts["real"], counts["fake"])
        if lo == 0:
            raise SystemExit(f"{name} has a class with 0 examples: {counts}")
        if lo < 50:
            raise SystemExit(
                f"{name} minority class is only {lo} clips ({counts}); EER "
                f"resolution would be 1/{lo} = {100.0 / lo:.1f} pp. Re-fetch it.")
        if lo / max(counts["real"], counts["fake"]) < 0.3:
            print(f"  WARNING: {name} is skewed ({counts}); EER is still "
                  f"prior-independent but its resolution is coarse.", flush=True)
    R["data"]["normalisation"] = {k: {"mean": round(m, 4), "std": round(s, 4)}
                                  for k, (m, s) in stats.items()}
    print("data:", json.dumps(R["data"], indent=2), flush=True)

    model = TripleBranchResNeXt(width_mult=WIDTH_MULT)
    R["run"]["parameters"] = model.n_params()

    # ---------------- Phase 2 ---------------------------------------------
    print("\n" + "=" * 72 + "\nPHASE 2 - base training on ASVspoof 2019 LA\n" + "=" * 72,
          flush=True)
    base_ckpt = os.path.join(CKPT, "base_asvspoof.pt")
    log_csv = os.path.join(RESULTS, "training_log.csv")

    # Resume path: reuse an existing base checkpoint instead of retraining.
    # Set REUSE_BASE=1 only when checkpoints/base_asvspoof.pt came from a completed
    # run whose per-epoch log is still on disk -- the reported training metadata is
    # then read back from that CSV, never re-invented.
    reuse = os.environ.get("REUSE_BASE") == "1" and os.path.exists(base_ckpt)
    if reuse:
        import csv as _csv
        with open(log_csv, encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        R["phase2_training"] = {
            "epochs": len(rows),
            "best_val_eer": min(float(r["val_eer_percent"]) for r in rows) / 100.0,
            "wall_clock_minutes": round(
                sum(float(r["epoch_seconds"]) for r in rows) / 60.0, 2),
            "checkpoint": base_ckpt,
            "reused_from_earlier_run": True,
            "note": "Base training was completed in an earlier invocation; these "
                    "figures are read back from results/training_log.csv, which that "
                    "run wrote.",
        }
        print(f"  REUSING base checkpoint ({len(rows)} epochs already trained, "
              f"best val EER {100 * R['phase2_training']['best_val_eer']:.2f}%)",
              flush=True)
    else:
        hist = train_model(model, train_ds, val_ds, EPOCHS_BASE, base_ckpt,
                           log_csv, device=device, tag="phase2")
        R["phase2_training"] = {k: v for k, v in hist.items() if k != "history"}
    model, ck = load_checkpoint(model, base_ckpt, device)
    print(f"  loaded best checkpoint from epoch {ck['epoch']}", flush=True)

    eng = evaluate(model, make_loader(test_ds), device)
    R["phase2"] = {"test_set": "ASVspoof 2019 LA (English, held-out subsample)", **eng}
    print(f"  PHASE 2 English baseline EER = {eng['eer_percent']:.2f}%", flush=True)

    # ---------------- Phase 3 ---------------------------------------------
    print("\n" + "=" * 72 + "\nPHASE 3 - zero-shot cross-lingual evaluation\n" + "=" * 72,
          flush=True)
    base_eer = eng["eer_percent"]
    R["phase3"] = {}
    for lang, ds in (("Gujarati", guj_ds), ("Hindi", hin_ds)):
        m = evaluate(model, make_loader(ds), device)
        abs_inc = m["eer_percent"] - base_eer
        rel_inc = (abs_inc / base_eer * 100.0) if base_eer > 0 else float("nan")
        R["phase3"][lang] = {**m,
                             "english_baseline_eer_percent": round(base_eer, 3),
                             "eer_increase_absolute_pp": round(abs_inc, 3),
                             "eer_increase_relative_percent": round(rel_inc, 2)}
        print(f"  {lang}: EER {m['eer_percent']:.2f}%  |  vs English baseline "
              f"{base_eer:.2f}%  ->  +{abs_inc:.2f} pp  ({rel_inc:+.1f}% relative)",
              flush=True)

    # ---------------- Phase 4 ---------------------------------------------
    print("\n" + "=" * 72 + "\nPHASE 4 - fusion-layer adaptation (10% of Indic data)\n"
          + "=" * 72, flush=True)
    R["phase4"] = {}
    for lang, ds in (("Gujarati", guj_ds), ("Hindi", hin_ds)):
        ft_idx, ho_idx = stratified_split(ds, FT_FRACTION)
        ft_ds = torch.utils.data.Subset(ds, ft_idx)
        ho_ds = torch.utils.data.Subset(ds, ho_idx)
        print(f"\n  {lang}: fine-tune on {len(ft_ds)} clips, hold out {len(ho_ds)}",
              flush=True)

        ft_model = TripleBranchResNeXt(width_mult=WIDTH_MULT)
        load_checkpoint(ft_model, base_ckpt, device)
        ft_model.freeze_branches()                       # branches frozen; MLP trainable

        ck_path = os.path.join(CKPT, f"finetuned_{lang.lower()}.pt")
        h = train_model(ft_model, ft_ds, ho_ds, EPOCHS_FT, ck_path,
                        os.path.join(RESULTS, f"training_log_finetune_{lang.lower()}.csv"),
                        device=device, params=ft_model.fusion_parameters(),
                        tag=f"phase4-{lang}")
        ft_model, _ = load_checkpoint(ft_model, ck_path, device)
        m = evaluate(ft_model, make_loader(ho_ds), device)

        zs = R["phase3"][lang]["eer_percent"]
        recovered = zs - m["eer_percent"]
        gap = zs - base_eer
        R["phase4"][lang] = {
            **m,
            "n_finetune": len(ft_ds), "n_heldout": len(ho_ds),
            "zero_shot_eer_percent": round(zs, 3),
            "eer_recovered_pp": round(recovered, 3),
            "gap_closed_percent": round(100.0 * recovered / gap, 2) if gap else None,
            "training": {k: v for k, v in h.items() if k != "history"},
        }
        print(f"  {lang}: adapted EER {m['eer_percent']:.2f}% "
              f"(zero-shot was {zs:.2f}%)  ->  recovered {recovered:+.2f} pp", flush=True)

    # ---------------- write out -------------------------------------------
    R["run"]["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    R["min_tdcf_note"] = ("Not computed. [11] Eq. (26), PDF p. 8, defines min t-DCF "
                          "over the combined ASV + CM system and requires the miss and "
                          "false-alarm rates of a fixed ASV back-end on the same trials. "
                          "No ASV back-end is available here and the ASVspoof mirror "
                          "ships no ASV scores, so any number computed from CM scores "
                          "alone would not be comparable to published t-DCF values.")

    with open(os.path.join(RESULTS, "table_v_results.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)

    rows = [
        ("Phase 2 - Base training", "ASVspoof 2019 LA (English, held-out)",
         R["phase2"]["eer_percent"], ""),
        ("Phase 3 - Zero-shot", "Gujarati (IndicSynth + FLEURS)",
         R["phase3"]["Gujarati"]["eer_percent"],
         f"+{R['phase3']['Gujarati']['eer_increase_absolute_pp']:.2f} pp vs English"),
        ("Phase 3 - Zero-shot", "Hindi (IndicSynth + FLEURS)",
         R["phase3"]["Hindi"]["eer_percent"],
         f"+{R['phase3']['Hindi']['eer_increase_absolute_pp']:.2f} pp vs English"),
        ("Phase 4 - Target-adapted", "Gujarati (10% fine-tune)",
         R["phase4"]["Gujarati"]["eer_percent"],
         f"{R['phase4']['Gujarati']['eer_recovered_pp']:+.2f} pp vs zero-shot"),
        ("Phase 4 - Target-adapted", "Hindi (10% fine-tune)",
         R["phase4"]["Hindi"]["eer_percent"],
         f"{R['phase4']['Hindi']['eer_recovered_pp']:+.2f} pp vs zero-shot"),
    ]
    md = ["# Table V - Classifier Evaluation Results (real numbers from this run)", "",
          f"Device: {device} | width_mult {WIDTH_MULT} | {EPOCHS_BASE} base epochs | "
          f"single run, no seed averaging", "",
          "| Training Phase | Test Set | EER (%) | min t-DCF | Change |",
          "|---|---|---|---|---|"]
    for phase, test, eer, note in rows:
        md.append(f"| {phase} | {test} | {eer:.2f} | not computed | {note} |")
    md += ["", "**min t-DCF**: " + R["min_tdcf_note"], "",
           "**Scale caveat**: trained on a subsampled ASVspoof 2019 LA "
           f"({R['data']['asvspoof_train']['real'] + R['data']['asvspoof_train']['fake']} "
           "clips) for " f"{EPOCHS_BASE} epochs on CPU, against [11]'s 25,380 clips for "
           "100 epochs on an RTX 3090. These are this run's numbers, not a reproduction "
           "of [11]."]
    with open(os.path.join(RESULTS, "table_v_results.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(md) + "\n")

    print("\n" + "\n".join(md), flush=True)
    print(f"\nwrote {RESULTS}/table_v_results.json and .md", flush=True)


if __name__ == "__main__":
    main()
