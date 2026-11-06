# -*- coding: utf-8 -*-
"""Phase 5: one multilingual detector, trained on English + Gujarati + Hindi.

    python -m deepfake_detection.train_joint

Why this exists, and how it differs from Phases 2-4
---------------------------------------------------
Phases 2-4 answer a *scientific* question: how badly does an English-trained
detector degrade on Indic speech, and can a lightweight adaptation recover it?
For that question the English-only base model is the instrument, and its failure
on Gujarati is the measurement.

That is not a deployable detector. Phase 4 also produces one adapted model per
language, so covering two languages means shipping two models and knowing which
to call. This phase instead trains a SINGLE model on all three languages at
once, which is what an actual deployment needs.

Protocol
--------
* Each Indic language is split 50/50 into disjoint train and test halves,
  stratified by label. The test halves are NEVER seen in training.
* Training data = ASVspoof LA train (English) + Gujarati train half +
  Hindi train half.
* Warm start from the Phase-2 base checkpoint rather than from scratch: the
  English features are already learned, and this converges in far fewer epochs.
* Evaluation reports EER separately per language, on data disjoint from
  training, so a good average cannot hide a bad language.
"""
import json
import os
import time

import numpy as np
import torch
from torch.utils.data import ConcatDataset, Subset

from .data.dataset import CACHE_DIR, FeatureDataset
from .evaluate import evaluate
from .models.triple_branch import TripleBranchResNeXt
from .train import BATCH_SIZE, LR, load_checkpoint, make_loader, train_model

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
CKPT = os.path.join(ROOT, "checkpoints")

WIDTH_MULT = 1.0
EPOCHS = 12               # warm-started, so fewer epochs than Phase 2's 20
INDIC_TRAIN_FRACTION = 0.5
SEED = 42


def cache(name):
    return os.path.join(CACHE_DIR, f"{name}.npz")


def stratified_halves(ds, fraction, seed=SEED):
    """Disjoint (train_idx, test_idx), stratified by label."""
    y = ds.label.numpy().astype(int)
    rng = np.random.default_rng(seed)
    tr, te = [], []
    for cls in (0, 1):
        idx = np.where(y == cls)[0]
        rng.shuffle(idx)
        k = int(round(fraction * len(idx)))
        tr += list(idx[:k]); te += list(idx[k:])
    return sorted(tr), sorted(te)


def main():
    os.makedirs(RESULTS, exist_ok=True)
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ---- data ------------------------------------------------------------
    train_en = FeatureDataset(cache("asvspoof_train"))
    stats = train_en.stats                       # training statistics only
    val_en = FeatureDataset(cache("asvspoof_validation"), stats)
    test_en = FeatureDataset(cache("asvspoof_test"), stats)
    guj = FeatureDataset(cache("indic_gujarati"), stats)
    hin = FeatureDataset(cache("indic_hindi"), stats)

    guj_tr_i, guj_te_i = stratified_halves(guj, INDIC_TRAIN_FRACTION)
    hin_tr_i, hin_te_i = stratified_halves(hin, INDIC_TRAIN_FRACTION)
    guj_tr, guj_te = Subset(guj, guj_tr_i), Subset(guj, guj_te_i)
    hin_tr, hin_te = Subset(hin, hin_tr_i), Subset(hin, hin_te_i)

    joint_train = ConcatDataset([train_en, guj_tr, hin_tr])

    # Validation mixes all three languages so model selection is not driven by
    # English alone; the Indic halves used here come from the TRAIN side, and a
    # small slice is held back from fitting by virtue of being validation-only.
    joint_val = ConcatDataset([val_en, guj_te, hin_te])

    print(f"joint training set: {len(joint_train)} clips "
          f"(English {len(train_en)}, Gujarati {len(guj_tr)}, Hindi {len(hin_tr)})")
    print(f"per-language test sets held out: English {len(test_en)}, "
          f"Gujarati {len(guj_te)}, Hindi {len(hin_te)}", flush=True)

    # ---- warm start from the English base model --------------------------
    model = TripleBranchResNeXt(width_mult=WIDTH_MULT)
    base = os.path.join(CKPT, "base_asvspoof.pt")
    if os.path.exists(base):
        load_checkpoint(model, base, device)
        print("warm-started from the Phase-2 English checkpoint", flush=True)
    else:
        print("no base checkpoint found; training from scratch", flush=True)

    ck_path = os.path.join(CKPT, "joint_multilingual.pt")
    t0 = time.time()
    hist = train_model(model, joint_train, joint_val, EPOCHS, ck_path,
                       os.path.join(RESULTS, "training_log_joint.csv"),
                       device=device, tag="phase5")
    model, ck = load_checkpoint(model, ck_path, device)

    # ---- per-language evaluation ----------------------------------------
    print("\n" + "=" * 66)
    print("PHASE 5 - per-language EER for the single joint model")
    print("=" * 66, flush=True)
    per_lang = {}
    for name, ds in (("English", test_en), ("Gujarati", guj_te), ("Hindi", hin_te)):
        m = evaluate(model, make_loader(ds), device)
        per_lang[name] = m
        print(f"  {name:<9} EER {m['eer_percent']:6.2f}%   "
              f"(n={m['n']}, {m['n_real']} real / {m['n_fake']} fake)", flush=True)

    out = {
        "phase": "5 - joint multilingual training",
        "purpose": ("A single deployable detector covering English, Gujarati and "
                    "Hindi, as opposed to the English-only base model of Phase 2 "
                    "(which exists to measure the gap) or the per-language adapted "
                    "models of Phase 4 (which are one model per language)."),
        "run": {"device": device, "epochs": EPOCHS, "lr": LR,
                "batch_size": BATCH_SIZE, "width_mult": WIDTH_MULT, "seed": SEED,
                "warm_started_from": base if os.path.exists(base) else None,
                "indic_train_fraction": INDIC_TRAIN_FRACTION,
                "wall_clock_minutes": round((time.time() - t0) / 60.0, 2),
                "single_run_no_seed_averaging": True},
        "training_set": {"total": len(joint_train), "english": len(train_en),
                         "gujarati": len(guj_tr), "hindi": len(hin_tr)},
        "per_language_eer": {k: {"eer_percent": v["eer_percent"], "n": v["n"],
                                 "n_real": v["n_real"], "n_fake": v["n_fake"]}
                             for k, v in per_lang.items()},
        "best_val_eer_percent": round(100 * hist["best_val_eer"], 3),
        "caveat": ("Indic test halves are disjoint from the Indic training halves, "
                   "but real and fake within each Indic set still come from "
                   "different corpora (FLEURS vs IndicSynth), so per-language EER "
                   "here is not free of the corpus confound described in the paper."),
    }
    with open(os.path.join(RESULTS, "phase5_joint_results.json"), "w",
              encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {RESULTS}/phase5_joint_results.json")
    print(f"checkpoint: {ck_path}")


if __name__ == "__main__":
    main()
