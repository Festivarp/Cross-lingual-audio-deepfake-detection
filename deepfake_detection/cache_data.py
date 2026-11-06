# -*- coding: utf-8 -*-
"""Phase 1: fetch a subsample of every corpus and cache its features.

Run from d:\\RMS:   python -m deepfake_detection.cache_data

Subsample sizes and why they are not the full protocol
------------------------------------------------------
Tahaoglu et al. [11] train on the full ASVspoof 2019 LA train partition (25,380
utterances) for 100 epochs on an NVIDIA RTX 3090 ([11], PDF p. 8). This project
has no GPU (PyTorch CPU build, 8 threads) and the ASVspoof mirror streams at a
measured ~7 rows/s, so the full protocol is out of reach here: the training data
alone would take about an hour to fetch and far longer to train on for 100
epochs.

The sizes below are therefore a deliberate, reported reduction. Every number
this pipeline produces is a number from THIS reduced run, and must not be read
as a reproduction of [11]'s protocol.

Class balance
-------------
ASVspoof LA is heavily imbalanced (train is 2,580 bonafide vs 22,800 spoof, a
1:8.8 ratio -- [2], PDF p. 6). We sample a BALANCED subset rather than applying
class weighting to an imbalanced one. Justification: EER is a threshold-free,
prior-independent metric, so a balanced sample does not bias it, and balanced
batches train more stably at this small scale. The true corpus ratio is recorded
here and in the README so the choice is visible.
"""
import argparse
import json
import os
import sys
import time

from .data import asvspoof, indic, indic_kathbath
from .data.dataset import CACHE_DIR, build_cache

# --- subsample sizes (per class) -------------------------------------------
N_TRAIN_PER_CLASS = 1200      # 2,400 total, vs [11]'s 25,380
N_VAL_PER_CLASS = 300         # 600 total
N_TEST_PER_CLASS = 500        # 1,000 total
N_INDIC_PER_CLASS = 300       # per language: 300 real + 300 fake

MANIFEST = os.path.join(CACHE_DIR, "manifest.json")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--indic-source", choices=("fleurs", "kathbath"), default="fleurs",
                    help="real-speech source for the Indic sets")
    args = ap.parse_args()

    os.makedirs(CACHE_DIR, exist_ok=True)
    t0 = time.time()
    if args.indic_source == "kathbath":
        return cache_kathbath(t0)
    manifest = {"subsample_sizes": {
        "asvspoof_train_per_class": N_TRAIN_PER_CLASS,
        "asvspoof_val_per_class": N_VAL_PER_CLASS,
        "asvspoof_test_per_class": N_TEST_PER_CLASS,
        "indic_per_class": N_INDIC_PER_CLASS},
        "full_protocol_for_reference": asvspoof.PROTOCOL,
        "balanced_sampling": True,
        "class_weighting_applied": False}

    # ---- ASVspoof 2019 LA -------------------------------------------------
    for split, n in (("train", N_TRAIN_PER_CLASS),
                     ("validation", N_VAL_PER_CLASS),
                     ("test", N_TEST_PER_CLASS)):
        name = f"asvspoof_{split}"
        if os.path.exists(os.path.join(CACHE_DIR, name + ".npz")):
            print(f"[skip] {name} already cached", flush=True)
            continue
        print(f"\n=== ASVspoof {split}: {n} bonafide + {n} spoof ===", flush=True)
        recs = asvspoof.load_subset(split, n, n)
        manifest[name] = {"requested_per_class": n, "obtained": len(recs),
                          "real": sum(1 for r in recs if r["label"] == 0),
                          "fake": sum(1 for r in recs if r["label"] == 1)}
        build_cache(recs, name)
        del recs

    # ---- Indic target domain ---------------------------------------------
    for language in ("Gujarati", "Hindi"):
        name = f"indic_{language.lower()}"
        if os.path.exists(os.path.join(CACHE_DIR, name + ".npz")):
            print(f"[skip] {name} already cached", flush=True)
            continue
        print(f"\n=== {language}: {N_INDIC_PER_CLASS} real + "
              f"{N_INDIC_PER_CLASS} fake ===", flush=True)
        recs = indic.load_language(language, N_INDIC_PER_CLASS, N_INDIC_PER_CLASS)
        manifest[name] = {"requested_per_class": N_INDIC_PER_CLASS,
                          "obtained": len(recs),
                          "real": sum(1 for r in recs if r["label"] == 0),
                          "fake": sum(1 for r in recs if r["label"] == 1),
                          "real_source": f"FLEURS {indic.LANGS[language]['fleurs']} test",
                          "fake_source": f"IndicSynth {language}"}
        build_cache(recs, name)
        del recs

    manifest["elapsed_minutes"] = round((time.time() - t0) / 60.0, 2)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nPhase 1 complete in {manifest['elapsed_minutes']:.1f} min")
    print(f"manifest -> {MANIFEST}")


def cache_kathbath(t0):
    """Speaker-matched Indic sets; ASVspoof caches are untouched."""
    manifest = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST, encoding="utf-8") as fh:
            manifest = json.load(fh)
    for language in ("Gujarati", "Hindi"):
        name = f"indic_{language.lower()}_kb"
        if os.path.exists(os.path.join(CACHE_DIR, name + ".npz")):
            print(f"[skip] {name} already cached", flush=True)
            continue
        print(f"\n=== {language} (speaker-matched): {N_INDIC_PER_CLASS} real + "
              f"{N_INDIC_PER_CLASS} fake ===", flush=True)
        recs, p = indic_kathbath.load_language(language, N_INDIC_PER_CLASS)
        manifest[name] = {"requested_per_class": N_INDIC_PER_CLASS,
                          "obtained": len(recs),
                          "real": sum(1 for r in recs if r["label"] == 0),
                          "fake": sum(1 for r in recs if r["label"] == 1),
                          "speakers": p["speakers"],
                          "fake_models": p["fake_models"],
                          "real_source": f"Kathbath {language.lower()} (same speakers as the fakes)",
                          "fake_source": f"IndicSynth {language}",
                          "selection": f"cache/raw_audio/{language.lower()}/selection.csv"}
        build_cache(recs, name)
        del recs
    manifest["kathbath_elapsed_minutes"] = round((time.time() - t0) / 60.0, 2)
    with open(MANIFEST, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2)
    print(f"\nKathbath Indic sets done in {manifest['kathbath_elapsed_minutes']:.1f} min")


if __name__ == "__main__":
    sys.exit(main())
