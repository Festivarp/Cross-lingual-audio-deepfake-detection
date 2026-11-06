# -*- coding: utf-8 -*-
"""Phases 3-5 on the speaker-matched Indic sets, with speaker-disjoint splits.

    python -m deepfake_detection.run_kathbath_phases            # Phase 3 + 4 (+5)
    python -m deepfake_detection.run_kathbath_phases --no-phase5
    python -m deepfake_detection.run_kathbath_phases --phase5-only   # add Phase 5 to an existing result

What changes relative to run_phases.py / train_joint.py
-------------------------------------------------------
1. Data: `indic_<lang>_kb` -- IndicSynth fakes + Kathbath real speech from the
   same speakers (data/indic_kathbath.py), instead of FLEURS real speech.
2. Splits are by SPEAKER, not by clip. Every speaker contributes equally many
   real and fake clips by construction, so a speaker-level split is also
   class-balanced. Without this, the same voice would appear on both sides of
   the split and adaptation could succeed by memorising speakers.
3. Model selection never sees the test clips. The original Phase 4 selected its
   checkpoint on the held-out set it then reported, and the original Phase 5
   validated on the Indic test halves. Here each language is split into
   adapt/train, validation and test speaker groups, and checkpoints are chosen
   on validation only.

Phase 3 (zero-shot) is reported on each test group, so "recovered" compares
zero-shot and adapted EER on exactly the same clips.

Phase 4 is cheap (fusion layer only), so it is repeated over 3 split seeds and
reported as mean and range. Phase 5 (joint training, ~45 min on CPU) is one run.

Writes results/kathbath_phases.json and checkpoints/*_kb*.pt.
"""
import argparse
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

EPOCHS_FT = 15            # as run_phases.py
EPOCHS_JOINT = 12         # as train_joint.py
FT_FRACTION = 0.10        # "10% of the Indic datasets", per the roadmap
VAL_FRACTION = 0.10       # speaker-disjoint validation group for model selection
JOINT_TRAIN_FRACTION = 0.5
PHASE4_SEEDS = (0, 1, 2)
JOINT_SEED = 42
LANGS = ("gujarati", "hindi")
SUFFIX = ""               # set from --suffix


def cache(name):
    return os.path.join(CACHE_DIR, f"{name}.npz")


def speaker_groups(path, fractions, seed):
    """Partition clip indices into len(fractions)+1 speaker-disjoint groups.

    Speakers are shuffled and assigned greedily until each group holds roughly
    its fraction of the clips; whatever remains is the last group (test).
    """
    spk = np.load(path)["speaker"]
    speakers = np.unique(spk)
    rng = np.random.default_rng(seed)
    rng.shuffle(speakers)
    n = len(spk)
    groups, i = [], 0
    for frac in fractions:
        target, members = frac * n, []
        while i < len(speakers) and (not members or
                                     sum((spk == s).sum() for s in members) < target):
            members.append(speakers[i]); i += 1
        groups.append(members)
    groups.append(list(speakers[i:]))
    return [sorted(np.where(np.isin(spk, g))[0].tolist()) for g in groups], \
           [len(g) for g in groups]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-phase5", action="store_true")
    ap.add_argument("--suffix", default="",
                    help="cache/result variant, e.g. _lp6k for the 6 kHz band-limited control")
    ap.add_argument("--phase5-only", action="store_true",
                    help="skip Phase 4; merge Phase 5 into the existing results file")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_en = FeatureDataset(cache("asvspoof_train"))
    stats = train_en.stats
    val_en = FeatureDataset(cache("asvspoof_validation"), stats)
    test_en = FeatureDataset(cache("asvspoof_test"), stats)
    global SUFFIX
    SUFFIX = args.suffix
    sets = {l: FeatureDataset(cache(f"indic_{l}_kb{SUFFIX}"), stats) for l in LANGS}
    base_ckpt = os.path.join(CKPT, "base_asvspoof.pt")

    base = TripleBranchResNeXt()
    load_checkpoint(base, base_ckpt, device)
    base.eval()
    en_eer = evaluate(base, make_loader(test_en), device)["eer_percent"]

    R = {"data": "speaker-matched: IndicSynth fake + Kathbath real, same speakers",
         "english_base_eer_percent": round(en_eer, 3),
         "phase3_zero_shot_full_set": {}, "phase4": {}}
    for l in LANGS:
        R["phase3_zero_shot_full_set"][l] = round(
            evaluate(base, make_loader(sets[l]), device)["eer_percent"], 3)
    print(f"English base {en_eer:.2f}% | zero-shot full sets: "
          f"{R['phase3_zero_shot_full_set']}", flush=True)

    if args.phase5_only:
        prev = os.path.join(RESULTS, f"kathbath_phases{SUFFIX}.json")
        if os.path.exists(prev):
            with open(prev, encoding="utf-8") as fh:
                R = {**json.load(fh), **{k: R[k] for k in ("english_base_eer_percent",
                                                            "phase3_zero_shot_full_set")}}
    # ---------------- Phase 4: fusion-only adaptation, 3 split seeds -----------
    print("\n" + "=" * 72 + "\nPHASE 4 - fusion-layer adaptation, speaker-disjoint\n"
          + "=" * 72, flush=True)
    for l in (() if args.phase5_only else LANGS):
        runs = []
        for sd in PHASE4_SEEDS:
            (ft_i, va_i, te_i), n_spk = speaker_groups(cache(f"indic_{l}_kb{SUFFIX}"),
                                                       (FT_FRACTION, VAL_FRACTION), sd)
            ds = sets[l]
            ft, va, te = Subset(ds, ft_i), Subset(ds, va_i), Subset(ds, te_i)
            zs = evaluate(base, make_loader(te), device)["eer_percent"]

            m = TripleBranchResNeXt()
            load_checkpoint(m, base_ckpt, device)
            m.freeze_branches()
            ck = os.path.join(CKPT, f"finetuned_{l}_kb{SUFFIX}_seed{sd}.pt")
            torch.manual_seed(sd); np.random.seed(sd)
            train_model(m, ft, va, EPOCHS_FT, ck,
                        os.path.join(RESULTS, "logs", f"phase4_{l}_kb{SUFFIX}_seed{sd}.csv"),
                        device=device, params=m.fusion_parameters(),
                        tag=f"phase4-{l}-kb-s{sd}")
            m, _ = load_checkpoint(m, ck, device)
            ad = evaluate(m, make_loader(te), device)["eer_percent"]
            run = {"seed": sd, "speakers": {"adapt": n_spk[0], "val": n_spk[1], "test": n_spk[2]},
                   "clips": {"adapt": len(ft), "val": len(va), "test": len(te)},
                   "zero_shot_eer_percent": round(zs, 3),
                   "adapted_eer_percent": round(ad, 3),
                   "recovered_pp": round(zs - ad, 3)}
            runs.append(run)
            print(f"  {l} seed {sd}: adapt {len(ft)} clips/{n_spk[0]} spk, test {len(te)} "
                  f"clips/{n_spk[2]} spk | zero-shot {zs:.2f}% -> adapted {ad:.2f}%",
                  flush=True)
        agg = {k: {"mean": round(float(np.mean([r[k] for r in runs])), 3),
                   "min": round(float(min(r[k] for r in runs)), 3),
                   "max": round(float(max(r[k] for r in runs)), 3)}
               for k in ("zero_shot_eer_percent", "adapted_eer_percent", "recovered_pp")}
        R["phase4"][l] = {"runs": runs, "aggregate": agg}
        print(f"  -> {l}: zero-shot {agg['zero_shot_eer_percent']['mean']:.2f}%  "
              f"adapted {agg['adapted_eer_percent']['mean']:.2f}%  "
              f"(range {agg['adapted_eer_percent']['min']:.2f}-"
              f"{agg['adapted_eer_percent']['max']:.2f})", flush=True)
    _write(R)

    # ---------------- Phase 5: joint training, one run --------------------------
    if args.no_phase5:
        return
    print("\n" + "=" * 72 + "\nPHASE 5 - joint multilingual training, speaker-disjoint\n"
          + "=" * 72, flush=True)
    tr_parts, va_parts, tests, info = [train_en], [val_en], {"english": test_en}, {}
    for l in LANGS:
        (tr_i, va_i, te_i), n_spk = speaker_groups(
            cache(f"indic_{l}_kb{SUFFIX}"), (JOINT_TRAIN_FRACTION, VAL_FRACTION), JOINT_SEED)
        tr_parts.append(Subset(sets[l], tr_i))
        va_parts.append(Subset(sets[l], va_i))
        tests[l] = Subset(sets[l], te_i)
        info[l] = {"speakers": {"train": n_spk[0], "val": n_spk[1], "test": n_spk[2]},
                   "clips": {"train": len(tr_i), "val": len(va_i), "test": len(te_i)}}
    m = TripleBranchResNeXt()
    load_checkpoint(m, base_ckpt, device)
    torch.manual_seed(JOINT_SEED); np.random.seed(JOINT_SEED)
    ck = os.path.join(CKPT, f"joint_multilingual_kb{SUFFIX}.pt")
    t0 = time.time()
    hist = train_model(m, ConcatDataset(tr_parts), ConcatDataset(va_parts), EPOCHS_JOINT, ck,
                       os.path.join(RESULTS, "logs", f"phase5_joint_kb{SUFFIX}.csv"),
                       device=device, tag="phase5-kb")
    m, _ = load_checkpoint(m, ck, device)
    per = {}
    for name, ds in tests.items():
        zs = evaluate(base, make_loader(ds), device)["eer_percent"]
        jt = evaluate(m, make_loader(ds), device)["eer_percent"]
        per[name] = {"zero_shot_or_base_eer_percent": round(zs, 3),
                     "joint_eer_percent": round(jt, 3), "n": len(ds)}
        print(f"  {name:<9} base {zs:6.2f}%  ->  joint {jt:6.2f}%  (n={len(ds)})", flush=True)
    R["phase5"] = {"run": {"epochs": EPOCHS_JOINT, "lr": LR, "batch_size": BATCH_SIZE,
                           "seed": JOINT_SEED, "warm_started_from": "base_asvspoof.pt",
                           "wall_clock_minutes": round((time.time() - t0) / 60.0, 2),
                           "single_run": True},
                   "splits": info, "per_language": per,
                   "best_val_eer_percent": round(100 * hist["best_val_eer"], 3)}
    _write(R)


def _write(R):
    os.makedirs(os.path.join(RESULTS, "logs"), exist_ok=True)
    with open(os.path.join(RESULTS, f"kathbath_phases{SUFFIX}.json"), "w", encoding="utf-8") as fh:
        json.dump(R, fh, indent=2)
    print(f"wrote {RESULTS}/kathbath_phases{SUFFIX}.json", flush=True)


if __name__ == "__main__":
    main()
