# -*- coding: utf-8 -*-
"""Live detector demo -- runs in seconds, no network needed.

Loads the trained Triple-Branch ResNeXt and scores audio clips as real or fake.

    python demo.py                      # score the 20 matched-pair demo clips
    python demo.py path/to/clip.wav     # score any single file
    python demo.py --model gujarati     # use the Gujarati-adapted checkpoint

What to watch for: the base model is trained on ENGLISH (ASVspoof 2019 LA). It
does well on English and inverts on Gujarati -- which is the finding this whole
project is about, visible directly in the per-clip verdicts.
"""
import argparse
import glob
import os
import sys
import time

import torch

from deepfake_detection.data.dataset import CACHE_DIR, FeatureDataset
from deepfake_detection.features.extract import extract_features
from deepfake_detection.models.triple_branch import TripleBranchResNeXt

ROOT = os.path.dirname(os.path.abspath(__file__))
CKPTS = {
    "base": "checkpoints/base_asvspoof.pt",          # English-only: the failure case
    "gujarati": "checkpoints/finetuned_gujarati.pt",  # fusion adapted to Gujarati
    "hindi": "checkpoints/finetuned_hindi.pt",        # fusion adapted to Hindi
    "joint": "checkpoints/joint_multilingual.pt",     # one model, all three languages
}


def load_model(which="base", device="cpu"):
    path = os.path.join(ROOT, CKPTS[which])
    if not os.path.exists(path):
        sys.exit(f"missing checkpoint {path} -- run the pipeline first")
    model = TripleBranchResNeXt(width_mult=1.0)
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    model.eval()
    return model, ck


def train_stats():
    """Normalisation statistics from the TRAINING cache, as used in training."""
    p = os.path.join(CACHE_DIR, "asvspoof_train.npz")
    if not os.path.exists(p):
        sys.exit("missing cache/asvspoof_train.npz -- needed for input normalisation")
    return FeatureDataset(p).stats


@torch.no_grad()
def score(model, wav_path, stats):
    """Returns (p_fake, verdict). Higher p_fake = more likely synthetic."""
    mfcc, lfcc, cqcc = extract_features(wav_path)
    tens = []
    for name, arr in (("mfcc", mfcc), ("lfcc", lfcc), ("cqcc", cqcc)):
        m, s = stats[name]
        t = (torch.from_numpy(arr) - m) / (s if s else 1.0)
        tens.append(t.unsqueeze(0).unsqueeze(0))          # (1, 1, 13, 398)
    logit = model(*tens)
    p_fake = torch.sigmoid(logit).item()
    return p_fake, ("FAKE" if p_fake >= 0.5 else "REAL")


CASCADE_CKPT = "checkpoints/ablation/e2_{fam}_seed0.pt"


def load_cascade(order, device="cpu"):
    """One SingleBranchDetector per family, in cascade order."""
    from deepfake_detection.models.single_branch import SingleBranchDetector
    models = []
    for fam in order:
        path = os.path.join(ROOT, CASCADE_CKPT.format(fam=fam))
        if not os.path.exists(path):
            sys.exit(f"missing {path} -- run `python -m deepfake_detection.run_ablation --e2`")
        m = SingleBranchDetector(fam)
        m.load_state_dict(torch.load(path, map_location=device,
                                     weights_only=False)["model"])
        models.append(m.eval())
    return models


@torch.no_grad()
def score_cascade(models, order, wav_path, stats, tau):
    """Lazily extract, stage by stage, stopping as soon as a stage is confident.

    Returns (p_fake, verdict, stage_reached, elapsed_ms). Features are extracted
    one family at a time -- extracting all three up front would spend the CQCC
    cost this whole method exists to avoid.
    """
    from deepfake_detection.features.extract import extract_one
    t0 = time.perf_counter()
    probs = []
    for stage, (fam, model) in enumerate(zip(order, models)):
        arr = extract_one(fam, wav_path)
        m, s = stats[fam]
        t = ((torch.from_numpy(arr) - m) / (s if s else 1.0)).unsqueeze(0).unsqueeze(0)
        p = torch.sigmoid(model(t, t, t)).item()      # detector uses only its own
        probs.append(p)
        if stage < len(order) - 1 and abs(p - 0.5) >= tau:
            ms = (time.perf_counter() - t0) * 1000.0
            return p, ("FAKE" if p >= 0.5 else "REAL"), stage + 1, ms
    p = sum(probs) / len(probs)                        # stage 3: score fusion
    ms = (time.perf_counter() - t0) * 1000.0
    return p, ("FAKE" if p >= 0.5 else "REAL"), len(order), ms


@torch.no_grad()
def score_baseline_timed(model, wav_path, stats):
    """The [11] baseline, timed the same way, for a like-for-like comparison."""
    t0 = time.perf_counter()
    p, verdict = score(model, wav_path, stats)
    return p, verdict, (time.perf_counter() - t0) * 1000.0


def run_cascade_demo(args):
    order = args.order.split(",")
    stats = train_stats()
    models = load_cascade(order)
    baseline, _ = load_model("base", "cpu")

    demo = os.path.join(ROOT, "demo_audio", "analysed_4s")
    files = sorted(glob.glob(os.path.join(demo, "*.wav")))
    if not files:
        sys.exit(f"no demo clips at {demo}")

    print(f"cascade order : {' -> '.join(f.upper() for f in order)}")
    print(f"tau           : {args.tau}")
    print(f"baseline      : [11] triple-branch, all three families always\n")
    print(f"  {'clip':<24}{'truth':<7}{'cascade':<9}{'stage':>6}"
          f"{'ms':>9}{'base ms':>10}{'saved':>9}")
    print("  " + "-" * 74)

    tot_c, tot_b, stages, correct = 0.0, 0.0, [], 0
    for f in files:
        truth = "FAKE" if "_fake_" in os.path.basename(f) else "REAL"
        p, verdict, stage, ms = score_cascade(models, order, f, stats, args.tau)
        _, _, bms = score_baseline_timed(baseline, f, stats)
        tot_c += ms; tot_b += bms; stages.append(stage)
        correct += verdict == truth
        print(f"  {os.path.basename(f):<24}{truth:<7}{verdict:<9}{stage:>6}"
              f"{ms:>9.1f}{bms:>10.1f}{bms - ms:>8.1f}ms")

    n = len(files)
    print("  " + "-" * 74)
    print(f"  cascade mean {tot_c / n:.1f} ms   baseline mean {tot_b / n:.1f} ms"
          f"   speed-up {tot_b / tot_c:.2f}x")
    print(f"  exits: stage1 {stages.count(1)}/{n}, stage2 {stages.count(2)}/{n}, "
          f"stage3 {stages.count(3)}/{n}")
    print(f"  cascade correct {correct}/{n}")
    print("\nNote: these are the 20 matched-pair demo clips, not a held-out test")
    print("set. The EER numbers that support the paper come from")
    print("results/e4_cascade.json; this view exists to show the cost saving live.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", nargs="?", help="a .wav file to score")
    ap.add_argument("--model", default="base", choices=list(CKPTS))
    ap.add_argument("--cascade", action="store_true",
                    help="run the cost-aware cascade and time it against [11]")
    ap.add_argument("--order", default="mfcc,lfcc,cqcc",
                    help="cascade branch order (cheapest first)")
    ap.add_argument("--tau", type=float, default=0.4,
                    help="early-exit confidence threshold")
    args = ap.parse_args()

    if args.cascade:
        run_cascade_demo(args)
        return

    model, ck = load_model(args.model)
    stats = train_stats()
    print(f"model: {args.model}  (checkpoint from epoch {ck.get('epoch')}, "
          f"val EER {100 * ck.get('val_eer', float('nan')):.2f}%)")
    print(f"params: {model.n_params()['total']:,}\n")

    if args.clip:
        p, verdict = score(model, args.clip, stats)
        print(f"{os.path.basename(args.clip)}: {verdict}  (P(fake) = {p:.3f})")
        return

    # No file given: score the matched-pair demo clips, grouped by language.
    demo = os.path.join(ROOT, "demo_audio", "analysed_4s")
    if not os.path.isdir(demo):
        sys.exit(f"no demo clips at {demo}")

    for lang in ("english", "gujarati"):
        files = sorted(glob.glob(os.path.join(demo, f"{lang}_*.wav")))
        if not files:
            continue
        print("=" * 62)
        print(f"{lang.upper()}  (ground truth is in the filename)")
        print("=" * 62)
        print(f"  {'clip':<26}{'truth':<8}{'verdict':<9}{'P(fake)':>9}   ")
        correct = 0
        for f in files:
            truth = "FAKE" if "_fake_" in os.path.basename(f) else "REAL"
            p, verdict = score(model, f, stats)
            ok = verdict == truth
            correct += ok
            print(f"  {os.path.basename(f):<26}{truth:<8}{verdict:<9}{p:>9.3f}"
                  f"   {'ok' if ok else 'MISS'}")
        print(f"  -> {correct}/{len(files)} correct\n")

    if args.model == "base":
        print("Note: this is the ENGLISH-ONLY model. Its behaviour on the Gujarati")
        print("clips is the cross-lingual failure the project measures (Table V:")
        print("15.00% EER on English, 88.33% zero-shot on Gujarati). For a detector")
        print("meant to work on both languages, run:  python demo.py --model joint")
    elif args.model == "joint":
        print("Note: this single model was trained on English + Gujarati + Hindi")
        print("together (Phase 5). The Gujarati fakes here are edge-tts renderings,")
        print("whereas it was trained on IndicSynth fakes -- so this is also a test")
        print("of generalisation to an unseen synthesis engine.")
    else:
        print(f"Note: this is the {args.model}-adapted model -- only the MLP fusion")
        print("layer was retrained, with the three ResNeXt branches frozen.")


if __name__ == "__main__":
    main()
