# Cross-Lingual Audio Deepfake Detection in Indic Languages

**Parv Shah (23BCE219) · Divya Patel (23BCE227)** — Department of Computer Science and Engineering, Nirma University, Ahmedabad

An end-to-end pipeline for quantifying and mitigating the cross-lingual generalization gap in audio deepfake detection, using a Triple-Branch ResNeXt that fuses MFCC, LFCC and CQCC features.

## Repository layout

```
deepfake_detection/
  data/asvspoof.py       ASVspoof 2019 LA loader (selective Parquet row-group reads)
  data/indic.py          IndicSynth (fake) + FLEURS (real), Gujarati and Hindi
  data/dataset.py        feature caching + torch Dataset
  features/extract.py    MFCC / LFCC / CQCC, 16 kHz, 25 ms / 10 ms, 13 coefficients
  models/resnext_branch.py   single ResNeXt branch (grouped convolutions)
  models/triple_branch.py    three branches + MLP fusion head
  train.py               training loop, checkpoints on best validation EER
  evaluate.py            EER; min t-DCF deliberately not computed (see below)
  cache_data.py          Phase 1 driver
  run_phases.py          Phases 2-4 driver
  run_pipeline.ipynb     Colab-runnable notebook wrapping all four phases
cache/                   cached feature tensors (.npz) + manifest.json
checkpoints/             base and fine-tuned model weights
results/                 table_v_results.{json,md}, training logs, figures
mentor_demo_features.ipynb   matched-pair spectral demo (separate study)
paper_draft.md / paper_final.tex / paper_draft_annotated.md
```

## How to run

```bash
pip install -r requirements.txt
python -m deepfake_detection.cache_data    # Phase 1: fetch + cache features
python -m deepfake_detection.run_phases    # Phases 2-4: train, zero-shot, adapt
python tools/make_readme.py                # regenerate this file from the results
```

Or open `deepfake_detection/run_pipeline.ipynb`, which runs all four phases and plots the training curves.

## What each phase does

| Phase | Purpose | Output |
|---|---|---|
| 1 | Fetch a subsample of ASVspoof 2019 LA, IndicSynth and FLEURS; extract and cache MFCC/LFCC/CQCC | `cache/*.npz` |
| 2 | Train the Triple-Branch ResNeXt on English (ASVspoof LA); checkpoint on best validation EER | English baseline EER |
| 3 | Evaluate that checkpoint zero-shot on Gujarati and Hindi | the generalization gap |
| 4 | Fine-tune **only** the MLP fusion layer on 10% of each Indic set | recovered EER |

## Results

| Training phase | Test set | EER (%) | min t-DCF | Change |
|---|---|---|---|---|
| Phase 2 — Base training | ASVspoof 2019 LA (English, held-out) | 15.00 | not computed | — |
| Phase 3 — Zero-shot | Gujarati (IndicSynth + FLEURS) | 88.33 | not computed | +73.33 pp (+488.9% relative) |
| Phase 3 — Zero-shot | Hindi (IndicSynth + FLEURS) | 72.33 | not computed | +57.33 pp (+382.2% relative) |
| Phase 4 — Target-adapted | Gujarati (10% fine-tune) | 41.85 | not computed | +46.48 pp vs zero-shot |
| Phase 4 — Target-adapted | Hindi (10% fine-tune) | 49.26 | not computed | +23.07 pp vs zero-shot |

### The generalization gap

- **Gujarati**: English baseline 15.00% → zero-shot 88.33% (an increase of +73.33 percentage points, +488.9% relative). After adapting the fusion layer on 60 clips: 41.85% (+46.48 pp recovered, 63.4% of the gap closed).
- **Hindi**: English baseline 15.00% → zero-shot 72.33% (an increase of +57.33 percentage points, +382.2% relative). After adapting the fusion layer on 60 clips: 49.26% (+23.07 pp recovered, 40.2% of the gap closed).

## Hyperparameters and their source

| Setting | Value | Source |
|---|---|---|
| Optimizer | Adam | Tahaoglu et al. [11], PDF p. 8 |
| Learning rate | 0.0001 | [11], PDF p. 8 |
| Batch size | 32 | [11], PDF p. 8 |
| Model selection | lowest validation EER | [11], PDF p. 8 |
| Branch channels | 64 → 128 → 256 → 512 → 1024 | [11], Table 1, PDF p. 7 |
| Stem | Conv2d 3×3/64/stride 1 → BN → ReLU → MaxPool 3×3/stride 2 | [11], Table 1, PDF p. 7 |
| Block form | 1×1 → grouped 3×3 → 1×1, each + BN | [11], Table 1, PDF p. 7 |
| Pooling | AdaptiveAvgPool2d → 1024-d embedding | [11], Table 1, PDF p. 7 |
| Fusion | MLP over concatenated embeddings | [11], PDF pp. 1, 2, 14 |
| Cardinality | 32 | **our default** — [11] Table 1's third value is ambiguous between stride and group count |
| MLP head shape | 3072 → 256 → 64 → 1, ReLU + dropout 0.3 | **our default** — [11] does not give layer sizes |
| Loss | BCEWithLogits on one logit | **our default** — [11] does not name a loss |
| Epochs (base) | 20 | **deviation** — [11] uses 100 on an RTX 3090; this is a CPU run |
| Epochs (fine-tune) | 15 | our choice |
| Input shape | (1, 13, 398) | **deviation** — [11] uses (1, 60, 750); 13 coefficients is fixed by this project's feature front-end |
| Clip length | 4.0 s (64,000 samples @ 16 kHz) | our choice, consistent across every notebook in this project |

Model size: **6,356,033** parameters (5,552,832 in the three branches, 803,201 in the fusion head — the fusion head is what Phase 4 adapts).

## Compute

- Device: **cpu** (Intel64 Family 6 Model 151 Stepping 2, GenuineIntel, 8 torch threads)
- Phase 2 base training: 20 epochs, **47.55 minutes** wall clock
- Phase 4 Gujarati: 15 epochs, 2.45 minutes
- Phase 4 Hindi: 15 epochs, 4.74 minutes
- Phase 1 data fetch + feature extraction: 98.45 minutes

## Data actually used

| Cache | Real | Fake | Source |
|---|---|---|---|
| `asvspoof_train` | 1200 | 1200 | ASVspoof 2019 LA train |
| `asvspoof_validation` | 300 | 300 | ASVspoof 2019 LA validation |
| `asvspoof_test` | 500 | 500 | ASVspoof 2019 LA test |
| `indic_gujarati` | 300 | 300 | FLEURS gu_in + IndicSynth Gujarati |
| `indic_hindi` | 300 | 300 | FLEURS hi_in + IndicSynth Hindi |

**Full protocol, for reference** (Wang et al. [2], PDF p. 6): train 2,580 bonafide / 22,800 spoof; validation 2,548 bonafide / 22,296 spoof; test 7,355 bonafide / 63,882 spoof.

**Class balance.** ASVspoof 2019 LA is imbalanced — train is 2,580 bonafide against 22,800 spoof, roughly 1:8.8. This pipeline samples a **balanced** subset rather than class-weighting an imbalanced one. EER is threshold-free and prior-independent, so a balanced sample does not bias it, and balanced batches train more stably at this scale. No class weighting is applied.

## What these numbers mean — and what they don't

- **Single run, single seed.** No multi-seed averaging was performed. Differences of a point or two between conditions are inside the range a different seed could produce. Multi-seed averaging with reported variance is the first thing to add before these numbers are quoted as stable.
- **Reduced scale.** Training used a subsample of ASVspoof 2019 LA for a fraction of [11]'s 100-epoch schedule, on CPU rather than the RTX 3090 [11] used. The Phase 2 EER is this run's number and is not a reproduction of [11]'s 1.05%; most of the distance to that figure is training budget, not method.
- **The Phase 3 comparison is the deliverable**, not the absolute EERs. What the project set out to quantify is the *increase* from English to Indic languages under one fixed decision boundary, and that comparison is computed explicitly rather than left for the reader to subtract.
- **The Indic evaluation sets pair two different corpora.** Real Gujarati/Hindi comes from FLEURS and fake from IndicSynth, so real and fake differ in recording chain and content as well as in synthesis. Some of the measured separation may be corpus rather than synthesis — exactly the confound `mentor_demo_features.ipynb` was built to control for, and it is not controlled here. Phase 3 numbers should be read with that caveat attached.
- **min t-DCF is not reported.** [11] Eq. (26), PDF p. 8, defines it over the combined ASV + countermeasure system and needs the miss and false-alarm rates of a fixed ASV back-end on the same trials. No ASV back-end is available here and the ASVspoof mirror ships no ASV scores, so any figure computed from countermeasure scores alone would not be comparable to published t-DCF values. The column is recorded as *not computed* rather than filled with an estimate.
- **IIT-Mandi / AIR was not obtained.** Bonafide Indic speech comes from FLEURS read speech instead of AIR broadcast material, as documented in the paper.

## References

[2] X. Wang et al., "ASVspoof 2019: A large-scale public database of synthesized, converted and replayed speech," *Computer Speech & Language*, vol. 64, 2020.

[11] G. Tahaoglu, D. Baracchi, A. Piva, D. Shullani, M. Iuliani, "Deepfake audio detection with spectral features and ResNeXt-based architecture," *Knowledge-Based Systems*, vol. 323, p. 113726, 2025.
