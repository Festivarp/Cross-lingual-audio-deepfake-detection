# -*- coding: utf-8 -*-
"""Training loop: phase-aware (base training / fusion-only fine-tuning).

Hyperparameters
---------------
PAPER-SPECIFIED (Tahaoglu et al. [11], PDF p. 8, "Implementation details"):
  * Adam optimizer
  * learning rate 1e-4
  * batch size 32
  * "the model checkpoint corresponding to the lowest validation EER was
    selected for evaluation"  -> we checkpoint on validation EER, not loss

DEVIATION, stated openly:
  * [11] trains for 100 epochs on a single NVIDIA RTX 3090. This project has no
    GPU (torch CPU build). The epoch budget is therefore a runtime parameter and
    whatever was actually used is recorded in results/ and the README. A shorter
    schedule on a subsampled training set is not equivalent to [11]'s protocol
    and the resulting EER must not be read as a reproduction of their 1.05%.
"""
import csv
import os
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from .evaluate import evaluate

LR = 1e-4          # [11] p.8
BATCH_SIZE = 32    # [11] p.8


def make_loader(ds, batch_size=BATCH_SIZE, shuffle=False):
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, drop_last=False)


def train_model(model, train_ds, val_ds, epochs, ckpt_path, log_path,
                lr=LR, batch_size=BATCH_SIZE, device="cpu",
                params=None, pos_weight=None, tag="phase2"):
    """Train, checkpointing the best validation EER. Returns a history dict."""
    model.to(device)
    train_loader = make_loader(train_ds, batch_size, shuffle=True)
    val_loader = make_loader(val_ds, batch_size)

    trainable = list(params) if params is not None else \
        [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.Adam(trainable, lr=lr)
    crit = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor(pos_weight, device=device) if pos_weight else None)

    os.makedirs(os.path.dirname(ckpt_path), exist_ok=True)
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    history, best = [], float("inf")
    t_start = time.time()
    for ep in range(1, epochs + 1):
        model.train()
        if params is not None:                      # fusion-only: branches stay eval
            model.mfcc_branch.eval(); model.lfcc_branch.eval(); model.cqcc_branch.eval()

        tot, nb, t0 = 0.0, 0, time.time()
        for mfcc, lfcc, cqcc, y in train_loader:
            mfcc, lfcc, cqcc, y = (mfcc.to(device), lfcc.to(device),
                                   cqcc.to(device), y.to(device))
            opt.zero_grad()
            loss = crit(model(mfcc, lfcc, cqcc), y)
            loss.backward()
            opt.step()
            tot += float(loss.item()); nb += 1

        tr_loss = tot / max(nb, 1)
        val = evaluate(model, val_loader, device)
        row = {"epoch": ep, "train_loss": round(tr_loss, 5),
               "val_eer_percent": round(val["eer_percent"], 3),
               "epoch_seconds": round(time.time() - t0, 1)}
        history.append(row)
        star = ""
        if val["eer"] < best:
            best = val["eer"]
            torch.save({"model": model.state_dict(), "val_eer": best, "epoch": ep},
                       ckpt_path)
            star = "  <- best, saved"
        print(f"  [{tag}] epoch {ep:>3}/{epochs}  loss {tr_loss:.4f}  "
              f"val EER {val['eer_percent']:.2f}%  ({row['epoch_seconds']:.0f}s){star}",
              flush=True)

        with open(log_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(history[0].keys()))
            w.writeheader(); w.writerows(history)

    total_min = (time.time() - t_start) / 60.0
    print(f"  [{tag}] done: best val EER {100 * best:.2f}% over {epochs} epochs, "
          f"{total_min:.1f} min wall clock", flush=True)
    return {"history": history, "best_val_eer": best, "epochs": epochs,
            "wall_clock_minutes": round(total_min, 2), "checkpoint": ckpt_path}


def load_checkpoint(model, path, device="cpu"):
    ck = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"])
    return model, ck
