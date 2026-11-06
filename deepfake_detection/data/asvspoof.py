# -*- coding: utf-8 -*-
"""ASVspoof 2019 LA loader over the `Bisher/ASVspoof_2019_LA` Parquet mirror.

Why not `datasets` streaming
----------------------------
The exploratory notebooks stream this mirror with `load_dataset(..., streaming=True)`
and a ClassLabel-safe `key` match. That pattern works, but it is strictly
sequential, and this mirror's Parquet files are **sorted by label**: probing the
file shows bonafide (`key`=0) occupying row groups 0-25 and spoof (`key`=1) from
row group 25 onward, in both the train and validation splits. Sequential
streaming at the measured ~7 rows/s would therefore need ~1 hour to reach any
spoof example at all, and a naive "first N rows" subsample would be 100%
bonafide.

The files are chunked into row groups of 100 rows, and Hugging Face serves HTTP
range requests, so `pyarrow` can read *selected* row groups directly. This module
does that: it picks row groups from each label region and reads only those. The
`key` value of every row is still checked explicitly, so nothing depends on the
label layout holding -- if the layout changed, the counts reported would simply
differ, and `load_subset` reports what it actually found.

Splits map to the official ASVspoof protocol partitions: `train` for base
training, `validation` for model selection, `test` for the Phase-2 eval figure
(Wang et al. [2], PDF p. 6).
"""
import time

import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

REPO = "Bisher/ASVspoof_2019_LA"
PATH = "datasets/" + REPO + "/data/{split}-00000-of-00001.parquet"

# Official LA partition sizes, from Wang et al. [2], PDF p. 6.
PROTOCOL = {
    "train":      {"bonafide": 2580, "spoof": 22800},
    "validation": {"bonafide": 2548, "spoof": 22296},
    "test":       {"bonafide": 7355, "spoof": 63882},
}

BONAFIDE, SPOOF = 0, 1


def _open(split):
    fs = HfFileSystem()
    return pq.ParquetFile(fs.open(PATH.format(split=split), "rb"))


def _row_group_plan(pf, want_bonafide, want_spoof):
    """Order the row groups to visit.

    The two splits differ in layout, so the plan must not assume either:

      * `train` and `validation` are SORTED BY LABEL -- bonafide fills row groups
        0-25, spoof runs from 25 to the end (verified by probing the `key` column).
      * `test` is INTERLEAVED -- the first 600 rows come out ~11% bonafide, which
        is the corpus's natural 7,355 : 63,882 ratio rather than a sorted block.

    So the plan is a visiting *order*, not a per-class assignment: the low groups
    first (which is where bonafide lives if the split is sorted), then a stride
    across the whole file (which picks up both classes if it is interleaved, and
    spans the different attack systems either way). `load_subset` walks this
    order and stops as soon as both quotas are met, so a sorted split costs only
    the handful of groups it needs while an interleaved split keeps going until
    the rarer class is satisfied.
    """
    md = pf.metadata
    n_rg = md.num_row_groups
    rg_rows = md.row_group(0).num_rows or 100              # 100 for this mirror

    head = list(range(0, min(-(-want_bonafide // rg_rows) + 1, n_rg)))

    # Enough groups that even a ~10% minority class can fill its quota.
    need = -(-max(want_bonafide, want_spoof) // rg_rows)
    stride = max(1, n_rg // max(1, need * 12))
    rest = [rg for rg in range(0, n_rg, stride) if rg not in set(head)]

    order = head + rest
    seen, plan = set(), []
    for rg in order:                                       # de-duplicate, keep order
        if rg not in seen:
            seen.add(rg); plan.append(rg)
    return plan


def load_subset(split, n_bonafide, n_spoof, verbose=True):
    """Read up to `n_bonafide` + `n_spoof` examples from one ASVspoof LA split.

    Returns a list of dicts: {"audio": bytes, "label": 0|1, "speaker_id",
    "system_id"} where label 0 = bonafide (real), 1 = spoof (fake).
    Reports the real counts obtained, which may be fewer than requested.
    """
    t0 = time.time()
    pf = _open(split)
    plan = _row_group_plan(pf, n_bonafide, n_spoof)

    out, got = [], {BONAFIDE: 0, SPOOF: 0}
    want = {BONAFIDE: n_bonafide, SPOOF: n_spoof}

    for rg in plan:
        if got[BONAFIDE] >= want[BONAFIDE] and got[SPOOF] >= want[SPOOF]:
            break                                        # both quotas met
        tbl = pf.read_row_group(
            rg, columns=["audio", "key", "speaker_id", "system_id"])
        keys = tbl.column("key").to_pylist()
        audios = tbl.column("audio").to_pylist()
        spks = tbl.column("speaker_id").to_pylist()
        syss = tbl.column("system_id").to_pylist()
        for k, a, sp, sy in zip(keys, audios, spks, syss):
            k = int(k)
            if k not in want or got[k] >= want[k]:
                continue
            blob = a["bytes"] if isinstance(a, dict) else a
            if not blob:
                continue
            out.append({"audio": blob, "label": k,
                        "speaker_id": sp, "system_id": sy})
            got[k] += 1
        if verbose:
            print(f"    [{split}] rg {rg:>4}  bonafide={got[BONAFIDE]:<5} "
                  f"spoof={got[SPOOF]:<5} ({time.time() - t0:.0f}s)", flush=True)

    if verbose:
        short = [f"{'bonafide' if c == BONAFIDE else 'spoof'} "
                 f"{got[c]}/{want[c]}" for c in (BONAFIDE, SPOOF) if got[c] < want[c]]
        print(f"  {split}: {got[BONAFIDE]} bonafide + {got[SPOOF]} spoof "
              f"= {len(out)} clips in {time.time() - t0:.0f}s"
              + (f"  [SHORT: {', '.join(short)}]" if short else ""), flush=True)
    return out
