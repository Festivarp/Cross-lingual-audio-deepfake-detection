# -*- coding: utf-8 -*-
"""E4: a cost-aware cascade over the three cepstral branches.

    python -m deepfake_detection.cascade --sweep

The method
----------
Run the cheapest useful branch first. If it is confident, emit its verdict and
stop. Escalate only on clips it cannot settle.

    stage 1   cheapest family            cost: feat[f1] + one branch
    stage 2   + next family              cost: stage 1 + feat[f2] + one branch
    stage 3   + last family, score fusion cost: all features + three branches

Two components. The early-exit rule is standard and we do not claim it: exit when
|p_fake - 0.5| >= tau. The contribution is the **ordering policy** -- branches are
ranked by measured detection quality per unit of measured cost, using E2's
accuracy and E1's timings, rather than by accuracy alone. Because CQCC extraction
costs 42.6x MFCC extraction (E1), a family can be the most accurate and still
belong last.

Why stage 3 fuses scores rather than calling the triple-branch model
--------------------------------------------------------------------
[11]'s TripleBranchResNeXt has its own branch weights, learned jointly with its
MLP fusion head. Those are different tensors from the three single-branch
detectors, so escalating to it would mean recomputing all three branch forwards
from scratch and the cascade's worst case would exceed the baseline's cost.

Fusing the three single-branch scores instead lets every stage reuse what earlier
stages already computed. The cascade's worst case is then one feature extraction
of each family plus three branch forwards -- essentially the baseline's own cost.
The price is that the cascade's accuracy ceiling is score-level fusion, which is
weaker than [11]'s learned fusion. Both are reported; neither is hidden.

Scores
------
Stages use different models whose raw logits are not on a common scale, so every
stage emits a calibrated probability. The cascade's output score is
`1 - p_fake`, matching the "higher = more real" convention in evaluate.py, so
EER is computed over one consistent axis regardless of where a clip exited.
"""
import argparse
import itertools
import json
import os

import numpy as np

from .evaluate import compute_eer
from .models.single_branch import FAMILIES

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS = os.path.join(ROOT, "results")
SCORES = os.path.join(RESULTS, "scores")

EVAL_SETS = ("english", "gujarati", "hindi")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def load_cost():
    with open(os.path.join(RESULTS, "e1_cost_accounting.json"), encoding="utf-8") as fh:
        c = json.load(fh)
    return {
        "feat": {k: c["feature_extraction_ms"][k]["median_ms"] for k in FAMILIES},
        "branch": c["model"]["one_branch_batch1"]["median_ms"],
        "baseline_end_to_end": c["end_to_end_batch1_ms"]["baseline_all_three"],
    }


def load_branch_scores(seed):
    """p_fake per clip, per family, per evaluation set, for one seed."""
    out = {}
    for fam in FAMILIES:
        path = os.path.join(SCORES, f"e2_{fam}_seed{seed}.json")
        if not os.path.exists(path):
            raise SystemExit(f"missing {path} -- run `--e2` first")
        with open(path, encoding="utf-8") as fh:
            d = json.load(fh)
        fam_out = {}
        for name, raw in d["raw"].items():
            # score_model emits -logit (higher = more real); recover p_fake
            fam_out[name] = {
                "p_fake": sigmoid(-np.asarray(raw["scores"], dtype=np.float64)),
                "labels": np.asarray(raw["labels"], dtype=int),
            }
        out[fam] = {"eer": d["eer"], "sets": fam_out}
    return out


def stage_costs(order, cost):
    """Cumulative end-to-end milliseconds for exiting at stage 1, 2 or 3."""
    c1 = cost["feat"][order[0]] + cost["branch"]
    c2 = c1 + cost["feat"][order[1]] + cost["branch"]
    c3 = c2 + cost["feat"][order[2]] + cost["branch"]
    return [c1, c2, c3]


def run_cascade(branch, order, tau, eval_set, costs):
    """Returns (scores, labels, exit_index per clip, mean cost ms)."""
    sets = [branch[f]["sets"][eval_set] for f in order]
    labels = sets[0]["labels"]
    n = len(labels)

    p = [s["p_fake"] for s in sets]
    final = np.empty(n, dtype=np.float64)
    exit_at = np.full(n, 2, dtype=int)          # default: reached stage 3

    undecided = np.ones(n, dtype=bool)
    for stage in (0, 1):
        conf = np.abs(p[stage] - 0.5)
        take = undecided & (conf >= tau)
        final[take] = p[stage][take]
        exit_at[take] = stage
        undecided &= ~take

    # stage 3: score-level fusion over all three families
    fused = np.mean(np.stack(p, axis=0), axis=0)
    final[undecided] = fused[undecided]

    mean_cost = float(np.mean([costs[i] for i in exit_at]))
    return 1.0 - final, labels, exit_at, mean_cost


def ordering_policy(branch, cost, eval_set="english"):
    """Rank families by detection quality per unit cost.

    Quality is (50 - EER) -- distance above chance in percentage points, so a
    useless branch scores 0 and a perfect one scores 50. Cost is the branch's own
    end-to-end millisecond price from E1. The resulting order is what makes this
    cost-aware rather than accuracy-only.
    """
    score = {}
    for fam in FAMILIES:
        eer = branch[fam]["eer"][eval_set]
        quality = max(0.0, 50.0 - eer)
        price = cost["feat"][fam] + cost["branch"]
        score[fam] = quality / price
    return sorted(FAMILIES, key=lambda f: score[f], reverse=True), score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--order", default=None,
                    help="comma-separated family order; default = policy")
    args = ap.parse_args()

    cost = load_cost()
    branch = load_branch_scores(args.seed)

    policy_order, policy_score = ordering_policy(branch, cost)
    order = args.order.split(",") if args.order else policy_order

    print("=" * 70)
    print("ORDERING POLICY  (quality per millisecond, English)")
    print("=" * 70)
    print(f"  {'family':<8}{'EER %':>9}{'cost ms':>10}{'quality/ms':>13}")
    for fam in FAMILIES:
        print(f"  {fam.upper():<8}{branch[fam]['eer']['english']:>9.2f}"
              f"{cost['feat'][fam] + cost['branch']:>10.2f}"
              f"{policy_score[fam]:>13.3f}")
    print(f"\n  order: {' -> '.join(f.upper() for f in order)}")

    costs = stage_costs(order, cost)
    print(f"  stage costs (ms): {costs[0]:.2f} / {costs[1]:.2f} / {costs[2]:.2f}")
    print(f"  baseline [11] end-to-end: {cost['baseline_end_to_end']:.2f} ms")

    # ---- invariant: tau = inf must equal pure stage-3 fusion --------------
    s_inf, y, ex, _ = run_cascade(branch, order, np.inf, "english", costs)
    fused = np.mean(np.stack([branch[f]["sets"]["english"]["p_fake"]
                              for f in order], axis=0), axis=0)
    assert np.allclose(s_inf, 1.0 - fused), "tau=inf must be pure fusion"
    assert (ex == 2).all(), "tau=inf must never exit early"
    print("\n  invariant ok: tau=inf is exactly the 3-branch fusion")

    if not args.sweep:
        return

    taus = [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.49, 0.499]
    out = {"seed": args.seed, "order": order, "stage_costs_ms": costs,
           "baseline_end_to_end_ms": cost["baseline_end_to_end"],
           "policy_score": policy_score, "sweep": {}}

    for eval_set in EVAL_SETS:
        rows = []
        print("\n" + "=" * 70)
        print(f"CASCADE SWEEP - {eval_set.upper()}")
        print("=" * 70)
        print(f"  {'tau':>6}{'EER %':>9}{'mean ms':>10}{'vs base':>9}"
              f"{'exit1':>8}{'exit2':>8}{'exit3':>8}")
        for tau in taus:
            s, y, ex, mc = run_cascade(branch, order, tau, eval_set, costs)
            eer = 100.0 * compute_eer(s, y)[0]
            rates = [float((ex == i).mean()) for i in range(3)]
            rows.append({"tau": tau, "eer_percent": round(eer, 3),
                         "mean_cost_ms": round(mc, 3),
                         "speedup_vs_baseline": round(
                             cost["baseline_end_to_end"] / mc, 3),
                         "exit_rates": [round(r, 4) for r in rates]})
            print(f"  {tau:>6.3f}{eer:>9.2f}{mc:>10.2f}"
                  f"{cost['baseline_end_to_end'] / mc:>8.2f}x"
                  f"{rates[0]:>8.1%}{rates[1]:>8.1%}{rates[2]:>8.1%}")
        out["sweep"][eval_set] = rows

    path = os.path.join(RESULTS, "e4_cascade.json")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
