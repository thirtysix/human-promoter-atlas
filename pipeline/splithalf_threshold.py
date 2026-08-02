#!/usr/bin/env python3
"""
Calibrate MIN_SCORE_ASSIGN by split-half reproducibility, per TF.

The question this answers
------------------------
Lowering the assignment score buys more (module, TF) assignments. Which of them
are real? Agreement with the incumbent q1e-50 atlas cannot say -- the premise of
loosening the threshold is that the incumbent is wrong -- and GO enrichment gives
a direction but no optimum, and is confounded by gene-set size.

Reproducibility across DISJOINT EXPERIMENTS is external to all of that. Split a
TF's ChIP experiments into two halves; an assignment supported by real binding
should appear in both halves, one driven by a single noisy peak should not.

    precision proxy  = replication rate: assignments from half A also made in B
    recall proxy     = number of assignments made

Modules are held FIXED (from the full build), so across every threshold and both
halves the only thing that varies is the evidence. Module discovery is
score-independent in this pipeline, so this is legitimate.

Per-TF, not global
------------------
A factor with 2,000 experiments and one with 6 do not deserve the same evidence
bar. This reports a curve per TF and picks each TF's own threshold as the LOWEST
score still holding a declared replication floor -- declared before looking.

Usage:
    python pipeline/splithalf_threshold.py \
        --modules  <full_build>/tss_modules \
        --half-a   <A_build>/tss_modules \
        --half-b   <B_build>/tss_modules \
        --floor 0.8 --out splithalf.tsv
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

SCORES = [500, 400, 300, 250, 200, 150, 100, 50]


def load_half(dn: Path):
    """peaks.parquet + tf_index.tsv -> (df with tf NAME, not index)."""
    p = pd.read_parquet(dn / "peaks.parquet", columns=["tss_id", "tf_idx", "local", "score"])
    tf = pd.read_csv(dn / "tf_index.tsv", sep="\t")
    names = tf.sort_values("tf_idx")["TF"].to_numpy() if "tf_idx" in tf.columns \
        else tf.iloc[:, 0].to_numpy()
    p["tf"] = names[p["tf_idx"].to_numpy()]
    return p


def assignments(peaks: pd.DataFrame, mods: pd.DataFrame, score: int) -> set:
    """Set of (module_id, tf) with >=1 peak of at least `score` inside the module.

    Mirrors the sp_mask in tss_modules: a TF is assigned to a module iff it has
    a peak at or above the threshold within [lo_offset, hi_offset].
    """
    p = peaks[peaks["score"] >= score]
    if p.empty:
        return set()
    m = mods[["module_id", "tss_id", "lo_offset", "hi_offset"]]
    j = p.merge(m, on="tss_id", copy=False)
    hit = (j["local"] >= j["lo_offset"]) & (j["local"] <= j["hi_offset"])
    j = j.loc[hit, ["module_id", "tf"]]
    return set(map(tuple, j.drop_duplicates().to_numpy()))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modules", required=True, type=Path)
    ap.add_argument("--half-a", required=True, type=Path)
    ap.add_argument("--half-b", required=True, type=Path)
    ap.add_argument("--floor", type=float, default=0.8,
                    help="replication floor a threshold must hold (declare before looking)")
    ap.add_argument("--min-assign", type=int, default=20,
                    help="TFs with fewer assignments at the strictest score are unreliable")
    ap.add_argument("--out", type=Path, default=Path("splithalf_threshold.tsv"))
    args = ap.parse_args()

    mods = pd.read_csv(args.modules / "modules.tsv", sep="\t", low_memory=False)
    n_modules = len(mods)
    A = load_half(args.half_a)
    B = load_half(args.half_b)
    print(f"  modules {len(mods):,}   half-A peaks {len(A):,}   half-B peaks {len(B):,}")

    rows = []
    for s in SCORES:
        sa = assignments(A, mods, s)
        sb = assignments(B, mods, s)
        by_tf_a, by_tf_b = {}, {}
        for mid, tf in sa:
            by_tf_a.setdefault(tf, set()).add(mid)
        for mid, tf in sb:
            by_tf_b.setdefault(tf, set()).add(mid)
        for tf, ma in by_tf_a.items():
            mb = by_tf_b.get(tf, set())
            inter = len(ma & mb)
            raw = (inter / len(ma) + inter / len(mb)) / 2 if mb else 0.0
            # RAW REPLICATION IS NOT A PRECISION MEASURE. As the threshold
            # falls, both halves assign a TF to nearly every module, so they
            # overlap almost perfectly by construction and raw replication rises
            # toward 1 -- rewarding exactly the saturation we are trying to
            # detect. Correct for chance: compare the observed overlap with what
            # two independent draws of the same sizes would give.
            exp = len(ma) * len(mb) / n_modules if mb else 0.0
            ceil = min(len(ma), len(mb))
            adj = (inter - exp) / (ceil - exp) if ceil > exp else 0.0
            rows.append({"tf": tf, "score": s, "n_a": len(ma), "n_b": len(mb),
                         "n_shared": inter, "expected": exp,
                         "replication_raw": raw, "replication": adj})
        print(f"  score {s:>4}: {len(sa):>9,} A-assignments  {len(sb):>9,} B  "
              f"({len(by_tf_a):,} TFs)")

    df = pd.DataFrame(rows)
    df.to_csv(args.out, sep="\t", index=False)

    # Per-TF pick: lowest score still holding the floor. Requires the TF to have
    # enough assignments at the strictest score to judge at all.
    picks = []
    for tf, g in df.groupby("tf"):
        g = g.sort_values("score", ascending=False)
        strict = g.iloc[0]
        if strict["n_a"] < args.min_assign:
            picks.append({"tf": tf, "pick": None, "reason": "too few assignments"})
            continue
        best = g.loc[g["replication"].idxmax()]
        picks.append({"tf": tf,
                      "pick": int(best["score"]),
                      "reason": "" if best["replication"] >= args.floor else "peak below floor",
                      "rep_at_pick": float(best["replication"]),
                      "n_assign_at_pick": int(best["n_a"]),
                      "n_assign_strict": int(strict["n_a"])})
    pk = pd.DataFrame(picks)
    pk.to_csv(args.out.with_suffix(".picks.tsv"), sep="\t", index=False)

    print(f"\n  replication floor {args.floor}")
    print(f"  TFs judged            {int(pk['pick'].notna().sum()):,}")
    print(f"  TFs never reaching it {int((pk['reason']=='never reaches floor').sum()):,}")
    print(f"  TFs too sparse        {int((pk['reason']=='too few assignments').sum()):,}")
    got = pk[pk["pick"].notna()]
    if len(got):
        print("\n  chosen threshold distribution:")
        for s, n in got["pick"].value_counts().sort_index(ascending=False).items():
            print(f"    score {int(s):>4}: {n:>5} TFs  {'#'*int(60*n/len(got))}")
        print(f"\n  median per-TF threshold {int(got['pick'].median())}")
    print(f"\n  wrote {args.out} and {args.out.with_suffix('.picks.tsv')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
