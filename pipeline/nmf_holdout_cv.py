#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Rank selection for the module x TF occupancy NMF by held-out imputation.

Why not the stability metrics
-----------------------------
Every rank statistic previously used here is confounded with k itself:

    reproducible fraction   k is in the denominator -> favours small k
    reproducible count      more components = more chances -> favours large k
    ARI on hard partitions  gets noisier as k grows -> favours small k
    cophenetic              flat 0.900-0.915 over k=8..20 -> decides nothing

They measure RELIABILITY, not FIT. A rank that merges two complexes into one
component can be perfectly reliable -- the merged blob is easy to recover every
seed -- which is exactly what k=10 does to cohesin, E-box and the pausing
machinery on the 1,793-TF axis.

The criterion here
------------------
Mask a random subset of matrix entries, fit rank-k NMF on the observed entries
only, and score the HELD-OUT entries. Under-fitting (k too small) cannot
reconstruct held-out structure; over-fitting (k too large) explains noise in
the observed entries and generalises worse. So this has an interior optimum by
construction, which is the property the stability metrics lack. Standard
model selection for matrix factorization: Wold (1978) style holdout, and
Owen & Perry (2009) bi-cross-validation.

Scoring is AUC over held-out entries. The matrix is ~2.8% dense, and AUC is a
rank statistic, so the class imbalance does not distort it the way accuracy or
Frobenius error would. Held-out Frobenius error is reported alongside.

Masked NMF
----------
sklearn's NMF cannot fit with missing entries, so the weighted multiplicative
updates are implemented here (Blondel et al. 2008). With mask O (1 = observed):

    W <- W * ((O*X) H^T) / ((O*(WH)) H^T)
    H <- H * (W^T (O*X)) / (W^T (O*(WH)))

Setting O to all ones recovers ordinary Frobenius NMF, which is what the rest
of the pipeline fits -- so the loss being optimised here is the same one, just
restricted to the observed entries.

Usage:
    python pipeline/nmf_holdout_cv.py --ranks 5,8,10,12,15,18,20,25,30 \
        --folds 3 --subsample 25000
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import datetime as dt
import json
import time

import numpy as np
import pandas as pd
import scipy.sparse as sp
from pathlib import Path

from config import OUT_DN, TIER, TF_SET, MIN_SCORE_ASSIGN

EPS = 1e-10


def _log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


################################################################################
# Masked NMF ###################################################################
################################################################################
def masked_nmf(X, O, k, seed, max_iter=200, tol=1e-5):
    """Frobenius NMF fitted only on entries where O == 1.

    X, O are dense float32 [n x m]. Returns (W, H).
    """
    rng = np.random.default_rng(seed)
    n, m = X.shape
    scale = np.sqrt(max(X[O > 0].mean(), EPS) / k)
    W = np.abs(rng.normal(scale=scale, size=(n, k))).astype(np.float32) + EPS
    H = np.abs(rng.normal(scale=scale, size=(k, m))).astype(np.float32) + EPS
    OX = O * X
    prev = None
    for it in range(max_iter):
        R = O * (W @ H)
        W *= (OX @ H.T) / (R @ H.T + EPS)
        R = O * (W @ H)
        H *= (W.T @ OX) / (W.T @ R + EPS)
        if it % 25 == 24:
            err = float(np.linalg.norm(OX - O * (W @ H)))
            if prev is not None and abs(prev - err) / max(prev, EPS) < tol:
                break
            prev = err
    return W, H


def auc(scores, labels):
    """Rank-based AUC. No sklearn dependency, handles ties by average rank."""
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    # average ranks within tied score groups
    s = scores[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[j + 1] == s[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = (i + 1 + j + 1) / 2.0
        i = j + 1
    pos = labels > 0
    n_pos, n_neg = int(pos.sum()), int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="5,8,10,12,15,18,20,25,30")
    ap.add_argument("--folds", type=int, default=3)
    ap.add_argument("--holdout", type=float, default=0.10,
                    help="fraction of entries masked per fold")
    ap.add_argument("--subsample", type=int, default=25000,
                    help="modules sampled for tractability (0 = all)")
    ap.add_argument("--max-iter", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--matrix", default=None,
                    help="occupancy .npz; default is the promoter build's "
                         "occupancy.modules.npz. Point this at a genome run's "
                         "occupancy.elements.npz to select a rank for it.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]
    if len(ranks) < 3:
        raise SystemExit(
            f"--ranks gave {len(ranks)} rank(s): {ranks}. Refusing to run.\n"
            f"  Selecting a 'best' rank from fewer than 3 points is meaningless,\n"
            f"  and this is a silent-failure trap: sbatch --export is COMMA\n"
            f"  separated, so --export=...,RANKS=\"5,8,10\" arrives as RANKS=5\n"
            f"  and the job reports a confident best-of-one. Export RANKS in the\n"
            f"  submitting shell and use --export=ALL instead.")

    root = OUT_DN / "tss_modules"
    mpath = Path(args.matrix) if args.matrix else root / "occupancy.modules.npz"
    M = sp.load_npz(str(mpath)).tocsr()
    _log(f"  matrix {mpath}")
    _log(f"build {OUT_DN.name} (tier={TIER} tf_set={TF_SET} "
         f"min_score_assign={MIN_SCORE_ASSIGN})")
    _log(f"  occupancy {M.shape}  nnz={M.nnz:,}  density={M.nnz/np.prod(M.shape):.3%}")

    rng = np.random.default_rng(args.seed)
    if args.subsample and args.subsample < M.shape[0]:
        # Sample modules, then drop TFs left with no assignment: an all-zero
        # column carries no held-out signal and only inflates the negative pool.
        idx = rng.choice(M.shape[0], args.subsample, replace=False)
        M = M[idx]
        keep = np.asarray(M.sum(axis=0)).ravel() > 0
        M = M[:, keep]
        _log(f"  subsampled to {M.shape}  nnz={M.nnz:,}")

    X = np.asarray(M.todense(), dtype=np.float32)
    n, m = X.shape
    total = n * m

    rows = []
    t_all = time.time()
    for fold in range(args.folds):
        # Mask a uniform random set of entries. Held out are mostly zeros
        # (the matrix is sparse); AUC is a rank statistic so that is fine, and
        # it is the honest question: can the fit place the few real 1s above
        # the many real 0s it never saw?
        fr = np.random.default_rng(args.seed + 1000 + fold)
        mask_flat = fr.random(total) < args.holdout
        O = (~mask_flat).reshape(n, m).astype(np.float32)
        held = mask_flat.reshape(n, m)
        y = X[held]
        _log(f"fold {fold}: held out {held.sum():,} entries "
             f"({y.sum():,.0f} positives, {y.mean():.3%} dense)")
        for k in ranks:
            t0 = time.time()
            W, H = masked_nmf(X, O, k, seed=args.seed + fold, max_iter=args.max_iter)
            P = (W @ H)[held]
            a = auc(P, y)
            rmse = float(np.sqrt(np.mean((P - y) ** 2)))
            rows.append(dict(fold=fold, k=k, auc=a, rmse=rmse,
                             secs=round(time.time() - t0, 1)))
            _log(f"    k={k:>3}  heldout AUC={a:.4f}  RMSE={rmse:.4f}  "
                 f"({time.time()-t0:.0f}s)")

    d = pd.DataFrame(rows)
    g = (d.groupby("k")
           .agg(auc_mean=("auc", "mean"), auc_sd=("auc", "std"),
                rmse_mean=("rmse", "mean"), secs=("secs", "sum"))
           .reset_index())
    # Write beside the matrix, not into the promoter build, or a genome-wide
    # sweep silently overwrites the promoter one that chose k=18.
    out = args.out or (mpath.parent / "k_selection" / "holdout_cv.tsv"
                       if args.matrix else
                       root / "k_selection" / "holdout_cv.tsv")
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    g.to_csv(out, sep="\t", index=False, float_format="%.6f")

    print()
    _log(f"=== held-out imputation, {args.folds} folds, "
         f"{args.holdout:.0%} masked ===")
    print(f"{'k':>5}{'AUC':>10}{'sd':>9}{'RMSE':>10}")
    for _, r in g.iterrows():
        star = "  <-- best" if r.k == g.loc[g.auc_mean.idxmax(), "k"] else ""
        print(f"{int(r.k):>5}{r.auc_mean:>10.4f}{r.auc_sd:>9.4f}{r.rmse_mean:>10.4f}{star}")
    best = int(g.loc[g.auc_mean.idxmax(), "k"])
    print(f"\n  best k by held-out AUC: {best}")
    print(f"  wrote {out}")
    _log(f"total {(time.time()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
