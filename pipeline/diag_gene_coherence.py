#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Do a gene's elements share regulatory character, or only a neighbourhood?

genome_gene_programs.py measured that a gene's program composition is nearly
flat at k=140 (median normalised spread 0.851; 98.6% of genes have a top
program below 25%). That result has two incompatible readings:

    STRUCTURAL   a gene's elements genuinely differ, so no gene-level program
                 identity exists at any granularity, and archetypes are
                 meaningless however the programs are coarsened.

    DILUTIVE     real coherence exists but is spread across 140 categories, so
                 clustering programs into meta-programs would recover it.

This decides between them, because the second reading is the entire premise of
the meta-program approach and coarsening will produce SOMETHING either way --
with ~5 elements per gene and 8 categories, a little concentration appears by
arithmetic alone.

THE NULL HAS TO CONTROL FOR DISTANCE
------------------------------------
Elements at one gene are also close together in the genome, and nearby elements
may share programs for reasons that have nothing to do with the gene -- shared
chromatin domain, one broad regulatory region, the same TAD. A null that pairs
elements at random genome-wide would credit all of that to gene identity.

So same-gene pairs are compared against different-gene pairs AT MATCHED
GENOMIC SEPARATION. If the two curves coincide, apparent coherence is
proximity and nothing more. If same-gene sits above at equal separation, gene
identity carries information that clustering could recover.

Usage:
    python pipeline/diag_gene_coherence.py --genome-dir <dir> --k 140
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Separation bins (bp). Same-gene promoter/proximal pairs are mostly < 20 kb,
# so the bins are fine at the short end where the comparison actually lives.
BINS = np.array([0, 500, 1000, 2000, 5000, 10000, 20000, 50000, 200000])
MAX_ELEMENTS_PER_GENE = 40      # guard against a pathological gene dominating
NULL_PAIRS_PER_BIN = 40000


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _bp(x):
    return f"{x // 1000}kb" if x >= 1000 else f"{x}bp"


def _pairs_within(idx_by_gene, rng, cap):
    """All within-gene index pairs (capped per gene)."""
    a, b = [], []
    for v in idx_by_gene:
        if len(v) < 2:
            continue
        if len(v) > cap:
            v = rng.choice(v, cap, replace=False)
        i, j = np.triu_indices(len(v), k=1)
        a.append(np.asarray(v)[i]); b.append(np.asarray(v)[j])
    return np.concatenate(a), np.concatenate(b)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--strata", default="promoter,proximal")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    root = Path(args.genome_dir)
    rng = np.random.default_rng(args.seed)

    el = pd.read_csv(root / "elements.genes.tsv", sep="\t", dtype={"chrom": str})
    z = np.load(root / f"nmf.k{args.k}.W.npz")
    W, wid = z["W"], z["element_id"]
    order = pd.Series(np.arange(len(wid)), index=wid).reindex(
        el.element_id.to_numpy()).to_numpy()
    if np.isnan(order).any():
        raise SystemExit("elements.genes.tsv and W.npz are not the same build")
    W = W[order.astype(np.int64)]
    n = np.linalg.norm(W, axis=1, keepdims=True)
    W = W / np.where(n > 0, n, 1.0)          # unit rows -> dot == cosine

    strata = [s.strip() for s in args.strata.split(",") if s.strip()]
    keep = (el.stratum.isin(strata) & el.nearest_gene_name.notna()
            & (el.nearest_gene_name.astype(str) != "")).to_numpy()
    sub = el[keep].reset_index(drop=True)
    Ws = W[keep]
    _log(f"{len(sub):,} elements in {strata}, {sub.nearest_gene_name.nunique():,} genes")

    idx_by_gene = [v.to_numpy() for _, v in
                   pd.Series(np.arange(len(sub))).groupby(
                       sub.nearest_gene_name.to_numpy())]
    ia, ib = _pairs_within(idx_by_gene, rng, MAX_ELEMENTS_PER_GENE)
    same_cos = np.einsum("ij,ij->i", Ws[ia], Ws[ib])
    same_sep = np.abs(sub.peak.to_numpy()[ia] - sub.peak.to_numpy()[ib])
    _log(f"within-gene pairs: {len(ia):,}")

    # Distance-matched null. Random same-chromosome pairs will NOT do: they are
    # almost never a few kb apart, so every short-separation bin -- which is
    # where promoter/proximal pairs actually live -- comes back empty and the
    # comparison silently reduces to the 10-50 kb tail. Measured on the first
    # attempt: only 35,738 of 417,147 within-gene pairs survived binning.
    #
    # Instead, TARGET each observed same-gene separation: pick a random element,
    # look for a partner at approximately that offset, and keep it if the gene
    # differs. That populates the same bins the observed data occupies.
    chroms = sub.chrom.to_numpy()
    genes = sub.nearest_gene_name.to_numpy()
    peaks = sub.peak.to_numpy()
    by_chrom = {c: np.flatnonzero(chroms == c) for c in pd.unique(chroms)}
    by_chrom = {c: v[np.argsort(peaks[v])] for c, v in by_chrom.items()
                if len(v) >= 100}
    targets = rng.choice(same_sep, min(len(same_sep), 400_000), replace=False)
    clist = list(by_chrom)
    na, nb = [], []
    cpick = rng.choice(len(clist), len(targets))
    for ci in range(len(clist)):
        want = targets[cpick == ci]
        if not len(want):
            continue
        v = by_chrom[clist[ci]]
        pv = peaks[v]
        i = rng.integers(0, len(v), len(want))
        sign = rng.choice([-1, 1], len(want))
        j = np.searchsorted(pv, pv[i] + sign * want)
        j = np.clip(j, 0, len(v) - 1)
        got = np.abs(pv[j] - pv[i])
        # keep only pairs whose realised separation is close to the target
        ok = ((i != j) & (genes[v[i]] != genes[v[j]])
              & (np.abs(got - want) <= np.maximum(want * 0.25, 200)))
        na.append(v[i[ok]]); nb.append(v[j[ok]])
    na = np.concatenate(na); nb = np.concatenate(nb)
    null_cos = np.einsum("ij,ij->i", Ws[na], Ws[nb])
    null_sep = np.abs(peaks[na] - peaks[nb])
    _log(f"different-gene pairs: {len(na):,}")

    print()
    _log("=== mean cosine between element program profiles ===")
    print(f"{'separation':>18}{'same gene':>12}{'n':>10}"
          f"{'diff gene':>12}{'n':>10}{'delta':>9}")
    rows = []
    for lo, hi in zip(BINS[:-1], BINS[1:]):
        s = same_cos[(same_sep >= lo) & (same_sep < hi)]
        d = null_cos[(null_sep >= lo) & (null_sep < hi)]
        if len(s) < 50 or len(d) < 50:
            continue
        # NOT "diff": DataFrame.diff is a built-in method, so r.diff would
        # return the bound method instead of the column and fail obscurely.
        rows.append(dict(lo=int(lo), hi=int(hi), same_gene=float(s.mean()),
                         n_same=len(s), diff_gene=float(d.mean()),
                         n_diff=len(d), delta=float(s.mean() - d.mean())))
        label = f"{_bp(lo)}-{_bp(hi)}"
        print(f"{label:>18}{s.mean():>12.4f}{len(s):>10,}"
              f"{d.mean():>12.4f}{len(d):>10,}{s.mean()-d.mean():>+9.4f}")
    r = pd.DataFrame(rows)
    out = root / f"gene_programs.k{args.k}" / "gene_coherence.tsv"
    out.parent.mkdir(parents=True, exist_ok=True)
    r.to_csv(out, sep="\t", index=False, float_format="%.6f")

    wt = r.n_same / r.n_same.sum()
    print(f"\n  overall same-gene    : {float((r.same_gene*wt).sum()):.4f}")
    print(f"  distance-matched null: {float((r.diff_gene*wt).sum()):.4f}")
    print(f"  DELTA                : "
          f"{float(((r.same_gene-r.diff_gene)*wt).sum()):+.4f}")
    print(f"\n  naive (unmatched) null: {null_cos.mean():.4f}  "
          f"<- what a non-distance-controlled test would have used")
    print(f"  wrote {out}")
    print("\n  Delta near zero => apparent gene coherence is proximity; "
          "coarsening\n  programs into meta-programs would manufacture "
          "structure, not recover it.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
