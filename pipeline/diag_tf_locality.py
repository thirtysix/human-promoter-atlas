#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Does a program's reproducibility track how much of its TFs' binding we can see?

The atlas scans +/-1500 bp around canonical TSSs, so a TF that binds mostly at
distal elements contributes only a thin tail to the occupancy matrix. Spot
checks on six TFs lined up suspiciously well with per-program stability:

    NFYA    69.8% of peaks within +/-1.5kb   ->  P10  stability 0.982
    MYC     48.0%                            ->  P8   stability 0.969
    JUND    13.0%                            ->  P6   stability 0.765
    ESR1     9.1%                            ->  P3   stability 0.897
    FOSL2    8.1%                            ->  P6   stability 0.765

This measures it for all TFs and tests the association properly. If it holds,
the unstable programs (AP-1, nuclear receptors) are unstable because we are
looking at 8-13% of their binding -- a field-of-view problem that no threshold
or rank can fix. If it does not hold, that explanation is wrong and the
alternative (cell-type pooling, see diag_cell_heterogeneity.py) is in play.

Deliberately uses the LOOSEST tier available (q1e-5 by default): the question is
where a TF binds in the genome, not where it binds significantly.

Sampling: large per-TF BEDs are stride-sampled. Rows are sorted by chromosome
then position, so a fixed stride is a systematic sample of the genome and
unbiased for a distance distribution. The stride is derived from the compressed
file size (a ~4x ratio for this data), so the realised sample size is
approximate -- it is reported per TF so a thin sample is visible rather than
silently trusted.

Usage:
    python pipeline/diag_tf_locality.py --k 18
    python pipeline/diag_tf_locality.py --tier q1e-5 --target-peaks 300000
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import datetime as dt
import gzip
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from config import CHIP_ATLAS_DN, OUT_DN, K_CANONICAL

PROXIMAL_BP = 1500          # the current analysis window half-width
MID_BP      = 10000         # proximal/distal boundary
DEFAULT_WORKERS = 8         # see the thermal note in CLAUDE.md; also plenty here
GZ_RATIO    = 4.0           # rough uncompressed:compressed ratio for these BEDs
BYTES_PER_ROW = 50          # rough mean row length


def _log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


_TSS = {}                   # chrom -> sorted np.array of TSS positions


def _init(tss_by_chrom):
    global _TSS
    _TSS = tss_by_chrom


def tf_locality(args):
    """Distance-to-nearest-TSS distribution for one TF's peaks."""
    path, target = args
    tf = path.name.replace(".bed.gz", "")
    try:
        est_rows = max(1, int(path.stat().st_size * GZ_RATIO / BYTES_PER_ROW))
        stride = max(1, est_rows // target)
        d = []
        with gzip.open(path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % stride:
                    continue
                f = line.split("\t", 3)
                c = f[0][3:] if f[0].startswith("chr") else f[0]
                arr = _TSS.get(c)
                if arr is None or not len(arr):
                    continue
                mid = (int(f[1]) + int(f[2])) // 2
                j = np.searchsorted(arr, mid)
                best = None
                for kk in (j - 1, j):
                    if 0 <= kk < len(arr):
                        dd = abs(mid - int(arr[kk]))
                        if best is None or dd < best:
                            best = dd
                if best is not None:
                    d.append(best)
        if not d:
            return dict(TF=tf, n_sampled=0, stride=stride)
        d = np.asarray(d)
        return dict(
            TF=tf, n_sampled=len(d), stride=stride,
            frac_proximal=float((d <= PROXIMAL_BP).mean()),
            frac_mid=float(((d > PROXIMAL_BP) & (d <= MID_BP)).mean()),
            frac_distal=float((d > MID_BP).mean()),
            median_dist=float(np.median(d)),
        )
    except Exception as exc:                       # one bad file must not kill the sweep
        return dict(TF=tf, n_sampled=0, stride=0, error=str(exc))


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="q1e-5")
    ap.add_argument("--k", type=int, default=None,
                    help="rank whose stability/H to join against (default: config)")
    ap.add_argument("--target-peaks", type=int, default=300000)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K = args.k or K_CANONICAL

    root = OUT_DN / "tss_modules"
    tss = pd.read_csv(root / "tss_table.tsv", sep="\t")
    tss_by_chrom = {str(c): np.sort(g["tss"].to_numpy())
                    for c, g in tss.groupby("chrom")}
    _log(f"{len(tss):,} canonical TSSs across {len(tss_by_chrom)} chromosomes")

    beds = sorted((CHIP_ATLAS_DN / "per_TF" / args.tier).glob("*.bed.gz"))
    if not beds:
        raise SystemExit(f"no BEDs under {CHIP_ATLAS_DN/'per_TF'/args.tier}")
    _log(f"tier {args.tier}: {len(beds):,} TFs, "
         f"target ~{args.target_peaks:,} sampled peaks each")

    t0 = time.time()
    rows = []
    with Pool(args.workers, initializer=_init, initargs=(tss_by_chrom,)) as pool:
        jobs = [(b, args.target_peaks) for b in beds]
        for i, r in enumerate(pool.imap_unordered(tf_locality, jobs, chunksize=4), 1):
            rows.append(r)
            if i % 200 == 0:
                _log(f"  {i}/{len(beds)}  ({time.time()-t0:.0f}s)")

    loc = pd.DataFrame(rows)
    ok = loc[loc.n_sampled > 0].copy()
    _log(f"locality computed for {len(ok):,}/{len(loc):,} TFs "
         f"in {(time.time()-t0)/60:.1f} min")

    # ---- join to the program loadings ------------------------------------
    H = pd.read_csv(root / f"nmf.k{K}.H.tsv.gz", sep="\t", index_col=0)
    stab = pd.read_csv(root / f"nmf.k{K}.stability.tsv", sep="\t")
    loc_by_tf = ok.set_index("TF")["frac_proximal"]
    shared = [c for c in H.columns if c in loc_by_tf.index]
    _log(f"  {len(shared):,}/{H.shape[1]:,} TFs in H have a locality estimate")

    Hs = H[shared].to_numpy(float)
    w = Hs / (Hs.sum(axis=1, keepdims=True) + 1e-12)   # loading weights per program
    prox = loc_by_tf.reindex(shared).to_numpy(float)
    stab = stab.sort_values("program").reset_index(drop=True)
    stab["weighted_proximal"] = w @ prox

    out = args.out or (root / "k_selection" / f"tf_locality.k{K}.tsv")
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    ok.sort_values("frac_proximal").to_csv(
        str(out).replace(".tsv", ".per_tf.tsv"), sep="\t", index=False,
        float_format="%.6f")
    stab.to_csv(out, sep="\t", index=False, float_format="%.6f")

    # ---- report -----------------------------------------------------------
    print()
    _log(f"=== per-program: stability vs loading-weighted proximal fraction (k={K}) ===")
    print(f"{'prog':>5}{'stability':>11}{'wProximal':>11}  top TFs")
    for _, r in stab.sort_values("weighted_proximal").iterrows():
        flag = "" if r["median_cosine"] >= 0.90 else "  <-- unstable"
        print(f"{int(r['program']):>5}{r['median_cosine']:>11.3f}"
              f"{r['weighted_proximal']:>11.3f}  {str(r['top_tfs'])[:40]}{flag}")
    rho = stab[["median_cosine", "weighted_proximal"]].corr(method="spearman").iloc[0, 1]
    pear = stab[["median_cosine", "weighted_proximal"]].corr().iloc[0, 1]
    print(f"\n  Spearman(stability, weighted proximal) = {rho:+.3f}   n={len(stab)}")
    print(f"  Pearson                                 = {pear:+.3f}")
    print("  NOTE observational, n = number of programs. A positive association")
    print("       supports the field-of-view explanation; it does not establish it.")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
