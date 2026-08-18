#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
How many TFs co-occur in an element by chance? Calibrating the support floor.

MIN_SUPPORT = 2 was calibrated for dense promoter windows. Applied genome-wide
it admits essentially any pair of overlapping peaks: on chr20-22, 82% of
elements are distal with a median of 2 assigned TFs, which IS the floor. An NMF
on that matrix would find "distal programs" that are artifacts of sparsity.

Why not an elbow
----------------
Measured on chr20-22, the cumulative retention curve has none -- 66/51/41/34/
30/23/19/13/10/7% at floors 2/3/4/5/6/8/10/15/20/30. It decays smoothly, so any
cut point is a judgement defensible one notch either side. Same failure as the
GO odds ratio (monotone) and cophenetic (flat): a criterion with no optimum
cannot select one.

The null
--------
Circularly shift each TF's peaks by an independent random offset within each
chromosome, then re-run detection unchanged. This preserves

    * each TF's exact peak count, and
    * each TF's OWN spatial clustering (peaks concentrate in open chromatin),

while destroying co-occurrence BETWEEN TFs -- which is exactly the null
hypothesis "these factors are near each other by coincidence". A uniform
shuffle would also destroy the within-TF clustering that makes real peaks pile
up, and would therefore be far too permissive.

FDR at a support floor f:

    FDR(f) = mean null elements with n_tfs_assigned >= f
             --------------------------------------------
             observed elements with n_tfs_assigned >= f

Reported per stratum as well as overall, because the local background differs
and a floor calibrated on promoters need not be right for distal elements.

Caveat carried forward: the null is per-TF independent, so it does not model
co-binding driven by shared chromatin accessibility. It answers "more than
chance given each TF's own distribution", not "more than expected given open
chromatin". Elements passing it are non-random, not necessarily functional.

Usage:
    python pipeline/genome_support_null.py --chroms 20,21,22 --shuffles 5
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

from config import (MIN_SCORE_ASSIGN, TIER, TF_SET,
                    analysis_dir, write_analysis_readme)

import importlib.util as _ilu


def _load(name, fname):
    spec = _ilu.spec_from_file_location(
        name, str(Path(__file__).resolve().parent / fname))
    mod = _ilu.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


_gm = _load("_gm", "genome_modules.py")          # peak loading + constants
_detect = _gm._detect_modules_density

KDE_BW, MIN_PEAK_DIST_BP = _gm.KDE_BW, _gm.MIN_PEAK_DIST_BP
BOUNDARY_FRAC, NBHD_BP, RECENTER_HALF = _gm.BOUNDARY_FRAC, _gm.NBHD_BP, _gm.RECENTER_HALF
PROMOTER_BP, PROXIMAL_BP = _gm.PROMOTER_BP, _gm.PROXIMAL_BP
DETECT_SUPPORT = 2                                # keep everything; floor applied after


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def local_weights(mid, tfi):
    """1 / (peaks of the same TF within +/-NBHD_BP), per peak."""
    w = np.empty(len(mid), np.float32)
    ordr = np.argsort(tfi, kind="stable")
    tf_s = tfi[ordr]
    b = np.flatnonzero(np.diff(tf_s)) + 1
    for s, e in zip(np.concatenate(([0], b)), np.concatenate((b, [len(tf_s)]))):
        idx = ordr[s:e]
        q = mid[idx]
        lo = np.searchsorted(q, q - NBHD_BP, "left")
        hi = np.searchsorted(q, q + NBHD_BP, "right")
        w[idx] = 1.0 / (hi - lo)
    return w


def detect_on(pos, tfi, sc, chrom_len):
    """Element support counts for one chromosome's peaks. Returns (n_tfs, width)."""
    order = np.argsort(pos, kind="mergesort")
    pos, tfi, sc = pos[order], tfi[order], sc[order]
    w = local_weights(pos, tfi)
    grid = np.zeros(chrom_len + 2 * KDE_BW + 1, np.float32)
    np.add.at(grid, pos, w)
    dens = gaussian_filter1d(grid, KDE_BW, mode="constant")
    del grid
    out_n, out_w = [], []
    for (a, b, _pk, _h) in _detect(dens, KDE_BW, DETECT_SUPPORT,
                                   BOUNDARY_FRAC, MIN_PEAK_DIST_BP):
        i0 = np.searchsorted(pos, a, "left")
        i1 = np.searchsorted(pos, b, "right")
        if i1 <= i0:
            continue
        tt, vv = tfi[i0:i1], sc[i0:i1]
        if np.unique(tt).size < DETECT_SUPPORT:
            continue
        out_n.append(int(np.unique(tt[vv >= MIN_SCORE_ASSIGN]).size))
        out_w.append(int(b - a + 1))
    del dens
    return np.asarray(out_n), np.asarray(out_w)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", default="20,21,22")
    ap.add_argument("--shuffles", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--target-fdr", type=float, default=0.05)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]

    out_dn = analysis_dir("supportnull", chr="-".join(chroms),
                          shuf=args.shuffles)
    if out_dn.exists() and any(out_dn.iterdir()) and not args.force:
        raise SystemExit(f"{out_dn} exists and is not empty; --force to replace")
    out_dn.mkdir(parents=True, exist_ok=True)

    # ---- load peaks (reuses genome_modules' reader) -----------------------
    from multiprocessing import Pool
    from config import discover_tf_files
    keep = {c: i for i, c in enumerate(chroms)}
    tf_files = discover_tf_files()
    _log(f"loading chr{','.join(chroms)} peaks across {len(tf_files):,} TFs")
    t0 = time.time()
    jobs = [(i, paths, keep) for i, (_s, paths) in enumerate(tf_files)]
    am, at, asc = [], [], []
    with Pool(args.workers) as pool:
        for ti, m, s in pool.imap_unordered(_gm._read_tf, jobs, chunksize=4):
            if len(m):
                am.append(m); asc.append(s)
                at.append(np.full(len(m), ti, np.int32))
    mid = np.concatenate(am); tfi = np.concatenate(at); sc = np.concatenate(asc)
    del am, at, asc
    _log(f"  {len(mid):,} peaks in {(time.time()-t0)/60:.1f} min")

    rng = np.random.default_rng(args.seed)
    obs_n, obs_w, obs_c = [], [], []
    null_n = {i: [] for i in range(args.shuffles)}

    for ci, cname in enumerate(chroms):
        m0 = ci << 40
        sel = np.flatnonzero((mid >= m0) & (mid < m0 + (1 << 40)))
        pos = (mid[sel] - m0).astype(np.int64)
        t_, v_ = tfi[sel], sc[sel]
        L = int(pos.max()) + 1
        n, w = detect_on(pos, t_, v_, L)
        obs_n.append(n); obs_w.append(w); obs_c.append(np.full(len(n), ci))
        _log(f"  chr{cname}: observed {len(n):,} elements")

        for s in range(args.shuffles):
            # independent circular shift PER TF: preserves each TF's peak count
            # and its own clustering, destroys cross-TF co-occurrence
            # Group by TF with one stable sort. The naive `t_ == ti` scan per
            # TF is O(n_TF * n_peaks) -- 1,793 x 22M here, per shuffle, per
            # chromosome. Same trap as the weight loop in genome_modules.
            shifted = pos.copy()
            o = np.argsort(t_, kind="stable")
            ts = t_[o]
            bnd = np.flatnonzero(np.diff(ts)) + 1
            offs = rng.integers(0, L, size=len(bnd) + 1)
            for gi, (s0, e0) in enumerate(zip(np.concatenate(([0], bnd)),
                                              np.concatenate((bnd, [len(ts)])))):
                idx = o[s0:e0]
                shifted[idx] = (pos[idx] + offs[gi]) % L
            nn, _ = detect_on(shifted, t_, v_, L)
            null_n[s].append(nn)
            _log(f"    chr{cname} shuffle {s}: {len(nn):,} null elements")

    obs_n = np.concatenate(obs_n); obs_w = np.concatenate(obs_w)
    obs_c = np.concatenate(obs_c)
    nulls = [np.concatenate(v) for v in null_n.values()]

    # ---- FDR by support floor --------------------------------------------
    floors = list(range(2, 41))
    rows = []
    for f in floors:
        o = int((obs_n >= f).sum())
        ns = np.array([int((x >= f).sum()) for x in nulls], float)
        rows.append(dict(floor=f, observed=o, null_mean=ns.mean(),
                         null_sd=ns.std(ddof=1) if len(ns) > 1 else 0.0,
                         fdr=(ns.mean() / o) if o else np.nan))
    fdr = pd.DataFrame(rows)
    hit = fdr[fdr.fdr <= args.target_fdr]
    chosen = int(hit.floor.min()) if len(hit) else None
    fdr.to_csv(out_dn / "support_fdr.tsv", sep="\t", index=False,
               float_format="%.6f")
    pd.DataFrame(dict(n_tfs_assigned=obs_n, width=obs_w,
                      chrom=[chroms[i] for i in obs_c])).to_csv(
        out_dn / "observed_support.tsv", sep="\t", index=False)

    print()
    _log(f"=== support floor vs FDR ({args.shuffles} circular-shift shuffles) ===")
    print(f"{'floor':>6}{'observed':>11}{'null mean':>12}{'null sd':>10}{'FDR':>9}")
    for _, r in fdr.iterrows():
        if r.floor > 25 and r.floor % 5:
            continue
        mark = "  <-- first <= target" if chosen == r.floor else ""
        print(f"{int(r.floor):>6}{int(r.observed):>11,}{r.null_mean:>12,.0f}"
              f"{r.null_sd:>10,.0f}{r.fdr:>9.3f}{mark}")
    print(f"\n  target FDR {args.target_fdr:.0%}  ->  "
          f"{'MIN_SUPPORT = ' + str(chosen) if chosen else 'NOT REACHED in 2..40'}")

    write_analysis_readme(
        out_dn,
        title=f"Support-floor calibration by circular-shift null "
              f"({', '.join('chr'+c for c in chroms)})",
        rationale=(
            "MIN_SUPPORT=2 was calibrated for dense promoter windows and is far "
            "too permissive genome-wide: 82% of discovered elements are distal "
            "with a median of exactly 2 assigned TFs, the floor itself.\n\n"
            "An elbow was considered and rejected -- the retention curve decays "
            "smoothly (66/51/41/34/30/23/19/13/10/7% at floors 2..30), so any "
            "cut point is a judgement defensible one notch either side.\n\n"
            "Instead each TF's peaks are circularly shifted by an independent "
            "random offset per chromosome and detection re-run. That preserves "
            "each TF's peak count and its own clustering while destroying "
            "co-occurrence between TFs, which is the null of interest. The "
            "support floor is then read off a target FDR.\n\n"
            "**Limitation:** the shifts are per-TF independent, so the null does "
            "not model co-binding driven by shared chromatin accessibility. "
            "Elements passing are non-random given each TF's own distribution; "
            "that is weaker than functional."),
        params={"shuffles": args.shuffles, "seed": args.seed,
                "target_fdr": args.target_fdr,
                "detect_support (during detection)": DETECT_SUPPORT,
                "min_score_assign": MIN_SCORE_ASSIGN, "kde_bw": KDE_BW,
                "nbhd_bp": NBHD_BP, "chroms": ",".join(chroms)},
        inputs={"tier": TIER, "tf_set": TF_SET, "peaks": int(len(mid))},
        stats={"observed elements": int(len(obs_n)),
               "null elements (mean)": f"{np.mean([len(x) for x in nulls]):,.0f}",
               f"chosen MIN_SUPPORT at FDR<={args.target_fdr:.0%}":
                   chosen if chosen else "not reached"})
    print(f"  wrote {out_dn}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
