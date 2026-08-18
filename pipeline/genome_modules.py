#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Annotation-free regulatory element discovery, genome-wide.

The atlas scans +/-1500 bp around canonical TSSs, which sees roughly 18% of all
peaks and cannot represent an enhancer at all. This finds elements wherever they
are and annotates them by position afterwards, which is the standard framing
(ENCODE cCREs, ChromHMM-style segmentation).

Elements are defined by exactly the same rule as promoter modules -- this reuses
``_detect_modules_density`` from tss_modules.001.py rather than reimplementing
it, so a promoter-proximal element here should reproduce the module found there.
That correspondence is the regression check for this whole approach.

Per-TF normalisation without a window
------------------------------------
tss_modules weights each peak by 1/count(that TF in this window), so every TF
contributes equal mass to where boundaries fall. Genome-wide there is no window.
Normalising per chromosome is not the analogue -- it would give a TF with 200
peaks a per-peak weight 300,000x that of CTCF, and CTCF sites would vanish.

Gap-merging peaks into "candidate regions" was tried and rejected: measured on
chr20-22, a 1 kb gap yields regions covering 118.8 Mb of 162 Mb (73% of the
sequence) with a maximum span of 319,725 bp. At ~136 peaks/kb a 1 kb gap almost
never occurs, so dense areas fuse and per-region normalisation degenerates to
the global normalisation it was meant to avoid.

What is used instead is a LOCAL NEIGHBOURHOOD weight: each peak is weighted
1/(that TF's peaks within +/-NBHD_BP), NBHD_BP = 1500 to match the promoter
window half-width. A TF with five clustered peaks contributes total mass ~1 for
that cluster, exactly as it would inside one TSS window, and no merging step is
required.

Labels (for asking which programs are stratum-specific)
------------------------------------------------------
    dist_to_tss   signed bp to the nearest canonical TSS
    stratum       promoter <=1kb | proximal 1-10kb | distal >10kb
    intragenic    inside any protein-coding gene body
    cluster_id    elements within --cluster-gap of each other (super-enhancer
                  candidates are large clusters)

Beware when using these: distal elements are expected to carry fewer TFs, so a
"distal-specific program" may only be the program that captures low-complexity
elements. Any such claim must be re-checked on elements matched for
n_tfs_assigned.

Usage:
    python pipeline/genome_modules.py --chroms 20,21,22 --stage regions
    python pipeline/genome_modules.py --chroms 20,21,22
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
import scipy.sparse as sp
from scipy.ndimage import gaussian_filter1d

from config import (OUT_DN, MIN_SCORE_ASSIGN, TIER, TF_SET,
                    discover_tf_files, analysis_dir,
                    write_analysis_readme)
# Reused verbatim so elements and promoter modules are defined identically.
import importlib.util as _ilu
_spec = _ilu.spec_from_file_location(
    "_tssmod", str(Path(__file__).resolve().parent / "tss_modules.001.py"))
_tssmod = _ilu.module_from_spec(_spec)
_spec.loader.exec_module(_tssmod)
_detect_modules_density = _tssmod._detect_modules_density

KDE_BW           = 25
MIN_SUPPORT      = 2
BOUNDARY_FRAC    = 0.20
MIN_PEAK_DIST_BP = 50
RECENTER_HALF    = 12
PROMOTER_BP, PROXIMAL_BP = 1000, 10000
NBHD_BP          = 1500     # local per-TF normalisation half-width
                            # (= OUTER_HALF, the promoter window)


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _read_tf(args):
    """(tf_idx, midpoints, scores) for one TF restricted to the wanted chroms."""
    tf_idx, paths, keep = args
    mids, scs = [], []
    for p in paths:
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    f = line.split("\t", 5)
                    c = f[0][3:] if f[0].startswith("chr") else f[0]
                    if c not in keep:
                        continue
                    mid = (int(f[1]) + int(f[2])) // 2
                    # Recenter BEFORE packing. Rounding the packed value would
                    # mix the chromosome bits into the position: 2**40 is not a
                    # multiple of 25, so each chromosome would get a different
                    # grid phase and positions near 0 could go negative.
                    mid = (mid // (2 * RECENTER_HALF + 1)) * (2 * RECENTER_HALF + 1)
                    mids.append((keep[c] << 40) | mid)     # pack chrom + pos
                    scs.append(int(f[4]))
        except Exception:
            continue
    if not mids:
        return tf_idx, np.empty(0, np.int64), np.empty(0, np.int16)
    return (tf_idx, np.asarray(mids, np.int64), np.asarray(scs, np.int16))


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chroms", default="20,21,22")
    ap.add_argument("--tier", default=None, help="default: the configured tier")
    ap.add_argument("--region-gap", type=int, default=1000,
                    help="merge peak midpoints into a region when gap <= this")
    ap.add_argument("--cluster-gap", type=int, default=12500,
                    help="element clustering distance (super-enhancer scale)")
    ap.add_argument("--stage", choices=["regions", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing analysis directory")
    args = ap.parse_args()

    chroms = [c.strip() for c in args.chroms.split(",") if c.strip()]
    keep = {c: i for i, c in enumerate(chroms)}
    # Its OWN directory, named by every axis that defines it -- never inside
    # the promoter build it reads from, and never colliding with a run at
    # different parameters.
    out_dn = (Path(args.out) if args.out else
              analysis_dir("genome", sup=MIN_SUPPORT, nbhd=NBHD_BP,
                           chr="-".join(chroms)))
    if out_dn.exists() and any(out_dn.iterdir()) and not args.force:
        raise SystemExit(
            f"{out_dn} already exists and is not empty. Refusing to overwrite a "
            f"previous result.\n  Re-run with --force to replace it, or change a "
            f"parameter so it lands in its own directory.")
    out_dn.mkdir(parents=True, exist_ok=True)

    # ---- 1. load peaks for the wanted chromosomes ------------------------
    # [(symbol, [paths]), ...] -- resolves legacy antigen names to current
    # symbols and returns ALL files backing a symbol, since some factors are
    # served under both names as separate antigens with disjoint experiments.
    tf_files = discover_tf_files()
    tf_names = [sym for sym, _ in tf_files]
    _log(f"tier peaks for chroms {chroms} across {len(tf_names):,} TFs")
    t0 = time.time()
    jobs = [(i, paths, keep) for i, (_sym, paths) in enumerate(tf_files)]
    all_mid, all_tf, all_sc = [], [], []
    with Pool(args.workers) as pool:
        for n, (ti, mids, scs) in enumerate(
                pool.imap_unordered(_read_tf, jobs, chunksize=4), 1):
            if len(mids):
                all_mid.append(mids); all_sc.append(scs)
                all_tf.append(np.full(len(mids), ti, np.int32))
            if n % 400 == 0:
                _log(f"  {n}/{len(tf_names)} TFs ({time.time()-t0:.0f}s)")
    mid = np.concatenate(all_mid); tfi = np.concatenate(all_tf)
    sc = np.concatenate(all_sc)
    del all_mid, all_tf, all_sc
    _log(f"  {len(mid):,} peaks loaded in {(time.time()-t0)/60:.1f} min")

    # already recentered per peak in _read_tf; just sort
    order = np.argsort(mid, kind="mergesort")
    mid, tfi, sc = mid[order], tfi[order], sc[order]

    # ---- 2. local per-TF weights ----------------------------------------
    # weight = 1 / (peaks of the SAME TF within +/-NBHD_BP), computed per TF.
    # This is the windowless form of tss_modules.001.py:316-326.
    _log(f"=== local per-TF normalisation (+/-{NBHD_BP} bp) ===")
    t1 = time.time()
    # Group by TF via one stable sort rather than a boolean scan per TF: the
    # naive `tfi == ti` loop is O(n_TF * n_peaks) = 1,793 x 22M here.
    # mid is already globally sorted, and a stable sort on tfi preserves that
    # order within each TF block, so each block's positions stay ascending and
    # searchsorted is valid.
    w = np.empty(len(mid), np.float32)
    ordr = np.argsort(tfi, kind="stable")
    tf_s = tfi[ordr]
    bounds = np.flatnonzero(np.diff(tf_s)) + 1
    for s, e in zip(np.concatenate(([0], bounds)),
                    np.concatenate((bounds, [len(tf_s)]))):
        idx = ordr[s:e]
        q = mid[idx]
        lo = np.searchsorted(q, q - NBHD_BP, "left")
        hi = np.searchsorted(q, q + NBHD_BP, "right")
        w[idx] = 1.0 / (hi - lo)
    _log(f"  weights in {time.time()-t1:.0f}s   "
         f"mean {w.mean():.4f}  min {w.min():.2e}")
    if args.stage == "regions":
        _log("  --stage regions is obsolete (gap-merging was rejected); "
             "run without --stage")
        return 0

    # ---- 3. per-chromosome density and element detection -----------------
    _log("=== element detection ===")
    t1 = time.time()
    rows, occ_r, occ_c = [], [], []
    eid = 0
    for ci, cname in enumerate(chroms):
        m0 = (ci << 40)
        sel = np.flatnonzero((mid >= m0) & (mid < m0 + (1 << 40)))
        if not len(sel):
            continue
        pos = (mid[sel] - m0).astype(np.int64)
        t_, v_, w_ = tfi[sel], sc[sel], w[sel]
        n = int(pos.max()) + 2 * KDE_BW + 1
        grid = np.zeros(n, np.float32)
        np.add.at(grid, pos, w_)
        dens = gaussian_filter1d(grid, KDE_BW, mode="constant")
        del grid
        found = _detect_modules_density(dens, KDE_BW, MIN_SUPPORT,
                                        BOUNDARY_FRAC, MIN_PEAK_DIST_BP)
        _log(f"  chr{cname}: {len(sel):,} peaks, {n/1e6:.1f} Mb grid, "
             f"{len(found):,} candidate elements ({time.time()-t1:.0f}s)")
        for (a, b, pk, h) in found:
            i0, i1 = np.searchsorted(pos, a, "left"), np.searchsorted(pos, b, "right")
            if i1 <= i0:
                continue
            tt, vv = t_[i0:i1], v_[i0:i1]
            supp = np.unique(tt)
            if supp.size < MIN_SUPPORT:
                continue
            assigned = np.unique(tt[vv >= MIN_SCORE_ASSIGN])
            rows.append((eid, cname, int(a), int(b), int(pk), int(b - a + 1),
                         float(h), int(i1 - i0), int(supp.size),
                         int(assigned.size)))
            occ_r.extend([eid] * assigned.size); occ_c.extend(assigned.tolist())
            eid += 1
        del dens
    _log(f"  {eid:,} elements retained in {(time.time()-t1)/60:.1f} min")

    el = pd.DataFrame(rows, columns=["element_id", "chrom", "start", "end",
                                     "peak", "width", "kde_height",
                                     "n_peaks_in", "n_tfs_supporting",
                                     "n_tfs_assigned"])

    M = sp.coo_matrix((np.ones(len(occ_r), np.float32), (occ_r, occ_c)),
                      shape=(eid, len(tf_names))).tocsr()
    M.sum_duplicates()
    sp.save_npz(str(out_dn / "occupancy.elements.npz"), M)
    pd.DataFrame({"TF": tf_names, "tf_idx": range(len(tf_names))}).to_csv(
        out_dn / "tf_index.tsv", sep="\t", index=False)

    # ---- 4. label --------------------------------------------------------
    _log("=== labelling ===")
    tss = pd.read_csv(OUT_DN / "tss_modules" / "tss_table.tsv", sep="\t")
    by_c = {str(c): np.sort(g["tss"].to_numpy()) for c, g in tss.groupby("chrom")}
    d = np.full(len(el), 1 << 30, np.int64)
    for c, sub in el.groupby("chrom"):
        arr = by_c.get(str(c))
        if arr is None or not len(arr):
            continue
        p = sub["peak"].to_numpy()
        j = np.searchsorted(arr, p)
        cand = np.stack([arr[np.clip(j - 1, 0, len(arr) - 1)],
                         arr[np.clip(j, 0, len(arr) - 1)]])
        best = cand[np.argmin(np.abs(cand - p), axis=0), np.arange(len(p))]
        d[sub.index] = p - best
    el["dist_to_tss"] = d
    ad = np.abs(d)
    el["stratum"] = np.where(ad <= PROMOTER_BP, "promoter",
                     np.where(ad <= PROXIMAL_BP, "proximal", "distal"))
    # element clusters (super-enhancer candidates)
    cid = np.zeros(len(el), np.int32); nxt = 0
    for c, sub in el.sort_values(["chrom", "start"]).groupby("chrom"):
        pk = sub["peak"].to_numpy()
        new = np.concatenate(([True], np.diff(pk) > args.cluster_gap))
        cid[sub.index] = nxt + np.cumsum(new) - 1
        nxt += int(new.sum())
    el["cluster_id"] = cid
    el["cluster_size"] = el.groupby("cluster_id")["element_id"].transform("size")
    el.to_csv(out_dn / "elements.tsv", sep="\t", index=False)

    print()
    _log("=== summary ===")
    print(el.groupby("stratum")
            .agg(n=("element_id", "size"), med_width=("width", "median"),
                 med_tfs=("n_tfs_assigned", "median"))
            .to_string())
    n_se = int((el.drop_duplicates("cluster_id")["cluster_size"] >= 5).sum())
    print(f"\n  clusters >= 5 elements (super-enhancer candidates): {n_se:,}")

    by = el.groupby("stratum")["n_tfs_assigned"].median().to_dict()
    write_analysis_readme(
        out_dn,
        title=f"Genome-wide element discovery ({', '.join('chr'+c for c in chroms)})",
        rationale=(
            "The promoter atlas scans +/-1500 bp around canonical TSSs, which "
            "covers ~18% of peaks and cannot represent an enhancer at all. This "
            "discovers elements without reference to genes, then labels them by "
            "distance to the nearest canonical TSS, so promoter/proximal/distal "
            "programs can be compared.\n\n"
            "Elements use the SAME detection rule as promoter modules "
            "(`_detect_modules_density` is imported from tss_modules.001.py, not "
            "reimplemented), so promoter-stratum elements should reproduce the "
            "modules found by the promoter pipeline. That correspondence is the "
            "regression check.\n\n"
            "**Read the median TF counts below before trusting any distal "
            "program.** MIN_SUPPORT was calibrated for dense promoter windows; "
            "if distal elements sit at the support floor they are two-TF "
            "coincidences, and an NMF on them will produce 'distal-specific' "
            "programs that are artifacts of sparsity."),
        params={"min_support": MIN_SUPPORT, "min_score_assign": MIN_SCORE_ASSIGN,
                "kde_bw": KDE_BW, "boundary_frac": BOUNDARY_FRAC,
                "min_peak_dist_bp": MIN_PEAK_DIST_BP,
                "nbhd_bp (per-TF local weight)": NBHD_BP,
                "cluster_gap": args.cluster_gap, "chroms": ",".join(chroms)},
        inputs={"tier": TIER, "tf_set": TF_SET, "n_tf": len(tf_names),
                "peaks_loaded": int(len(mid))},
        stats={"elements": int(len(el)),
               "promoter / proximal / distal":
                   " / ".join(str(int(v)) for v in
                              el["stratum"].value_counts()
                                .reindex(["promoter", "proximal", "distal"])
                                .fillna(0)),
               "median n_tfs_assigned by stratum":
                   " / ".join(f"{k}={by.get(k, float('nan')):.0f}"
                              for k in ("promoter", "proximal", "distal")),
               "super-enhancer candidate clusters (>=5)": n_se})
    print(f"  wrote {out_dn}/  (elements.tsv, occupancy.elements.npz, README.md)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
