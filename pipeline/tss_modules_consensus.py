#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Multi-seed NMF at the canonical rank, with per-program reproducibility.

Replaces the single pinned-seed fit (``tss_modules_k10.py``) for new builds.
That script fits seed 0 and writes the result as *the* answer; nothing records
how much of it survives a different random start. On this data that omission
matters -- per-program reproducibility ranges from 0.61 to 0.99 within one fit,
and a program at 0.61 is presented identically to one at 0.99.

Two changes:

representative fit
    N seeds are fitted and the MEDOID is written out -- the seed whose programs
    best match all the others. A pinned seed is an arbitrary draw; the medoid is
    the one most typical of the ensemble. It is a real fit, so W and H stay
    mutually consistent (averaging H across seeds would need matching first and
    blurs programs that genuinely differ).

per-program stability
    For each program in the medoid, the best-match cosine against every other
    seed's programs. The median of those is the program's stability. Written to
    ``nmf.k{K}.stability.tsv`` and intended to be shown next to the program
    wherever it is presented, not buried in a methods note.

Collapsed fits are detected and re-seeded (see ``nmf_fit``); a collapse scored
as data is how a solver failure gets reported as instability.

Outputs (tss_modules/), matching tss_modules_k10.py so downstream is unchanged:
    nmf.k{K}.{W,H}.tsv.gz
    nmf.k{K}.top_tfs.tsv
    nmf.k{K}.stability.tsv        <- new
    nmf.k{K}.consensus.json       <- new: seeds, medoid, collapses, provenance

Usage:
    python pipeline/tss_modules_consensus.py [--k 10] [--seeds 20] [--top-tfs 30]
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS",      "8")
os.environ.setdefault("MKL_NUM_THREADS",      "8")

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

# Machine-specific paths and build axes -> pipeline/config.py
import nmf_fit
from config import OUT_DN, TIER, TF_SET, MIN_SCORE_ASSIGN


################################################################################
# Initiating Variables #########################################################
################################################################################
ROOT         = OUT_DN / "tss_modules"
NMF_MAX_ITER = 300
COS_MIN      = 0.90        # a program counts as reproducible at or above this


def _log(msg: str):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _unit(A):
    A = np.asarray(A, float)
    return A / (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)


################################################################################
# Helpers ######################################################################
################################################################################
def medoid_index(Hs) -> int:
    """Index of the fit whose programs best match every other fit.

    Score is the mean over other fits of the mean best-match cosine, so a fit is
    penalised both for having an odd program and for missing a common one.
    """
    n = len(Hs)
    scores = [
        np.mean([(Hs[i] @ Hs[j].T).max(axis=1).mean() for j in range(n) if j != i])
        for i in range(n)
    ]
    return int(np.argmax(scores))


def program_stability(Hs, ref: int):
    """Per-program best-match cosine of the reference fit against all others.

    Returns (median, min, frac>=0.90, frac>=0.80), each length k.
    """
    per_seed = np.stack([(Hs[ref] @ Hs[j].T).max(axis=1)
                         for j in range(len(Hs)) if j != ref])
    return (np.median(per_seed, axis=0), per_seed.min(axis=0),
            (per_seed >= 0.90).mean(axis=0), (per_seed >= 0.80).mean(axis=0))


def relabel_by_size(W, H):
    """Programs ordered by how many modules they dominate -- same convention as
    tss_modules_k10.py, so program numbering stays comparable."""
    order = np.argsort(-np.bincount(W.argmax(axis=1), minlength=W.shape[1]))
    return W[:, order], H[order, :], order


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--top-tfs", type=int, default=30)
    args = ap.parse_args()
    K, N_SEEDS = args.k, args.seeds

    _log(f"build {OUT_DN.name}  (tier={TIER} tf_set={TF_SET} "
         f"min_score_assign={MIN_SCORE_ASSIGN})")
    M = sp.load_npz(str(ROOT / "occupancy.modules.npz")).tocsr()
    tf_names = (pd.read_csv(ROOT / "tf_index.tsv", sep="\t")
                  .sort_values("tf_idx")["TF"].tolist())
    modules_df = pd.read_csv(ROOT / "modules.tsv", sep="\t", low_memory=False)
    modules_in = modules_df[modules_df["n_tfs_assigned"].to_numpy() > 0].reset_index(drop=True)
    _log(f"  M={M.shape}  n_tf={len(tf_names)}  modules in matrix={len(modules_in):,}")
    if M.shape[0] != len(modules_in):
        raise SystemExit(
            f"occupancy rows ({M.shape[0]:,}) != modules with >=1 assigned TF "
            f"({len(modules_in):,}) -- the matrix and modules.tsv disagree")

    t0 = time.time()
    Ws, Hs, errs, retries_tot = [], [], [], 0
    for s in range(N_SEEDS):
        W, H, err, retries = nmf_fit.fit_nmf_stable(M, K, s, max_iter=NMF_MAX_ITER)
        Ws.append(W); Hs.append(_unit(H)); errs.append(err); retries_tot += retries
        _log(f"  seed {s:>2}  err={err:.2f}" + (f"  re-seeded {retries}x" if retries else ""))
    if retries_tot:
        _log(f"  {retries_tot} collapsed fit(s) re-seeded -- a collapse scored as "
             f"data would read as instability")

    ref = medoid_index(Hs)
    _log(f"medoid = seed {ref}  (most representative of the {N_SEEDS} fits)")
    med, mn, f90, f80 = program_stability(Hs, ref)

    # Refit is not needed: Ws[ref]/Hs[ref] are the medoid, but Hs was unit-normalised
    # for matching, so take H from a clean refit of the same seed.
    W, H, err, _ = nmf_fit.fit_nmf_stable(M, K, ref, max_iter=NMF_MAX_ITER)
    W, H, order = relabel_by_size(W, H)
    med, mn, f90, f80 = med[order], mn[order], f90[order], f80[order]
    prog = [f"prog{p+1}" for p in range(K)]

    pd.DataFrame(W, index=modules_in["module_id"].values, columns=prog).to_csv(
        ROOT / f"nmf.k{K}.W.tsv.gz", sep="\t", compression="gzip", index_label="module_id")
    pd.DataFrame(H, index=prog, columns=tf_names).to_csv(
        ROOT / f"nmf.k{K}.H.tsv.gz", sep="\t", compression="gzip", index_label="program")

    top = [{"program": p + 1, "rank": r + 1, "tf": tf_names[i], "loading": H[p, i]}
           for p in range(K)
           for r, i in enumerate(np.argsort(H[p])[::-1][:args.top_tfs])]
    pd.DataFrame(top).to_csv(ROOT / f"nmf.k{K}.top_tfs.tsv", sep="\t", index=False)

    # module_program / gene_configurations / summary, identical in form to
    # tss_modules_k10.py. Everything downstream (enrichment, archetypes,
    # build_app_db, compare_builds) reads these, so a stage that writes only
    # W/H leaves the build unusable while looking finished.
    row_sum = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(row_sum > 0, row_sum, 1.0)
    dom = W_norm.argmax(axis=1) + 1
    mp = pd.DataFrame({
        "module_id":        modules_in["module_id"].values,
        "tss_id":           modules_in["tss_id"].values,
        "gene_name":        modules_in["gene_name"].values,
        "transcript_id":    modules_in["transcript_id"].values,
        "center_offset":    modules_in["center_offset"].values,
        "width":            modules_in["width"].values,
        "dominant_program": dom,
        "dominant_weight":  W_norm[np.arange(W.shape[0]), dom - 1],
    })
    for p in range(K):
        mp[f"prog{p+1}_w"] = W_norm[:, p]
    mp.to_csv(ROOT / f"nmf.k{K}.module_program.tsv", sep="\t", index=False)

    (mp.sort_values(["transcript_id", "center_offset"])
       .groupby(["transcript_id", "gene_name"])
       .agg(n_modules=("module_id", "size"),
            program_path=("dominant_program", lambda s: ",".join(map(str, s))),
            centers=("center_offset", lambda s: ",".join(map(str, s))),
            widths=("width", lambda s: ",".join(map(str, s))))
       .reset_index()
       .to_csv(ROOT / f"nmf.k{K}.gene_configurations.tsv", sep="\t", index=False))

    srows = []
    for p in range(1, K + 1):
        sub = mp[mp["dominant_program"] == p]
        tops = list(pd.Series(H[p - 1], index=tf_names)
                      .sort_values(ascending=False).head(8).index)
        srows.append({
            "program": p, "n_modules": len(sub),
            "median_center": int(sub["center_offset"].median()) if len(sub) else 0,
            "median_width":  int(sub["width"].median()) if len(sub) else 0,
            "mean_dom_weight": round(float(sub["dominant_weight"].mean()), 4) if len(sub) else 0.0,
            "top_tfs": ",".join(tops), "reading": ", ".join(tops[:3]),
            # stability travels WITH the summary so anything reading the program
            # table gets the reproducibility alongside it, not from a side file.
            "median_cosine": round(float(med[p - 1]), 4),
            "reproducible": bool(med[p - 1] >= COS_MIN),
        })
    (pd.DataFrame(srows).sort_values("n_modules", ascending=False)
       .to_csv(ROOT / f"nmf.k{K}.summary.tsv", sep="\t", index=False))

    stab = pd.DataFrame({
        "program": range(1, K + 1),
        "median_cosine": med, "min_cosine": mn,
        "frac_seeds_ge_0.90": f90, "frac_seeds_ge_0.80": f80,
        "reproducible": med >= COS_MIN,
        "top_tfs": [",".join(np.array(tf_names)[np.argsort(H[p])[::-1][:6]]) for p in range(K)],
    })
    stab.to_csv(ROOT / f"nmf.k{K}.stability.tsv", sep="\t", index=False,
                float_format="%.6f")

    (ROOT / f"nmf.k{K}.consensus.json").write_text(json.dumps({
        "k": K, "n_seeds": N_SEEDS, "medoid_seed": ref,
        "collapsed_seeds_reseeded": retries_tot,
        "cos_min": COS_MIN,
        "n_reproducible": int((med >= COS_MIN).sum()),
        "reconstruction_err_medoid": err,
        "reconstruction_err_mean": float(np.mean(errs)),
        "max_iter": NMF_MAX_ITER,
        "tier": TIER, "tf_set": TF_SET, "min_score_assign": MIN_SCORE_ASSIGN,
    }, indent=2) + "\n")

    _log(f"reproducible programs: {(med >= COS_MIN).sum()}/{K} at cos>={COS_MIN}")
    for _, r in stab.iterrows():
        flag = "" if r["reproducible"] else "   <-- NOT reproducible"
        _log(f"  prog{int(r['program']):<3} median_cos={r['median_cosine']:.3f}  "
             f"{r['top_tfs'][:44]}{flag}")
    _log(f"DONE in {(time.time()-t0)/60:.1f} min -> {ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
