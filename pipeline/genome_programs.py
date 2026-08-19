#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Consensus NMF over genome-wide elements, and which strata each program favours.

The promoter equivalent is tss_modules_consensus.py. Its seed/medoid/stability
machinery is imported rather than reimplemented, so a program here is defined
exactly as a program there. What differs is the output side: elements are not
indexed by gene, so there are no gene_configurations, and there IS a stratum
label to ask questions of.

Memory. tss_modules_consensus keeps every seed's W to pick the medoid. At
117,006 x 18 that is nothing; at 467,223 x 125 each W is 467 MB and twenty of
them is 9.3 GB held for no purpose -- the medoid is chosen on H alone and the
winning seed is refitted afterwards regardless. Only H is retained here.

THE CONFOUND, which any claim from this output must survive
-----------------------------------------------------------
Distal elements carry fewer assigned TFs than promoter elements (medians 21 vs
48 in the sup11 build). A program that loads on sparse elements will therefore
look "distal-specific" whether or not it has anything to do with distal
biology. Raw stratum fractions cannot tell those apart.

So enrichment is reported twice: raw, and stratified by n_tfs_assigned into
bins and recombined (a Cochran-Mantel-Haenszel style adjustment). A program
whose distal enrichment survives the matched version is enriched for distal
elements; one whose enrichment disappears was only ever enriched for sparse
ones. The two columns are printed side by side so the difference cannot be
overlooked.

Usage:
    python pipeline/genome_programs.py --genome-dir <dir> --k 125 --seeds 20
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
import scipy.sparse as sp

import nmf_fit
from config import TIER, TF_SET, MIN_SCORE_ASSIGN, write_analysis_readme
from tss_modules_consensus import (medoid_index, program_stability,
                                   relabel_by_size, _unit, NMF_MAX_ITER)

STRATA = ["promoter", "proximal", "distal"]
# Complexity bins for the matched enrichment. Edges are quantiles of
# n_tfs_assigned so each bin carries a comparable number of elements.
N_COMPLEXITY_BINS = 8


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def stratum_enrichment(dom, el, k):
    """Raw and complexity-matched log2 enrichment of each program per stratum.

    Matched: within each n_tfs_assigned bin, compare the program's stratum rate
    to that bin's background rate, then average over bins weighted by how many
    of the program's elements fall in each. That removes the sparsity gradient
    which would otherwise manufacture distal-specific programs.
    """
    strat = el["stratum"].to_numpy()
    bg = {s: float((strat == s).mean()) for s in STRATA}
    qs = np.unique(np.quantile(el["n_tfs_assigned"],
                               np.linspace(0, 1, N_COMPLEXITY_BINS + 1)))
    binid = np.clip(np.searchsorted(qs, el["n_tfs_assigned"], "right") - 1,
                    0, len(qs) - 2)
    # background stratum rate within each complexity bin
    bg_bin = {s: np.array([float((strat[binid == b] == s).mean())
                           if (binid == b).any() else np.nan
                           for b in range(len(qs) - 1)]) for s in STRATA}
    rows = []
    for p in range(k):
        sel = dom == p
        n = int(sel.sum())
        r = {"program": p + 1, "n_elements": n,
             "median_n_tfs": float(np.median(el["n_tfs_assigned"][sel])) if n else np.nan}
        for s in STRATA:
            obs = float((strat[sel] == s).mean()) if n else np.nan
            r[f"{s}_frac"] = obs
            r[f"{s}_log2FE"] = (np.log2((obs + 1e-9) / (bg[s] + 1e-9))
                                if n else np.nan)
            # matched: weight each bin by the program's own occupancy of it
            if n:
                w = np.array([float(((binid == b) & sel).sum())
                              for b in range(len(qs) - 1)])
                ob = np.array([float((strat[(binid == b) & sel] == s).mean())
                               if ((binid == b) & sel).any() else np.nan
                               for b in range(len(qs) - 1)])
                ok = (~np.isnan(ob)) & (~np.isnan(bg_bin[s])) & (w > 0)
                r[f"{s}_log2FE_matched"] = (
                    float(np.average(np.log2((ob[ok] + 1e-9) /
                                             (bg_bin[s][ok] + 1e-9)),
                                     weights=w[ok])) if ok.any() else np.nan)
            else:
                r[f"{s}_log2FE_matched"] = np.nan
        rows.append(r)
    return pd.DataFrame(rows), bg


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--seeds", type=int, default=20)
    ap.add_argument("--top-tfs", type=int, default=30)
    args = ap.parse_args()
    K, N_SEEDS = args.k, args.seeds
    root = Path(args.genome_dir)

    M = sp.load_npz(str(root / "occupancy.elements.npz")).tocsr()
    tf_names = (pd.read_csv(root / "tf_index.tsv", sep="\t")
                  .sort_values("tf_idx")["TF"].tolist())
    el = pd.read_csv(root / "elements.tsv", sep="\t", dtype={"chrom": str})
    _log(f"M={M.shape}  n_tf={len(tf_names)}  elements={len(el):,}")
    if M.shape[0] != len(el):
        raise SystemExit(f"occupancy rows ({M.shape[0]:,}) != elements "
                         f"({len(el):,}) -- matrix and elements.tsv disagree")

    t0 = time.time()
    Hs, errs, retries_tot = [], [], 0
    for s in range(N_SEEDS):
        # W is discarded: the medoid is chosen on H, and the winning seed is
        # refitted below. Keeping 20 of them would be 9.3 GB for nothing.
        _W, H, err, retries = nmf_fit.fit_nmf_stable(M, K, s,
                                                     max_iter=NMF_MAX_ITER)
        del _W
        Hs.append(_unit(H)); errs.append(err); retries_tot += retries
        _log(f"  seed {s:>2}  err={err:.2f}"
             + (f"  re-seeded {retries}x" if retries else "")
             + f"  ({time.time()-t0:.0f}s)")
    if retries_tot:
        _log(f"  {retries_tot} collapsed fit(s) re-seeded -- a collapse scored "
             f"as data would read as instability")

    ref = medoid_index(Hs)
    _log(f"medoid = seed {ref} of {N_SEEDS}")
    med, mn, f90, f80 = program_stability(Hs, ref)
    W, H, err, _ = nmf_fit.fit_nmf_stable(M, K, ref, max_iter=NMF_MAX_ITER)
    W, H, order = relabel_by_size(W, H)
    med, mn, f90, f80 = med[order], mn[order], f90[order], f80[order]
    prog = [f"prog{p+1}" for p in range(K)]

    # W is 467k x 125; .npz not a gzipped TSV, which would be ~500 MB of text.
    np.savez_compressed(root / f"nmf.k{K}.W.npz", W=W.astype(np.float32),
                        element_id=el["element_id"].values)
    pd.DataFrame(H, index=prog, columns=tf_names).to_csv(
        root / f"nmf.k{K}.H.tsv.gz", sep="\t", compression="gzip",
        index_label="program")
    pd.DataFrame([{"program": p + 1, "rank": r + 1, "tf": tf_names[i],
                   "loading": H[p, i]}
                  for p in range(K)
                  for r, i in enumerate(np.argsort(H[p])[::-1][:args.top_tfs])]
                 ).to_csv(root / f"nmf.k{K}.top_tfs.tsv", sep="\t", index=False)

    row_sum = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(row_sum > 0, row_sum, 1.0)
    dom = W_norm.argmax(axis=1)
    pd.DataFrame({"element_id": el["element_id"].values,
                  "chrom": el["chrom"].values,
                  "stratum": el["stratum"].values,
                  "n_tfs_assigned": el["n_tfs_assigned"].values,
                  "dominant_program": dom + 1,
                  "dominant_weight": W_norm[np.arange(len(dom)), dom]}
                 ).to_csv(root / f"nmf.k{K}.element_program.tsv.gz", sep="\t",
                          index=False, compression="gzip")

    stab = pd.DataFrame({"program": np.arange(1, K + 1), "median_cosine": med,
                         "min_cosine": mn, "frac_ge_0.90": f90,
                         "frac_ge_0.80": f80,
                         "reproducible": med >= 0.90})
    stab.to_csv(root / f"nmf.k{K}.stability.tsv", sep="\t", index=False,
                float_format="%.6f")

    enr, bg = stratum_enrichment(dom, el, K)
    summary = stab.merge(enr, on="program")
    summary["top_tfs"] = [", ".join(tf_names[i] for i in
                                    np.argsort(H[p])[::-1][:6])
                          for p in range(K)]
    summary.to_csv(root / f"nmf.k{K}.summary.tsv", sep="\t", index=False,
                   float_format="%.6f")

    n_repro = int(stab.reproducible.sum())
    print()
    _log(f"=== k={K}: {n_repro}/{K} programs reproducible (median cosine >= 0.90) ===")
    print(f"  background strata: "
          + "  ".join(f"{s} {bg[s]:.3f}" for s in STRATA))
    print(f"\n{'prog':>5}{'stab':>7}{'nElem':>9}{'medTF':>7}"
          f"{'distFE':>8}{'distFEm':>9}{'promFE':>8}{'promFEm':>9}  top TFs")
    for _, r in summary.sort_values("distal_log2FE_matched",
                                    ascending=False).head(20).iterrows():
        print(f"{int(r['program']):>5}{r['median_cosine']:>7.3f}"
              f"{int(r['n_elements']):>9,}{r['median_n_tfs']:>7.0f}"
              f"{r['distal_log2FE']:>8.2f}{r['distal_log2FE_matched']:>9.2f}"
              f"{r['promoter_log2FE']:>8.2f}{r['promoter_log2FE_matched']:>9.2f}"
              f"  {str(r['top_tfs'])[:34]}")

    readme_dn = root / f"programs.k{K}"
    readme_dn.mkdir(parents=True, exist_ok=True)
    write_analysis_readme(
        readme_dn,
        title=f"Genome-wide programs at k={K}",
        rationale=(
            "Consensus NMF over annotation-free elements, using the same "
            "seed/medoid/stability definition as the promoter build "
            "(tss_modules_consensus.py is imported, not reimplemented).\n\n"
            "**Distal enrichment must be read from the MATCHED column.** Distal "
            "elements carry fewer assigned TFs than promoter elements (medians "
            "21 vs 48), so a program loading on sparse elements looks "
            "distal-specific regardless of biology. The matched columns "
            "stratify by n_tfs_assigned and recombine, so an enrichment that "
            "survives them is about position and one that does not was about "
            "sparsity."),
        params={"k": K, "seeds": N_SEEDS, "nmf_max_iter": NMF_MAX_ITER,
                "complexity_bins": N_COMPLEXITY_BINS,
                "collapsed_refits": retries_tot},
        inputs={"genome_dir": str(root), "tier": TIER, "tf_set": TF_SET,
                "min_score_assign": MIN_SCORE_ASSIGN,
                "elements": int(len(el)), "n_tf": len(tf_names)},
        stats={"programs": K, "reproducible (median cosine >= 0.90)": n_repro,
               "medoid seed": int(ref),
               "median stability": f"{float(stab.median_cosine.median()):.3f}"})
    _log(f"wrote {root}/nmf.k{K}.*")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
