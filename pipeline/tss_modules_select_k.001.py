#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Algorithmic NMF rank (k) selection for the tss_modules occupancy matrix.

Two stability measures, computed for k in KS:

    A) ARI stability (full matrix, fast)
       For each k, run NMF with N_SEEDS random initializations. Take dominant
       program (argmax W) per module per run. Compute pairwise Adjusted Rand
       Index across runs; report median + IQR.

    B) Brunet consensus + cophenetic correlation (5k-module subsample)
       Subsample SUBSAMPLE modules. For each k, run NMF N_SEEDS times. Build
       consensus matrix C[i,j] = fraction of runs where modules i, j land in
       the same dominant program. Hierarchically cluster (average linkage on
       1 - C); compute cophenetic correlation coefficient. Higher = more
       stable. The largest k that holds high cophenetic is the data-preferred
       rank.

Outputs (tss_modules/k_selection/):
    ari_stability.tsv         (k, median_ari, p25, p75, mean_ari)
    cophenetic.tsv            (k, cophenetic_corr)
    plots/
        ari_vs_k.{png,pdf}
        cophenetic_vs_k.{png,pdf}
        scree_vs_k.{png,pdf}     # combined view with reconstruction error
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS",      "4")
os.environ.setdefault("MKL_NUM_THREADS",      "4")

import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
from scipy.cluster.hierarchy import linkage, cophenet
from scipy.spatial.distance import squareform

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN


################################################################################
# Initiating Variables #########################################################
################################################################################
ROOT       = OUT_DN / "tss_modules"
NPZ_FN     = ROOT / "occupancy.modules.npz"
SEL_DN     = ROOT / "k_selection"
PLOTS_DN   = SEL_DN / "plots"

KS         = [5, 8, 10, 12, 15, 18, 20, 25]
N_SEEDS    = 20            # runs per k
SUBSAMPLE  = 5000          # modules sampled for the cophenetic stage
NMF_MAX_ITER = 250         # slightly relaxed for many fits
RANDOM_STATE = 7           # reproducibility for the subsample draw

sns.set_style("whitegrid")
plt.rcParams["font.size"]      = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["figure.dpi"]     = 100


################################################################################
# Helpers ######################################################################
################################################################################
def _ts():
    return dt.datetime.now().strftime("%H:%M:%S")


def fit_nmf_dominant(M, k: int, seed: int):
    """Fit NMF, return (dominant_program_per_row, frobenius_err)."""
    # init stays "random". This metric asks how consistently independent starts
    # converge on the same partition, so the starts must genuinely differ.
    #
    # The multiplicative-update solver can drive a component to zero from a
    # random start -- 1 of 20 seeds at k=18 on the 1,793-TF matrix collapsed to a
    # SINGLE component -- which yields reconstruction_err_=nan and an ARI of
    # exactly 0.000 against any healthy partition. Those were solver failures
    # being scored as instability. The caller detects and re-seeds them; see
    # fit_nmf_stable.
    #
    # Two alternatives were measured at k=18 (20 seeds) and rejected:
    #   nndsvdar+mu : no collapses, but ARI rises to 0.819 here and 0.98 at low
    #                 k -- every seed starts from nearly the same NNDSVD point,
    #                 so the metric measures init determinism, not the data.
    #   random+cd   : no collapses, but ARI halves to 0.233, changing the scale
    #                 of the statistic and breaking comparison with the
    #                 published selection.
    # Re-seeding keeps the median intact (0.420 -> 0.425) while lifting the
    # minimum off the floor (0.000 -> 0.308): it removes the artifact and
    # nothing else.
    nmf = NMF(n_components=k, init="random",
              max_iter=NMF_MAX_ITER, random_state=seed,
              beta_loss="frobenius", solver="mu", tol=1e-4)
    W = nmf.fit_transform(M)
    dom = W.argmax(axis=1).astype(np.int32)
    err = float(nmf.reconstruction_err_)
    # A fit where some component never wins a single row is degenerate even when
    # the error is finite, so check both.
    collapsed = (not np.isfinite(err)) or (len(np.unique(dom)) < k)
    return dom, err, collapsed


def fit_nmf_stable(M, k: int, seed: int, max_extra: int = 50):
    """fit_nmf_dominant, re-seeding past collapsed fits.

    Returns (dom, err, n_retries). Seeds beyond the requested one are drawn from
    a disjoint high range so a retry can never duplicate another slot's seed and
    silently turn two independent fits into one.
    """
    dom, err, collapsed = fit_nmf_dominant(M, k, seed)
    if not collapsed:
        return dom, err, 0
    for i in range(max_extra):
        alt = 100_000 + seed * max_extra + i
        dom, err, collapsed = fit_nmf_dominant(M, k, alt)
        if not collapsed:
            return dom, err, i + 1
    raise RuntimeError(f"k={k} seed={seed}: NMF collapsed on {max_extra} retries")


def median_pairwise_ari(label_matrix: np.ndarray):
    """label_matrix shape: (n_seeds, n_samples)."""
    n_seeds = label_matrix.shape[0]
    aris = []
    for i in range(n_seeds):
        for j in range(i + 1, n_seeds):
            aris.append(adjusted_rand_score(label_matrix[i], label_matrix[j]))
    aris = np.array(aris, dtype=np.float64)
    return aris


def cophenetic_from_consensus(C: np.ndarray) -> float:
    """C: square consensus matrix in [0, 1]. Returns cophenetic correlation
    of average-linkage hierarchical clustering of (1 - C)."""
    D = 1.0 - C
    np.fill_diagonal(D, 0.0)
    # Force symmetry & non-negativity for numerical safety
    D = 0.5 * (D + D.T)
    D = np.clip(D, 0, 1)
    cond = squareform(D, checks=False)
    Z = linkage(cond, method="average")
    coph_corr, _ = cophenet(Z, cond)
    return float(coph_corr)


################################################################################
# Execution ####################################################################
################################################################################
def main():
    SEL_DN.mkdir(parents=True, exist_ok=True)
    PLOTS_DN.mkdir(parents=True, exist_ok=True)

    print(f"[{_ts()}] loading {NPZ_FN.name}")
    M = sp.load_npz(str(NPZ_FN)).tocsr()
    n_module, n_tf = M.shape
    print(f"[{_ts()}]   M shape = {M.shape}, nnz = {M.nnz:,}")

    # -------------------------------------------------------------------------
    # Stage A: ARI stability on full matrix
    # -------------------------------------------------------------------------
    print(f"\n[{_ts()}] stage A: ARI stability on full matrix "
          f"(n={n_module}, seeds={N_SEEDS})")
    ari_rows = []
    err_rows = []
    for k in KS:
        t0 = time.time()
        labels = np.empty((N_SEEDS, n_module), dtype=np.int32)
        errs   = np.empty(N_SEEDS, dtype=np.float64)
        n_retry = 0
        for s in range(N_SEEDS):
            dom, err, r = fit_nmf_stable(M, k, seed=s)
            labels[s] = dom
            errs[s]   = err
            n_retry  += r
        if n_retry:
            # Report it: a rank needing many re-seeds is itself a signal that the
            # factorisation is marginal there, which a clean ARI would hide.
            print(f"[{_ts()}]   k={k}: {n_retry} collapsed fit(s) re-seeded")
        aris = median_pairwise_ari(labels)
        ari_rows.append({
            "k":         k,
            "n_seeds":   N_SEEDS,
            "median_ari": float(np.median(aris)),
            "mean_ari":  float(np.mean(aris)),
            "p25":       float(np.quantile(aris, 0.25)),
            "p75":       float(np.quantile(aris, 0.75)),
            "min_ari":   float(np.min(aris)),
            "max_ari":   float(np.max(aris)),
        })
        err_rows.append({
            "k":             k,
            "mean_err":      float(np.mean(errs)),
            "min_err":       float(np.min(errs)),
            "max_err":       float(np.max(errs)),
            "std_err":       float(np.std(errs, ddof=1)),
        })
        print(f"[{_ts()}]   k={k:2d}  median_ARI={ari_rows[-1]['median_ari']:.3f}  "
              f"IQR=[{ari_rows[-1]['p25']:.3f}, {ari_rows[-1]['p75']:.3f}]  "
              f"err={err_rows[-1]['mean_err']:.2f} ± {err_rows[-1]['std_err']:.2f}  "
              f"({time.time() - t0:.1f}s)")
    pd.DataFrame(ari_rows).to_csv(SEL_DN / "ari_stability.tsv",
                                   sep="\t", index=False)
    pd.DataFrame(err_rows).to_csv(SEL_DN / "scree.tsv",
                                   sep="\t", index=False)

    # -------------------------------------------------------------------------
    # Stage B: Brunet consensus + cophenetic, on a SUBSAMPLE of modules
    # -------------------------------------------------------------------------
    rng = np.random.default_rng(RANDOM_STATE)
    sub_idx = np.sort(rng.choice(n_module, size=min(SUBSAMPLE, n_module),
                                  replace=False))
    M_sub = M[sub_idx]
    print(f"\n[{_ts()}] stage B: cophenetic on subsample "
          f"(n={M_sub.shape[0]}, seeds={N_SEEDS})")

    coph_rows = []
    for k in KS:
        t0 = time.time()
        labels = np.empty((N_SEEDS, M_sub.shape[0]), dtype=np.int32)
        for s in range(N_SEEDS):
            # Same re-seeding as stage A: a collapsed fit would put every
            # subsampled module in one cluster and inflate the consensus matrix,
            # making the cophenetic correlation look BETTER the worse the fit is.
            dom, _, _ = fit_nmf_stable(M_sub, k, seed=s)
            labels[s] = dom

        # Build consensus matrix in chunks to stay memory-friendly.
        # C[i,j] = mean over seeds of (labels[s,i] == labels[s,j]).
        n = M_sub.shape[0]
        C = np.zeros((n, n), dtype=np.float32)
        for s in range(N_SEEDS):
            lab = labels[s]
            # one-hot on lab, then C += A @ A.T row-by-row would be huge;
            # use bincount-by-cluster trick: for each cluster c, find indices
            # in cluster c, set those (i,j) to 1 in a temporary indicator.
            for c in range(int(lab.max()) + 1):
                idx = np.flatnonzero(lab == c)
                if idx.size:
                    C[np.ix_(idx, idx)] += 1.0
        C /= N_SEEDS

        coph = cophenetic_from_consensus(C)
        # Also dispersion coefficient (Brunet alt; higher = more decisive)
        dispersion = float(np.mean(4.0 * (C - 0.5) ** 2))
        coph_rows.append({
            "k":             k,
            "n_seeds":       N_SEEDS,
            "n_subsample":   int(n),
            "cophenetic":    coph,
            "dispersion":    dispersion,
        })
        print(f"[{_ts()}]   k={k:2d}  cophenetic={coph:.4f}  "
              f"dispersion={dispersion:.4f}  ({time.time() - t0:.1f}s)")

    pd.DataFrame(coph_rows).to_csv(SEL_DN / "cophenetic.tsv",
                                    sep="\t", index=False)

    # -------------------------------------------------------------------------
    # Plots
    # -------------------------------------------------------------------------
    ari_df  = pd.DataFrame(ari_rows)
    err_df  = pd.DataFrame(err_rows)
    coph_df = pd.DataFrame(coph_rows)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ari_df["k"], ari_df["median_ari"],
                yerr=[ari_df["median_ari"] - ari_df["p25"],
                      ari_df["p75"] - ari_df["median_ari"]],
                fmt="-o", color="C0", capsize=3,
                label="median ARI (IQR)")
    ax.set_xlabel("k (n_components)")
    ax.set_ylabel("Pairwise Adjusted Rand Index across seeds")
    ax.set_title(f"NMF stability vs k  (full matrix, {N_SEEDS} seeds)")
    ax.set_xticks(ari_df["k"])
    ax.set_ylim(0, 1)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DN / "ari_vs_k.png", dpi=300)
    fig.savefig(PLOTS_DN / "ari_vs_k.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(coph_df["k"], coph_df["cophenetic"], "-o",
            color="C2", label="cophenetic ρ")
    ax.plot(coph_df["k"], coph_df["dispersion"], "-s",
            color="C3", label="dispersion")
    ax.set_xlabel("k (n_components)")
    ax.set_ylabel("Stability metric")
    ax.set_title(f"NMF cophenetic + dispersion vs k  "
                 f"(subsample n={SUBSAMPLE}, {N_SEEDS} seeds)")
    ax.set_xticks(coph_df["k"])
    ax.set_ylim(0, 1.02)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DN / "cophenetic_vs_k.png", dpi=300)
    fig.savefig(PLOTS_DN / "cophenetic_vs_k.pdf")
    plt.close(fig)

    # Combined scree
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(err_df["k"], err_df["mean_err"], yerr=err_df["std_err"],
                fmt="-o", color="C3", capsize=3, label="mean Frobenius err")
    ax.set_xlabel("k (n_components)")
    ax.set_ylabel("Reconstruction error (Frobenius)")
    ax.set_title(f"NMF reconstruction error vs k  ({N_SEEDS} seeds)")
    ax.set_xticks(err_df["k"])
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(PLOTS_DN / "scree_vs_k.png", dpi=300)
    fig.savefig(PLOTS_DN / "scree_vs_k.pdf")
    plt.close(fig)

    # Combined panel for the report
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].errorbar(ari_df["k"], ari_df["median_ari"],
                     yerr=[ari_df["median_ari"] - ari_df["p25"],
                           ari_df["p75"] - ari_df["median_ari"]],
                     fmt="-o", color="C0", capsize=3)
    axes[0].set_title("ARI stability")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("median pairwise ARI")
    axes[0].set_xticks(ari_df["k"]); axes[0].set_ylim(0, 1)
    axes[1].plot(coph_df["k"], coph_df["cophenetic"], "-o", color="C2",
                 label="cophenetic")
    axes[1].plot(coph_df["k"], coph_df["dispersion"], "-s", color="C3",
                 label="dispersion")
    axes[1].set_title("Brunet cophenetic + dispersion")
    axes[1].set_xlabel("k"); axes[1].set_ylabel("stability")
    axes[1].set_xticks(coph_df["k"]); axes[1].set_ylim(0, 1.02)
    axes[1].legend()
    axes[2].errorbar(err_df["k"], err_df["mean_err"], yerr=err_df["std_err"],
                     fmt="-o", color="C3", capsize=3)
    axes[2].set_title("Reconstruction error (scree)")
    axes[2].set_xlabel("k"); axes[2].set_ylabel("mean Frobenius err")
    axes[2].set_xticks(err_df["k"])
    fig.suptitle("NMF rank-selection diagnostics", y=1.02)
    fig.tight_layout()
    fig.savefig(PLOTS_DN / "k_selection_combined.png", dpi=300,
                bbox_inches="tight")
    fig.savefig(PLOTS_DN / "k_selection_combined.pdf",
                bbox_inches="tight")
    plt.close(fig)

    print(f"\n[{_ts()}] DONE")


if __name__ == "__main__":
    main()
