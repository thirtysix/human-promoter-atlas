#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Gene-level archetypes from per-gene program-presence vectors.

Builds an [n_gene x 10] count matrix where M[g, p] = number of modules at
gene g whose dominant k=10 program is p. NMF decomposes this matrix into
A archetypes — recurring patterns of program *composition* across the
genome. Each gene gets assigned a dominant archetype.

Algorithmic K selection: 20-seed pairwise ARI stability across A in
{4,5,6,7,8,9}. Pick the A with peak median ARI, then run a final NMF.

Outputs (tss_archetypes/):
    occupancy.gene_x_program.npz     # CSR sparse [n_gene x 10]
    gene_index.tsv
    ari_stability.tsv
    nmf.A{A}.{W,H}.tsv.gz            # one set per A
    nmf.A{A}.archetype_summary.tsv
    nmf.A{A}.gene_archetype.tsv      # gene -> archetype + weights
    plots/
        ari_vs_A.{png,pdf}
        archetype_program_loadings.A{A}.{png,pdf}
        archetype_sizes.A{A}.{png,pdf}
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS",      "4")
os.environ.setdefault("MKL_NUM_THREADS",      "4")

import datetime as dt
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN


################################################################################
# Initiating Variables #########################################################
################################################################################
ROOT                = OUT_DN   # config.OUT_DN
MOD_DN     = ROOT / "tss_modules"
OUT_DN     = ROOT / "tss_archetypes"
PLOTS_DN   = OUT_DN / "plots"
K_PROGRAMS = 10
A_CANDIDATES = [4, 5, 6, 7, 8, 9]
N_SEEDS    = 20
NMF_MAX_ITER = 300

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11


################################################################################
# Helpers ######################################################################
################################################################################
def _ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}")


def fit_nmf(M, k: int, seed: int):
    nmf = NMF(n_components=k, init="random", solver="mu",
              beta_loss="frobenius", max_iter=NMF_MAX_ITER,
              random_state=seed, tol=1e-4)
    W = nmf.fit_transform(M)
    return W, nmf.components_, float(nmf.reconstruction_err_)


################################################################################
# Execution ####################################################################
################################################################################
def main():
    OUT_DN.mkdir(parents=True, exist_ok=True)
    PLOTS_DN.mkdir(parents=True, exist_ok=True)

    # 1) Build [n_gene x K_PROGRAMS] count matrix from module_program (k=10)
    _log("loading module_program (k=10)…")
    mp = pd.read_csv(MOD_DN / f"nmf.k{K_PROGRAMS}.module_program.tsv",
                      sep="\t", usecols=["transcript_id", "gene_name",
                                          "dominant_program"])
    counts = (mp.groupby(["transcript_id", "gene_name", "dominant_program"])
                .size()
                .unstack(fill_value=0))
    # Ensure all 10 program columns present, in order
    for p in range(1, K_PROGRAMS + 1):
        if p not in counts.columns:
            counts[p] = 0
    counts = counts[[p for p in range(1, K_PROGRAMS + 1)]]
    counts = counts.reset_index()
    n_gene = len(counts)
    _log(f"  n_gene = {n_gene:,}  (transcripts with >=1 module)")

    M_dense = counts[[p for p in range(1, K_PROGRAMS + 1)]].to_numpy(np.float32)
    M = sp.csr_matrix(M_dense)
    _log(f"  M shape = {M.shape}  nnz = {M.nnz:,}")

    sp.save_npz(str(OUT_DN / "occupancy.gene_x_program.npz"), M)
    counts[["transcript_id", "gene_name"]].to_csv(
        OUT_DN / "gene_index.tsv", sep="\t", index=False,
        index_label="gene_idx")

    # 2) ARI stability across A candidates
    _log(f"ARI stability across A in {A_CANDIDATES} ({N_SEEDS} seeds each)")
    ari_rows = []
    for A in A_CANDIDATES:
        t0 = time.time()
        labels = np.empty((N_SEEDS, n_gene), dtype=np.int32)
        errs = np.empty(N_SEEDS, dtype=np.float64)
        for s in range(N_SEEDS):
            W, H, err = fit_nmf(M, A, seed=s)
            labels[s] = W.argmax(axis=1)
            errs[s] = err
        aris = []
        for i in range(N_SEEDS):
            for j in range(i + 1, N_SEEDS):
                aris.append(adjusted_rand_score(labels[i], labels[j]))
        ari_rows.append({
            "A": A, "n_seeds": N_SEEDS,
            "median_ari": float(np.median(aris)),
            "mean_ari":  float(np.mean(aris)),
            "p25":       float(np.quantile(aris, 0.25)),
            "p75":       float(np.quantile(aris, 0.75)),
            "mean_err":  float(np.mean(errs)),
        })
        _log(f"  A={A}  median_ARI={ari_rows[-1]['median_ari']:.3f}  "
             f"IQR=[{ari_rows[-1]['p25']:.3f}, {ari_rows[-1]['p75']:.3f}]  "
             f"err={ari_rows[-1]['mean_err']:.2f}  "
             f"({time.time() - t0:.1f}s)")

    ari_df = pd.DataFrame(ari_rows)
    ari_df.to_csv(OUT_DN / "ari_stability.tsv", sep="\t", index=False)

    # ARI vs A plot
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.errorbar(ari_df["A"], ari_df["median_ari"],
                yerr=[ari_df["median_ari"] - ari_df["p25"],
                      ari_df["p75"] - ari_df["median_ari"]],
                fmt="-o", capsize=3, color="C0")
    ax.set_xlabel("A (n_archetypes)")
    ax.set_ylabel("median pairwise ARI across seeds")
    ax.set_title("Gene-archetype rank stability")
    ax.set_xticks(ari_df["A"]); ax.set_ylim(0, 1)
    fig.tight_layout()
    fig.savefig(PLOTS_DN / "ari_vs_A.png", dpi=300)
    fig.savefig(PLOTS_DN / "ari_vs_A.pdf")
    plt.close(fig)

    A_canonical = int(ari_df.loc[ari_df["median_ari"].idxmax(), "A"])
    _log(f"canonical A = {A_canonical} (peak median ARI)")

    # 3) Final NMF + tables for every A in A_CANDIDATES (for browsing)
    program_names = [f"P{p}" for p in range(1, K_PROGRAMS + 1)]
    for A in A_CANDIDATES:
        W, H, err = fit_nmf(M, A, seed=0)
        # Order archetypes by size (dominant assignment count)
        dom = W.argmax(axis=1)
        order = np.argsort(-np.bincount(dom, minlength=A))
        W = W[:, order]
        H = H[order, :]
        dom = W.argmax(axis=1)

        # Save W and H
        pd.DataFrame(
            W,
            index=counts["transcript_id"].values,
            columns=[f"A{a+1}" for a in range(A)],
        ).to_csv(OUT_DN / f"nmf.A{A}.W.tsv.gz", sep="\t",
                 compression="gzip", index_label="transcript_id")
        pd.DataFrame(
            H,
            index=[f"A{a+1}" for a in range(A)],
            columns=program_names,
        ).to_csv(OUT_DN / f"nmf.A{A}.H.tsv.gz", sep="\t",
                 compression="gzip", index_label="archetype")

        # Per-archetype summary: n_genes, top programs, mean modules/gene
        rows = []
        n_mod_per_gene = M_dense.sum(axis=1)
        for a in range(A):
            mask = (dom == a)
            n_genes = int(mask.sum())
            mean_mod = float(n_mod_per_gene[mask].mean()) if n_genes else 0.0
            top_idx = np.argsort(H[a])[::-1][:5]
            top_programs = ",".join(program_names[i] for i in top_idx)
            top_loadings = ",".join(f"{H[a, i]:.2f}" for i in top_idx)
            rows.append({
                "archetype":   a + 1,
                "n_genes":     n_genes,
                "frac_genes":  round(n_genes / max(n_gene, 1), 4),
                "mean_modules_per_gene": round(mean_mod, 2),
                "top_programs":          top_programs,
                "top_loadings":          top_loadings,
            })
        summary = pd.DataFrame(rows).sort_values("n_genes", ascending=False)
        summary.to_csv(OUT_DN / f"nmf.A{A}.archetype_summary.tsv",
                        sep="\t", index=False)

        # Per-gene archetype + soft weights (all archetypes)
        row_sum = W.sum(axis=1, keepdims=True)
        W_norm = W / np.where(row_sum > 0, row_sum, 1.0)
        ga = pd.DataFrame({
            "transcript_id":      counts["transcript_id"].values,
            "gene_name":          counts["gene_name"].values,
            "n_modules":          n_mod_per_gene.astype(int),
            "dominant_archetype": dom + 1,
            "dominant_weight":    W_norm[np.arange(n_gene), dom],
        })
        for a in range(A):
            ga[f"A{a+1}_w"] = W_norm[:, a]
        ga.to_csv(OUT_DN / f"nmf.A{A}.gene_archetype.tsv",
                  sep="\t", index=False)

        # Plots: program loadings heatmap (A x 10)
        fig, ax = plt.subplots(figsize=(6, 0.5 * A + 1.5))
        im = ax.imshow(H, aspect="auto", cmap="magma",
                        vmin=0, vmax=float(np.quantile(H, 0.99) or 1.0),
                        interpolation="nearest")
        ax.set_xticks(range(K_PROGRAMS))
        ax.set_xticklabels(program_names, fontsize=10)
        ax.set_yticks(range(A))
        ax.set_yticklabels([f"A{a+1}" for a in range(A)], fontsize=10)
        ax.set_xlabel("k=10 program")
        ax.set_ylabel("archetype")
        ax.set_title(f"Archetype × program loadings (A={A})")
        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("H loading")
        for a in range(A):
            for p in range(K_PROGRAMS):
                ax.text(p, a, f"{H[a, p]:.1f}", ha="center", va="center",
                         fontsize=8,
                         color="white" if H[a, p] > H.mean() else "black")
        fig.tight_layout()
        fig.savefig(PLOTS_DN / f"archetype_program_loadings.A{A}.png", dpi=300)
        fig.savefig(PLOTS_DN / f"archetype_program_loadings.A{A}.pdf")
        plt.close(fig)

        # Sizes bar
        fig, ax = plt.subplots(figsize=(max(5, 0.6 * A + 2), 4))
        ax.bar(np.arange(A) + 1, np.bincount(dom, minlength=A),
                color="steelblue", edgecolor="black", linewidth=0.5)
        ax.set_xlabel("archetype")
        ax.set_ylabel("# genes (dominant)")
        ax.set_title(f"Archetype sizes (A={A}, n_genes={n_gene:,})")
        ax.set_xticks(np.arange(A) + 1)
        for i, c in enumerate(np.bincount(dom, minlength=A)):
            ax.text(i + 1, c, f"{c}", ha="center", va="bottom", fontsize=9)
        fig.tight_layout()
        fig.savefig(PLOTS_DN / f"archetype_sizes.A{A}.png", dpi=300)
        fig.savefig(PLOTS_DN / f"archetype_sizes.A{A}.pdf")
        plt.close(fig)

        _log(f"  A={A}  err={err:.2f}  n_archetypes={A}")
        for _, r in summary.iterrows():
            _log(f"    A{int(r['archetype']):d}  n_genes={int(r['n_genes']):5d}  "
                 f"mean_mod={r['mean_modules_per_gene']:.2f}  "
                 f"top: {r['top_programs']}  H: [{r['top_loadings']}]")

    # Note canonical A in a small marker file for the build_app_db script
    (OUT_DN / "canonical_A.txt").write_text(str(A_canonical))
    _log(f"DONE — canonical A = {A_canonical}")


if __name__ == "__main__":
    main()
