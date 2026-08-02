#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
One-shot NMF at k=10 (algorithmically-preferred rank from
tss_modules_select_k.001.py: ARI=0.553, cophenetic=0.930). Loads the existing
occupancy.modules.npz + modules.tsv and emits all the standard outputs for k=10
(W, H, summary, top_tfs, module_program, gene_configurations) plus plots.
"""

import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS",      "4")
os.environ.setdefault("MKL_NUM_THREADS",      "4")

import numpy as np
import pandas as pd
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from sklearn.decomposition import NMF

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN


ROOT    = OUT_DN / "tss_modules"
PLOTS   = ROOT / "plots"
TF_CL_FN = OUT_DN / "clustering" / "tf_cluster_table.tsv"
K       = 10
TOP_TFS = 30
SEED    = 0
OUTER_HALF = 1500

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 11


def relabel_by_size(W, H):
    dom = W.argmax(axis=1)
    counts = np.bincount(dom, minlength=W.shape[1])
    order = np.argsort(-counts)
    return W[:, order], H[order, :]


def main():
    print(f"loading {ROOT}/occupancy.modules.npz")
    M = sp.load_npz(str(ROOT / "occupancy.modules.npz")).tocsr()
    tf_names = pd.read_csv(ROOT / "tf_index.tsv", sep="\t").sort_values("tf_idx")["TF"].tolist()
    modules_df = pd.read_csv(ROOT / "modules.tsv", sep="\t")
    keep = modules_df["n_tfs_assigned"].to_numpy() > 0
    modules_in = modules_df[keep].reset_index(drop=True)
    print(f"  M={M.shape}  n_tf={len(tf_names)}  modules in matrix={len(modules_in)}")

    print(f"NMF k={K}")
    nmf = NMF(n_components=K, init="random", max_iter=300,
              random_state=SEED, beta_loss="frobenius",
              solver="mu", tol=1e-4)
    W = nmf.fit_transform(M)
    H = nmf.components_
    print(f"  err={nmf.reconstruction_err_:.2f}")
    W, H = relabel_by_size(W, H)

    # Save W / H
    pd.DataFrame(W,
                 index=modules_in["module_id"].values,
                 columns=[f"prog{p+1}" for p in range(K)]
    ).to_csv(ROOT / f"nmf.k{K}.W.tsv.gz", sep="\t",
             compression="gzip", index_label="module_id")
    pd.DataFrame(H,
                 index=[f"prog{p+1}" for p in range(K)],
                 columns=tf_names
    ).to_csv(ROOT / f"nmf.k{K}.H.tsv.gz", sep="\t",
             compression="gzip", index_label="program")

    # Top TFs
    top_rows = []
    for p in range(K):
        idx = np.argsort(H[p])[::-1][:TOP_TFS]
        for rank, j in enumerate(idx, 1):
            top_rows.append({"program": p + 1, "rank": rank,
                             "tf": tf_names[j], "loading": float(H[p, j])})
    pd.DataFrame(top_rows).to_csv(ROOT / f"nmf.k{K}.top_tfs.tsv",
                                   sep="\t", index=False)

    # Module → program
    row_sum = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(row_sum > 0, row_sum, 1.0)
    dom = W_norm.argmax(axis=1) + 1
    mp_df = pd.DataFrame({
        "module_id":         modules_in["module_id"].values,
        "tss_id":            modules_in["tss_id"].values,
        "gene_name":         modules_in["gene_name"].values,
        "transcript_id":     modules_in["transcript_id"].values,
        "center_offset":     modules_in["center_offset"].values,
        "width":             modules_in["width"].values,
        "dominant_program":  dom,
        "dominant_weight":   W_norm[np.arange(W.shape[0]), dom - 1],
    })
    for p in range(K):
        mp_df[f"prog{p+1}_w"] = W_norm[:, p]
    mp_df.to_csv(ROOT / f"nmf.k{K}.module_program.tsv",
                 sep="\t", index=False)

    # Gene configurations
    gene_conf = (mp_df
                 .sort_values(["transcript_id", "center_offset"])
                 .groupby(["transcript_id", "gene_name"])
                 .agg(n_modules=("module_id", "size"),
                      program_path=("dominant_program",
                                    lambda s: ",".join(map(str, s))),
                      centers=("center_offset",
                               lambda s: ",".join(map(str, s))),
                      widths=("width",
                              lambda s: ",".join(map(str, s))))
                 .reset_index())
    gene_conf.to_csv(ROOT / f"nmf.k{K}.gene_configurations.tsv",
                     sep="\t", index=False)

    # Summary
    rows = []
    for p in range(1, K + 1):
        sub = mp_df[mp_df["dominant_program"] == p]
        n_dom = len(sub)
        med_c = int(sub["center_offset"].median()) if n_dom else 0
        med_w = int(sub["width"].median())         if n_dom else 0
        mean_w = float(sub["dominant_weight"].mean()) if n_dom else 0.0
        row = pd.Series(H[p - 1], index=tf_names)
        top = list(row.sort_values(ascending=False).head(8).index)
        rows.append({"program": p, "n_modules": n_dom,
                     "median_center": med_c, "median_width": med_w,
                     "mean_dom_weight": round(mean_w, 4),
                     "top_tfs": ",".join(top),
                     "reading": ", ".join(top[:3])})
    summary_df = pd.DataFrame(rows).sort_values("n_modules", ascending=False)
    summary_df.to_csv(ROOT / f"nmf.k{K}.summary.tsv", sep="\t", index=False)
    print(summary_df.to_string(index=False))

    # ---- Plots ----
    # H heatmap
    top_set = set()
    for p in range(K):
        top_set.update(np.argsort(H[p])[::-1][:TOP_TFS].tolist())
    cols = sorted(top_set)
    H_sub = H[:, cols]
    dom_h = H_sub.argmax(axis=0)
    dom_load = H_sub[dom_h, np.arange(H_sub.shape[1])]
    order = np.lexsort((-dom_load, dom_h))
    H_ord = H_sub[:, order]
    tf_ord = [tf_names[cols[i]] for i in order]
    fig, ax = plt.subplots(figsize=(max(10, 0.10 * len(tf_ord) + 2), 0.32 * K + 1.2))
    vmax = float(np.quantile(H_ord, 0.99)) if H_ord.size else 1.0
    im = ax.imshow(H_ord, aspect="auto", cmap="magma",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(K)); ax.set_yticklabels([f"P{p+1}" for p in range(K)])
    ax.set_xticks(range(len(tf_ord)))
    ax.set_xticklabels(tf_ord, rotation=90, fontsize=7)
    ax.set_title(f"NMF program × TF loadings (k={K})")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01).set_label("H loading")
    fig.tight_layout()
    fig.savefig(PLOTS / f"program_tf_loadings.k{K}.png", dpi=300)
    fig.savefig(PLOTS / f"program_tf_loadings.k{K}.pdf")
    plt.close(fig)

    # Position density per program
    bins = np.linspace(-OUTER_HALF, OUTER_HALF, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    fig, axes = plt.subplots(K, 1, figsize=(8, 1.4 * K + 1.0),
                              sharex=True, squeeze=False)
    for p in range(1, K + 1):
        ax = axes[p - 1, 0]
        sub = mp_df[mp_df["dominant_program"] == p]
        h, _ = np.histogram(sub["center_offset"].values, bins=bins)
        ax.fill_between(centers, h, color=f"C{(p-1) % 10}", alpha=0.85)
        ax.plot(centers, h, color="black", linewidth=0.4)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.8)
        ax.set_ylabel(f"P{p}", rotation=0, ha="right", va="center",
                      fontsize=11, fontweight="bold")
        ax.set_yticks([])
        ax.text(0.99, 0.85, f"n={len(sub):,}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color="dimgray")
    axes[-1, 0].set_xlabel("module center, bp from TSS (txn-oriented)")
    fig.suptitle(f"Module center positions per program (k={K})", y=0.995)
    fig.tight_layout()
    fig.savefig(PLOTS / f"program_position_density.k{K}.png",
                dpi=300, bbox_inches="tight")
    fig.savefig(PLOTS / f"program_position_density.k{K}.pdf",
                bbox_inches="tight")
    plt.close(fig)

    # Program sizes
    counts = np.bincount(dom - 1, minlength=K)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * K + 2), 4))
    ax.bar(np.arange(K) + 1, counts, color="steelblue",
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Program"); ax.set_ylabel("Modules dominant")
    ax.set_title(f"Program sizes (k={K}, n_modules={W.shape[0]:,})")
    ax.set_xticks(np.arange(K) + 1)
    for i, c in enumerate(counts):
        ax.text(i + 1, c, f"{c}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / f"program_sizes.k{K}.png", dpi=300)
    fig.savefig(PLOTS / f"program_sizes.k{K}.pdf")
    plt.close(fig)

    # vs filtered K8 TF clusters
    if TF_CL_FN.exists():
        tab = pd.read_csv(TF_CL_FN, sep="\t")
        tf_to_cluster = dict(zip(tab["TF"], tab["cluster"]))
        clusters = sorted(set(tab["cluster"]))
        Mc = np.zeros((K, len(clusters)), dtype=np.int32)
        for p in range(K):
            for j in np.argsort(H[p])[::-1][:TOP_TFS]:
                c = tf_to_cluster.get(tf_names[j])
                if c is not None:
                    Mc[p, clusters.index(c)] += 1
        fig, ax = plt.subplots(figsize=(0.6 * len(clusters) + 4, 0.4 * K + 2))
        im = ax.imshow(Mc, aspect="auto", cmap="Blues", interpolation="nearest")
        ax.set_xticks(range(len(clusters)))
        ax.set_xticklabels([f"TFc{c}" for c in clusters])
        ax.set_yticks(range(K))
        ax.set_yticklabels([f"P{p+1}" for p in range(K)])
        ax.set_xlabel("Filtered-K8 TF cluster")
        ax.set_ylabel("Program")
        ax.set_title(f"Program top-{TOP_TFS} TFs by TF cluster (k={K})")
        if Mc.max():
            for p in range(K):
                for c in range(len(clusters)):
                    if Mc[p, c] > 0:
                        ax.text(c, p, str(Mc[p, c]), ha="center", va="center",
                                color="black" if Mc[p, c] < Mc.max() / 2 else "white",
                                fontsize=8)
        fig.colorbar(im, ax=ax, fraction=0.025,
                     pad=0.02).set_label(f"# of top-{TOP_TFS} TFs in cluster")
        fig.tight_layout()
        fig.savefig(PLOTS / f"program_vs_tfcluster.k{K}.png", dpi=300)
        fig.savefig(PLOTS / f"program_vs_tfcluster.k{K}.pdf")
        plt.close(fig)
    print("DONE")


if __name__ == "__main__":
    main()
