#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Cluster TFs by their TSS-proximal binding shape.

Consumes matrices produced by canonical_promoter_aggregate.001.py:
    matrices/tf_x_position.binary.parquet  (rows = TF, cols = -1000..+1000,
                                            values = fraction of canonical TSSs
                                            bound at that offset, in [0, 1])

Pipeline:
    1. Filter low-signal TFs (max binding < SIGNAL_FLOOR).
    2. Light Gaussian smoothing (sigma = SMOOTH_SIGMA bp).
    3. Peak-height normalize each TF row (row.max -> 1) — clustering is on SHAPE,
       independent of absolute binding magnitude.
    4. Correlation-distance matrix (1 - Pearson r) across TFs.
    5. Hierarchical clustering (Ward linkage, Euclidean distance) → cut at K_CLUSTERS.
    6. Plots: clustermap, per-cluster mean profile small-multiples, UMAP scatter.
    7. Outputs: tf_cluster_assignments.tsv, cluster_mean_profiles.tsv.
"""

################################################################################
# Libraries ####################################################################
################################################################################
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.distance import pdist, squareform
from scipy.cluster.hierarchy import linkage, fcluster, dendrogram
import umap

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN


################################################################################
# Initiating Variables #########################################################
################################################################################
INPUT_MATRIX    = "matrices/tf_x_position.binary.parquet"      # for clustering
HIST_MATRIX     = "matrices/tf_x_position.raw_score1000.parquet"  # for per-cluster histograms

HALF            = 1000
LEN             = 2 * HALF + 1
HIST_BIN_WIDTH  = 5
HIST_TITLE_TFS  = 20

SIGNAL_FLOOR    = 0.0        # NO/LOW filtering: keep every TF with any
                             #   nonzero signal at any offset. (Strict zero
                             #   rows still drop, otherwise peak-normalize
                             #   would divide by zero.)
SMOOTH_SIGMA    = 10         # Gaussian smoothing sigma in bp
K_CLUSTERS      = 12         # bumped from 8 — many more TFs in scope
LINKAGE_METHOD  = "ward"     # average / ward / complete
DISTANCE        = "euclidean"     # ward requires euclidean; on peak-normalized
                                  # rows this is shape-driven and gives more
                                  # balanced clusters than corr+average
RANDOM_STATE    = 42

HIGHLIGHT_TFS   = ["CTCF", "YY1", "MYC", "SP1", "NRF1", "REST", "TBP", "EP300"]

sns.set_style("whitegrid")
plt.rcParams["font.size"]      = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["figure.dpi"]     = 100


################################################################################
# Functions ####################################################################
################################################################################
def load_matrix(out_dn: str, input_matrix: str) -> pd.DataFrame:
    df = pd.read_parquet(Path(out_dn) / input_matrix).set_index("TF")
    df.columns = df.columns.astype(int)
    return df


def filter_and_smooth(matrix: pd.DataFrame, signal_floor: float,
                      smooth_sigma: float) -> tuple:
    """Return (smoothed_filtered_matrix, dropped_tfs)."""
    row_max = matrix.max(axis=1)
    # signal_floor == 0 means "no/low filtering" — keep any TF with strictly
    # positive max (zero-rows can't be peak-normalized).
    keep    = row_max > signal_floor if signal_floor == 0.0 else row_max >= signal_floor
    dropped = matrix.index[~keep].tolist()
    M = matrix.loc[keep].copy()

    smoothed = gaussian_filter1d(M.values, sigma=smooth_sigma, axis=1)
    M = pd.DataFrame(smoothed, index=M.index, columns=M.columns)
    return M, dropped


def peak_normalize(M: pd.DataFrame) -> pd.DataFrame:
    rmax = M.max(axis=1).replace(0, np.nan)
    return M.div(rmax, axis=0).fillna(0)


def cluster_tfs(M_norm: pd.DataFrame, distance: str, linkage_method: str,
                k: int) -> tuple:
    """Returns (labels Series indexed by TF, linkage Z, condensed_dist_array)."""
    D = pdist(M_norm.values, metric=distance)
    D = np.nan_to_num(D, nan=1.0)            # zero-variance rows → max distance
    Z = linkage(D, method=linkage_method)
    labels = fcluster(Z, t=k, criterion="maxclust")
    return pd.Series(labels, index=M_norm.index, name="cluster"), Z, D


def relabel_by_argmax(M_norm: pd.DataFrame, labels: pd.Series) -> pd.Series:
    """
    Reorder cluster IDs so that cluster 1 has the most-upstream peak center
    and cluster K has the most-downstream peak center. Stable, interpretable.
    """
    cols = M_norm.columns.values
    centers = {}
    for c in sorted(labels.unique()):
        members = labels[labels == c].index
        mean_profile = M_norm.loc[members].mean(axis=0).values
        centers[c] = float(cols[np.argmax(mean_profile)])
    order = sorted(centers, key=lambda c: centers[c])
    remap = {old: new for new, old in enumerate(order, start=1)}
    return labels.map(remap)


def cluster_mean_profiles(M: pd.DataFrame, labels: pd.Series) -> pd.DataFrame:
    """Per-cluster mean of the (smoothed, NOT row-normalized) profile."""
    rows = []
    for c in sorted(labels.unique()):
        members = labels[labels == c].index
        rows.append({
            "cluster": c,
            "n_tfs":   len(members),
            **{int(col): float(M.loc[members, col].mean()) for col in M.columns},
        })
    return pd.DataFrame(rows)


# ---- Plotting ---------------------------------------------------------------
def plot_clustermap(M_norm: pd.DataFrame, labels: pd.Series, Z: np.ndarray,
                    out_path_stem: str):
    palette = sns.color_palette("tab10", n_colors=labels.max())
    row_colors = labels.map(lambda c: palette[c - 1])

    g = sns.clustermap(
        M_norm, row_linkage=Z, col_cluster=False,
        cmap="viridis", row_colors=row_colors,
        figsize=(9, max(6, 0.008 * len(M_norm))),
        xticklabels=False, yticklabels=False,
        cbar_kws={"label": "Peak-normalized binding"},
        rasterized=True,
    )
    g.ax_heatmap.set_xlabel("Distance from TSS (bp)")
    xt = [-1000, -500, 0, 500, 1000]
    cols = M_norm.columns.values
    g.ax_heatmap.set_xticks([np.searchsorted(cols, x) for x in xt])
    g.ax_heatmap.set_xticklabels([str(x) for x in xt])
    g.ax_heatmap.axvline(np.searchsorted(cols, 0),
                         color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    g.ax_heatmap.set_ylabel(f"TFs (n = {len(M_norm)}, clustered)")
    g.fig.suptitle("TF clustering by TSS-proximal binding shape", y=1.0)
    g.savefig(out_path_stem + ".png", dpi=300)
    g.savefig(out_path_stem + ".pdf")
    plt.close(g.fig)


def plot_cluster_profiles(M: pd.DataFrame, M_norm: pd.DataFrame,
                          labels: pd.Series, out_path_stem: str):
    """Small-multiples grid: one panel per cluster with mean +/- SEM."""
    clusters = sorted(labels.unique())
    n = len(clusters)
    cols_grid = min(4, n)
    rows_grid = int(np.ceil(n / cols_grid))
    fig, axes = plt.subplots(rows_grid, cols_grid,
                             figsize=(3.6 * cols_grid, 2.6 * rows_grid),
                             sharex=True, sharey=False, squeeze=False)
    palette = sns.color_palette("tab10", n_colors=n)
    x = M.columns.values

    for i, c in enumerate(clusters):
        ax = axes[i // cols_grid][i % cols_grid]
        members = labels[labels == c].index
        sub = M.loc[members].values
        mean = sub.mean(axis=0)
        sem  = sub.std(axis=0, ddof=1) / np.sqrt(max(len(sub), 1))
        ax.fill_between(x, mean - sem, mean + sem,
                        color=palette[c - 1], alpha=0.25, linewidth=0)
        ax.plot(x, mean, color=palette[c - 1], linewidth=1.6)
        ax.axvline(0, color="red", linestyle="--", linewidth=0.7, alpha=0.6)

        # annotate up to 4 representative TFs per cluster (highest peak signal)
        repr_tfs = M.loc[members].max(axis=1).sort_values(ascending=False).head(4).index.tolist()
        ax.set_title(f"Cluster {c}  (n={len(members)})\n{', '.join(repr_tfs)}",
                     fontsize=9)
        ax.set_xticks([-1000, -500, 0, 500, 1000])

    for j in range(n, rows_grid * cols_grid):
        axes[j // cols_grid][j % cols_grid].set_visible(False)

    fig.supxlabel("Distance from TSS (bp)")
    fig.supylabel("Mean per-bp coverage probability (smoothed)")
    fig.suptitle("Per-cluster mean TSS-proximal binding profile", y=1.00)
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300)
    fig.savefig(out_path_stem + ".pdf")
    plt.close(fig)


def plot_cluster_histograms(hist_matrix: pd.DataFrame, labels: pd.Series,
                            out_dn: Path, bin_width: int,
                            title_tfs: int) -> None:
    """
    Per-cluster 5-nt overlap-density histogram (analogous to
    plots/overlap_histogram_5nt.score1000.png but restricted to the TFs in
    each cluster). Title lists up to `title_tfs` member TFs, ranked by
    per-TF total overlaps.
    """
    out_dn.mkdir(exist_ok=True)
    half = bin_width // 2
    n_bins = LEN // bin_width
    centers = np.arange(n_bins, dtype=np.int32) * bin_width - HALF + half

    palette = sns.color_palette("tab10", n_colors=labels.max())
    n_clusters = labels.max()
    fig_grid, axes_grid = plt.subplots(
        n_clusters, 1, figsize=(10, 2.6 * n_clusters), sharex=True,
        squeeze=False,
    )

    for c in sorted(labels.unique()):
        members = labels[labels == c].index
        members_in_hist = [t for t in members if t in hist_matrix.index]
        sub = hist_matrix.loc[members_in_hist]
        per_bp = sub.sum(axis=0).values.astype(np.int64)         # length LEN
        binned = per_bp[: n_bins * bin_width].reshape(n_bins, bin_width).sum(axis=1)

        # Title: up to `title_tfs` TFs ranked by per-TF total overlaps
        ranked = sub.sum(axis=1).sort_values(ascending=False).index.tolist()
        title_list = ranked[:title_tfs]
        more = len(ranked) - len(title_list)
        title_tfs_str = ", ".join(title_list)
        if more > 0:
            title_tfs_str += f" (+{more} more)"
        # wrap long TF lists
        wrap_len = 90
        wrapped = []
        line = ""
        for tok in title_tfs_str.split(", "):
            cand = (line + ", " + tok) if line else tok
            if len(cand) > wrap_len:
                wrapped.append(line); line = tok
            else:
                line = cand
        if line:
            wrapped.append(line)
        title = (f"Cluster {c} (n={len(members)}) — score==1000 overlap density "
                 f"({bin_width}-nt bins)\n" + "\n".join(wrapped))

        # Per-cluster standalone figure
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(centers, binned, width=bin_width, color=palette[c - 1],
               edgecolor="none", align="center")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.9, alpha=0.7,
                   label="TSS")
        ax.set_xticks([-1000, -500, 0, 500, 1000])
        ax.set_xlabel("Distance from TSS (bp)")
        ax.set_ylabel(f"Total overlaps per {bin_width} nt bin")
        ax.set_title(title, fontsize=10, loc="left")
        ax.legend(loc="upper right")
        fig.tight_layout()
        stem = out_dn / f"overlap_histogram_5nt.score1000.cluster{c}"
        fig.savefig(str(stem) + ".png", dpi=300)
        fig.savefig(str(stem) + ".pdf")
        plt.close(fig)

        # Companion grid panel
        ax = axes_grid[c - 1][0]
        ax.bar(centers, binned, width=bin_width, color=palette[c - 1],
               edgecolor="none", align="center")
        ax.axvline(0, color="red", linestyle="--", linewidth=0.7, alpha=0.6)
        ax.set_xticks([-1000, -500, 0, 500, 1000])
        ax.set_ylabel("overlaps")
        ax.set_title(f"Cluster {c} (n={len(members)}): {', '.join(title_list[:8])}"
                     + (f" (+{len(ranked) - 8} more)" if len(ranked) > 8 else ""),
                     fontsize=9, loc="left")

        # Save per-bin counts tsv
        pd.DataFrame({
            "bin_center_bp": centers,
            "bin_start_bp":  centers - half,
            "bin_end_bp":    centers + half + 1,
            "overlap_sum":   binned,
        }).to_csv(str(stem) + ".tsv", sep="\t", index=False)

    axes_grid[-1][0].set_xlabel("Distance from TSS (bp)")
    fig_grid.suptitle("Per-cluster TSS-proximal overlap density (score==1000, 5-nt bins)",
                      y=1.0)
    fig_grid.tight_layout()
    grid_stem = out_dn / "overlap_histogram_5nt.score1000.by_cluster"
    fig_grid.savefig(str(grid_stem) + ".png", dpi=300)
    fig_grid.savefig(str(grid_stem) + ".pdf")
    plt.close(fig_grid)


def plot_umap(M_norm: pd.DataFrame, labels: pd.Series,
              out_path_stem: str, random_state: int = 42):
    reducer = umap.UMAP(metric="correlation", n_neighbors=15, min_dist=0.1,
                        random_state=random_state)
    emb = reducer.fit_transform(M_norm.values)
    fig, ax = plt.subplots(figsize=(7.5, 7))
    palette = sns.color_palette("tab10", n_colors=labels.max())
    for c in sorted(labels.unique()):
        m = (labels.values == c)
        ax.scatter(emb[m, 0], emb[m, 1], s=12, alpha=0.7,
                   color=palette[c - 1], label=f"C{c} (n={m.sum()})")

    # highlight curated TFs
    for tf in HIGHLIGHT_TFS:
        if tf in M_norm.index:
            i = M_norm.index.get_loc(tf)
            ax.scatter(emb[i, 0], emb[i, 1], s=70, facecolors="none",
                       edgecolors="black", linewidth=1.2, zorder=5)
            ax.annotate(tf, (emb[i, 0], emb[i, 1]),
                        fontsize=8, xytext=(4, 4), textcoords="offset points")

    ax.set_xlabel("UMAP 1")
    ax.set_ylabel("UMAP 2")
    ax.set_title("UMAP of TFs by TSS-proximal binding shape (correlation metric)")
    ax.legend(loc="best", fontsize=8, framealpha=0.9, title="Cluster")
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300)
    fig.savefig(out_path_stem + ".pdf")
    plt.close(fig)
    return emb


################################################################################
# Execution ####################################################################
################################################################################
def main():
    out_dn = Path(OUT_DN)
    plots_dn = out_dn / "plots"
    cluster_dn = out_dn / "clustering_no_filter"        # separate from filtered run
    hist_dn    = plots_dn / "cluster_histograms_no_filter"
    cluster_dn.mkdir(exist_ok=True)
    plots_dn.mkdir(exist_ok=True)
    hist_dn.mkdir(exist_ok=True)
    plot_stem_suffix = ".no_filter"                     # disambiguate plot files

    print(f"loading {INPUT_MATRIX}...")
    M_full = load_matrix(OUT_DN, INPUT_MATRIX)
    print(f"  {M_full.shape}")

    M, dropped = filter_and_smooth(M_full, SIGNAL_FLOOR, SMOOTH_SIGMA)
    print(f"  filtered: kept {len(M)} / {len(M_full)} TFs "
          f"(row.max >= {SIGNAL_FLOOR}); dropped {len(dropped)}")

    M_norm = peak_normalize(M)

    print(f"clustering: distance={DISTANCE}, linkage={LINKAGE_METHOD}, k={K_CLUSTERS}")
    labels, Z, _ = cluster_tfs(M_norm, DISTANCE, LINKAGE_METHOD, K_CLUSTERS)
    labels = relabel_by_argmax(M_norm, labels)
    print("  cluster sizes:")
    print(labels.value_counts().sort_index().to_string())

    # ---- assignments + per-cluster summary ---------------------------------
    cols = M.columns.values
    assignments = pd.DataFrame({
        "TF": M.index,
        "cluster": labels.values,
        "row_max": M.max(axis=1).values,
        "argmax_bp": [int(cols[np.argmax(r)]) for r in M.values],
        "row_total_smoothed": M.sum(axis=1).values,
    }).sort_values(["cluster", "row_max"], ascending=[True, False])
    assignments.to_csv(cluster_dn / "tf_cluster_assignments.tsv",
                       sep="\t", index=False)

    # Simple requested table: TF, cluster, Peak distance from TSS
    simple = assignments[["TF", "cluster", "argmax_bp"]].rename(
        columns={"argmax_bp": "Peak distance from TSS"})
    simple.to_csv(cluster_dn / "tf_cluster_table.tsv", sep="\t", index=False)

    means = cluster_mean_profiles(M, labels)
    means.to_csv(cluster_dn / "cluster_mean_profiles.tsv",
                 sep="\t", index=False)

    # ---- plots --------------------------------------------------------------
    print("plotting clustermap...")
    plot_clustermap(M_norm, labels, Z,
                    out_path_stem=str(plots_dn / f"tf_clustermap.binary{plot_stem_suffix}"))

    print("plotting per-cluster profiles...")
    plot_cluster_profiles(M, M_norm, labels,
                          out_path_stem=str(plots_dn / f"tf_cluster_profiles{plot_stem_suffix}"))

    print("plotting UMAP...")
    plot_umap(M_norm, labels,
              out_path_stem=str(plots_dn / f"tf_umap{plot_stem_suffix}"),
              random_state=RANDOM_STATE)

    print("plotting per-cluster histograms (score==1000 overlap density)...")
    hist_matrix = pd.read_parquet(out_dn / HIST_MATRIX).set_index("TF")
    hist_matrix.columns = hist_matrix.columns.astype(int)
    plot_cluster_histograms(hist_matrix, labels,
                            out_dn=hist_dn,
                            bin_width=HIST_BIN_WIDTH,
                            title_tfs=HIST_TITLE_TFS)

    print("DONE")


if __name__ == "__main__":
    main()
