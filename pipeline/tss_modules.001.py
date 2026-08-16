#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Discover regulatory MODULES at canonical TSSs, then NMF on module × TF.

Each canonical protein-coding TSS gets its OWN set of modules — local
concentrations of TF binding within +/- 1500 bp — discovered by KDE on peak
midpoints (strand-oriented, weighted per-TSS-per-TF so every TF contributes
total mass = 1 at each TSS). A gene with a sharp focused promoter gets one
module; a gene with an upstream activation cluster and a downstream insulator
gets two; bipartite/multipartite cases come out naturally.

Pipeline:
    1. Per TF, find all peak midpoints whose recentered 25-nt block overlaps
       any TSS's +/- 1500 bp window. Emit (tss_id, local_offset, score).
       (Peaks NOT score-filtered for module discovery — position structure
        benefits from more data; per-TF mass normalization handles noise.)
    2. Group peak records by TSS. For each TSS:
         a. Per-TSS KDE (sigma = 25 bp) on peak midpoints, each peak weighted
            by 1 / n_peaks_of_that_TF_at_this_TSS so each TF contributes
            total mass = 1.
         b. find_peaks on the smoothed density -> module centers.
         c. Module boundaries: walk outward to FRAC * peak height OR to the
            valley between adjacent peaks.
         d. Filter to >= MIN_SUPPORT distinct TFs supporting the module
            (any score).
    3. For each surviving module, assign TF binary occupancy: TF j "binds"
       iff it has >= 1 peak with score >= MIN_SCORE_ASSIGN (=500) in [lo, hi].
    4. Build [n_module x n_tf] sparse binary matrix; run NMF for K ∈ KS.
    5. Per-gene "configuration": list of (module_idx, dominant_program,
       center_offset, width) — gene-level archetype falls out of this.

Outputs (tss_modules/):
    peaks.parquet              # per-peak records (intermediate)
    modules.tsv                # one row per module across all TSSs
    occupancy.modules.npz      # [n_module x n_tf] sparse binary
    tf_index.tsv, tss_table.tsv, module_index.tsv
    nmf.k{K}.{W,H}.tsv.gz
    nmf.k{K}.top_tfs.tsv
    nmf.k{K}.module_program.tsv
    nmf.k{K}.gene_configurations.tsv
    plots/
        module_count_distribution.{png,pdf}
        module_width_distribution.{png,pdf}
        module_position_density.{png,pdf}
        program_tf_loadings.k{K}.{png,pdf}
        program_position_density.k{K}.{png,pdf}
        program_sizes.k{K}.{png,pdf}
        program_vs_tfcluster.k{K}.{png,pdf}
        nmf_reconstruction_error.{png,pdf}
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
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
import scipy.sparse as sp
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks
from sklearn.decomposition import NMF

# Machine-specific paths and build axes -> pipeline/config.py
from config import normalize_chrom, MIN_SCORE_ASSIGN, discover_tf_files, read_peak_beds, tf_name_set, DNA_BINDING_FN, GTF_FN, OUT_DN, PER_TF_DN


################################################################################
# Initiating Variables #########################################################
################################################################################
TF_CLUSTER_FN       = OUT_DN / "clustering" / "tf_cluster_table.tsv"

OUTER_HALF          = 1500       # TSS +/- OUTER_HALF
PEAK_RECENTER_HALF  = 12         # 25-nt peak block, matches main pipeline
KDE_BW              = 25         # bandwidth for per-TSS density (sigma in bp)
MIN_SUPPORT         = 2          # >= unique TFs supporting a module (any score)

BOUNDARY_FRAC       = 0.20       # boundary at this fraction of peak height
MIN_PEAK_DIST_BP    = 50         # min separation between distinct module centers

VALID_CHROMS        = {str(c) for c in list(range(1, 23)) + ["X", "Y", "MT"]}

# Ranks to factorize. Overridable because some consumers want the module scan
# and nothing else: the split-half analysis reads only peaks.parquet and
# tf_index.tsv, so its four NMF fits were pure waste -- and on the 1,793-TF axis
# the mu solver's dense ~100k x 1,793 intermediates are what pushed that job
# out of memory. HPA_KS="" runs the scan and skips factorization entirely.
KS                  = [int(k) for k in
                       os.environ.get("HPA_KS", "8,12,15,20").split(",")
                       if k.strip()]
NMF_MAX_ITER        = 300
NMF_RANDOM_STATE    = 0
TOP_TFS_PER_PROGRAM = 30
RESUME_FROM_PEAKS   = True

# Worker count. SLURM_CPUS_PER_TASK wins on HPC: os.cpu_count() reports the whole
# node there (384 cores on a Roihu CPU node) while the cgroup owns only what was
# requested. Locally, cap at 12 to leave headroom and limit sustained thermal load.
# Override explicitly with HPA_WORKERS.
WORKERS             = max(1, int(os.environ.get(
    "HPA_WORKERS",
    os.environ.get("SLURM_CPUS_PER_TASK") or min(12, (os.cpu_count() or 2) - 2))))

sns.set_style("whitegrid")
plt.rcParams["font.size"]      = 11
plt.rcParams["axes.labelsize"] = 12
plt.rcParams["axes.titlesize"] = 14
plt.rcParams["figure.dpi"]     = 100


################################################################################
# Base helpers #################################################################
################################################################################


def _ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _make_logger(log_path: str):
    fh = open(log_path, "w")
    def _log(msg: str):
        line = f"[{_ts()}] {msg}"
        print(line)
        fh.write(line + "\n")
        fh.flush()
    return _log, fh


################################################################################
# Data prep ####################################################################
################################################################################
def load_canonical_tss(gtf_fn: str, valid_chroms: set) -> pd.DataFrame:
    raw = pd.read_csv(
        gtf_fn, sep="\t", comment="#", header=None, low_memory=False,
        names=["seqid", "source", "feature", "start", "end", "score",
               "strand", "frame", "attributes"],
        dtype={"seqid": "string", "feature": "category",
               "start": "int32", "end": "int32",
               "strand": "string", "attributes": "string"},
    )
    tx = raw[raw["feature"] == "transcript"].copy()
    attrs        = tx["attributes"]
    has_canon    = attrs.str.contains(r'tag "Ensembl_canonical"', regex=True, na=False)
    gene_biotype = attrs.str.extract(r'gene_biotype "([^"]+)"', expand=False)
    tx_biotype   = attrs.str.extract(r'transcript_biotype "([^"]+)"', expand=False)
    gene_id      = attrs.str.extract(r'gene_id "([^"]+)"', expand=False)
    gene_name    = attrs.str.extract(r'gene_name "([^"]+)"', expand=False)
    transcript_id = attrs.str.extract(r'transcript_id "([^"]+)"', expand=False)
    keep = (
        has_canon
        & (gene_biotype == "protein_coding")
        & (tx_biotype   == "protein_coding")
        & tx["seqid"].isin(valid_chroms)
    )
    tss = np.where(tx.loc[keep, "strand"] == "+",
                   tx.loc[keep, "start"], tx.loc[keep, "end"]).astype(np.int32)
    return pd.DataFrame({
        "chrom":         tx.loc[keep, "seqid"].astype(str).values,
        "tss":           tss,
        "strand":        tx.loc[keep, "strand"].astype(str).values,
        "gene_id":       gene_id[keep].values,
        "gene_name":     gene_name[keep].values,
        "transcript_id": transcript_id[keep].values,
    }).reset_index(drop=True)




# Worker globals --------------------------------------------------------------
_TSS_WINDOWS_PR = None
_VALID_CHROMS   = None


def _init_worker(tss_windows_df: pd.DataFrame, valid_chroms: set):
    global _TSS_WINDOWS_PR, _VALID_CHROMS
    _TSS_WINDOWS_PR = pr.PyRanges(tss_windows_df)
    _VALID_CHROMS = valid_chroms


def accumulate_tf_peaks(args) -> dict:
    """
    Per-TF worker. Returns per-peak records: (tss_id, local_offset, score) for
    every peak whose recentered midpoint falls within ANY TSS's +/- OUTER_HALF
    window. Strand-oriented offset (- = upstream in txn direction).
    """
    tf_name, bed_paths, tf_idx = args
    t0 = time.time()
    tss_id_arr = np.empty(0, dtype=np.int32)
    loc_arr    = np.empty(0, dtype=np.int16)
    score_arr  = np.empty(0, dtype=np.int16)
    n_peaks_kept = 0
    err = None
    try:
        peaks_df = read_peak_beds(bed_paths)
        peaks_df["Chromosome"] = normalize_chrom(peaks_df["Chromosome"])
        peaks_df = peaks_df[peaks_df["Chromosome"].isin(_VALID_CHROMS)].copy()
        n_peaks_kept = len(peaks_df)
        if n_peaks_kept:
            mid = ((peaks_df["Start"].astype(np.int64)
                  + peaks_df["End"].astype(np.int64)) // 2).astype(np.int32)
            peaks_df["mid"]   = mid
            peaks_df["score_i"] = peaks_df["score"].clip(0, 1000).astype(np.int16)
            peaks_df["Start"] = np.maximum(mid - PEAK_RECENTER_HALF, 0).astype(np.int32)
            peaks_df["End"]   = (mid + PEAK_RECENTER_HALF + 1).astype(np.int32)
            peaks_pr = pr.PyRanges(peaks_df)
            ov = peaks_pr.join(
                _TSS_WINDOWS_PR, suffix="_w",
                strandedness=False, apply_strand_suffix=False,
            ).df
            if not ov.empty:
                m   = ov["mid"].to_numpy(np.int32)
                tss = ov["tss_pos"].to_numpy(np.int32)
                sp_ = (ov["Strand"].to_numpy() == "+")
                local = np.where(sp_, m - tss, tss - m).astype(np.int32)
                # restrict to within OUTER_HALF (the join used a 25-nt block,
                # so a few might fall just outside if the block's edge touched
                # the window edge)
                keep = (local >= -OUTER_HALF) & (local <= OUTER_HALF)
                tss_id_arr = ov["tss_id"].to_numpy(np.int32)[keep]
                loc_arr    = local[keep].astype(np.int16)
                score_arr  = ov["score_i"].to_numpy(np.int16)[keep]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    return {"tf": tf_name, "tf_idx": tf_idx,
            "tss_id": tss_id_arr, "local": loc_arr, "score": score_arr,
            "n_peaks_kept": n_peaks_kept, "n_records": int(tss_id_arr.size),
            "runtime_s": time.time() - t0, "error": err}


################################################################################
# Module discovery #############################################################
################################################################################
def _detect_modules_density(density: np.ndarray,
                            kde_bw: int,
                            min_support: int,
                            boundary_frac: float,
                            min_peak_dist_bp: int) -> list:
    """
    Given a smoothed 1-D density on a 1-bp grid of length 2*OUTER_HALF+1
    (origin = OUTER_HALF), return a list of (lo_idx, hi_idx, peak_idx, height)
    for each detected module BEFORE TF-support filtering. Boundaries are walked
    outward to boundary_frac * peak_height OR a valley to a neighbor peak.
    """
    n = density.size
    # Threshold: integrated density of one TF (mass=1, gaussian sigma=kde_bw)
    # has peak height = 1 / (sigma * sqrt(2*pi)). Setting min height halfway
    # between (min_support - 1) and min_support TFs lets through modules
    # supported by ~min_support TFs and rejects single-TF modes.
    one_tf_peak = 1.0 / (kde_bw * np.sqrt(2 * np.pi))
    min_height  = (min_support - 0.5) * one_tf_peak
    min_prom    = 0.5 * one_tf_peak

    peaks, _ = find_peaks(density,
                          height=min_height,
                          prominence=min_prom,
                          distance=int(min_peak_dist_bp))
    if len(peaks) == 0:
        return []

    out = []
    for i, pk in enumerate(peaks):
        h = float(density[pk])
        thr = max(boundary_frac * h, 0.5 * one_tf_peak)
        # Left bound: between previous peak (if any) and pk.
        left_limit = peaks[i - 1] if i > 0 else 0
        # If a valley exists between, find it; lo cannot cross it.
        if i > 0:
            valley = left_limit + int(np.argmin(density[left_limit:pk + 1]))
        else:
            valley = 0
        lo = pk
        while lo > valley and density[lo - 1] >= thr:
            lo -= 1
        # Right bound mirror
        right_limit = peaks[i + 1] if i < len(peaks) - 1 else n - 1
        if i < len(peaks) - 1:
            valley_r = pk + int(np.argmin(density[pk:right_limit + 1]))
        else:
            valley_r = n - 1
        hi = pk
        while hi < valley_r and density[hi + 1] >= thr:
            hi += 1
        out.append((lo, hi, pk, h))
    return out


def process_tss_records(tss_id: int, tss_meta: dict,
                        pk_local: np.ndarray, pk_tf: np.ndarray,
                        pk_score: np.ndarray,
                        kde_bw: int, outer_half: int,
                        min_support: int, min_score_assign: int,
                        boundary_frac: float, min_peak_dist_bp: int):
    """
    Return (modules_list, occupancy_pairs_list) for one TSS.
    modules_list[i] is a dict; occupancy_pairs_list[i] is list of tf_idx ints
    aligned to modules_list[i] (so module_local_idx == i).
    """
    n_grid = 2 * outer_half + 1
    grid = np.zeros(n_grid, dtype=np.float64)

    # Per-TSS-per-TF normalization: weight = 1/count(TF at TSS).
    # Vectorized: group counts via bincount on tf array.
    if pk_tf.size == 0:
        return [], []
    tf_max = int(pk_tf.max()) + 1
    tf_counts = np.bincount(pk_tf, minlength=tf_max)
    weights = 1.0 / tf_counts[pk_tf]

    grid_idx = (pk_local.astype(np.int64) + outer_half)
    valid = (grid_idx >= 0) & (grid_idx < n_grid)
    np.add.at(grid, grid_idx[valid], weights[valid])

    smoothed = gaussian_filter1d(grid, sigma=kde_bw)

    candidates = _detect_modules_density(
        smoothed, kde_bw, min_support, boundary_frac, min_peak_dist_bp)
    if not candidates:
        return [], []

    modules_out = []
    occupancy_out = []
    for (lo_idx, hi_idx, pk_idx, h) in candidates:
        lo_off = lo_idx - outer_half
        hi_off = hi_idx - outer_half
        center = pk_idx - outer_half
        in_mod = (pk_local >= lo_off) & (pk_local <= hi_off)
        n_in = int(in_mod.sum())
        if n_in == 0:
            continue
        tfs_supp = np.unique(pk_tf[in_mod])
        if tfs_supp.size < min_support:
            continue
        # binary occupancy: TF must have a peak in [lo,hi] with score>=min_score_assign
        sp_mask = in_mod & (pk_score >= min_score_assign)
        tfs_assigned = np.unique(pk_tf[sp_mask])
        modules_out.append({
            **tss_meta,
            "tss_id":           tss_id,
            "module_local_idx": len(modules_out),
            "lo_offset":        int(lo_off),
            "hi_offset":        int(hi_off),
            "center_offset":    int(center),
            "width":            int(hi_off - lo_off + 1),
            "kde_height":       float(h),
            "n_peaks_in":       n_in,
            "n_tfs_supporting": int(tfs_supp.size),
            "n_tfs_assigned":   int(tfs_assigned.size),
        })
        occupancy_out.append(tfs_assigned.astype(np.int32))
    return modules_out, occupancy_out


################################################################################
# NMF utilities ################################################################
################################################################################
def run_nmf(M, k: int) -> tuple:
    nmf = NMF(n_components=k, init="random",
              max_iter=NMF_MAX_ITER, random_state=NMF_RANDOM_STATE,
              beta_loss="frobenius", solver="mu", tol=1e-4)
    W = nmf.fit_transform(M)
    H = nmf.components_
    return W, H, float(nmf.reconstruction_err_)


def relabel_programs_by_size(W: np.ndarray, H: np.ndarray) -> tuple:
    dom = W.argmax(axis=1)
    counts = np.bincount(dom, minlength=W.shape[1])
    order = np.argsort(-counts)
    return W[:, order], H[order, :]


################################################################################
# Plotting #####################################################################
################################################################################
def plot_module_count_distribution(modules_df: pd.DataFrame, n_tss_total: int,
                                   out_stem: str):
    counts = modules_df.groupby("tss_id").size().reindex(
        np.arange(n_tss_total), fill_value=0)
    vc = counts.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(vc.index.values, vc.values, color="steelblue",
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("modules per TSS")
    ax.set_ylabel("# canonical TSSs")
    ax.set_title(f"Modules per canonical TSS  (n_tss={n_tss_total}, "
                 f"n_modules={len(modules_df)}, mean={counts.mean():.2f})")
    for x, y in zip(vc.index.values, vc.values):
        ax.text(x, y, f"{y}", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_module_width_distribution(modules_df: pd.DataFrame, out_stem: str):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(modules_df["width"].values, bins=80, color="seagreen",
            edgecolor="black", linewidth=0.3)
    ax.axvline(modules_df["width"].median(), color="red", linestyle="--",
               linewidth=1, label=f"median={modules_df['width'].median():.0f} bp")
    ax.set_xlabel("module width (bp)")
    ax.set_ylabel("# modules")
    ax.set_title(f"Module width distribution  (n_modules={len(modules_df)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_module_position_density(modules_df: pd.DataFrame, out_stem: str):
    """KDE of module CENTER positions, pooled across TSSs."""
    centers = modules_df["center_offset"].values
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(centers, bins=120, range=(-OUTER_HALF, OUTER_HALF),
            color="C0", alpha=0.85, edgecolor="black", linewidth=0.2)
    ax.axvline(0, color="red", linestyle="--", linewidth=1, label="TSS")
    ax.set_xlabel("module center, bp from TSS (txn-oriented)")
    ax.set_ylabel("# modules")
    ax.set_title(f"Module center positions (pooled, n={len(centers)})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_program_tf_heatmap(H: np.ndarray, tf_names: list,
                            top_n_per_program: int, out_stem: str):
    top_set = set()
    for p in range(H.shape[0]):
        top_set.update(np.argsort(H[p])[::-1][:top_n_per_program].tolist())
    cols = sorted(top_set)
    H_sub = H[:, cols]
    dom = H_sub.argmax(axis=0)
    dom_load = H_sub[dom, np.arange(H_sub.shape[1])]
    order = np.lexsort((-dom_load, dom))
    H_ord = H_sub[:, order]
    tf_ord = [tf_names[cols[i]] for i in order]
    width  = max(10, 0.10 * len(tf_ord) + 2)
    height = max(3.5, 0.32 * H.shape[0] + 1.2)
    fig, ax = plt.subplots(figsize=(width, height))
    vmax = float(np.quantile(H_ord, 0.99)) if H_ord.size else 1.0
    if vmax <= 0:
        vmax = float(H_ord.max() or 1.0)
    im = ax.imshow(H_ord, aspect="auto", cmap="magma",
                   vmin=0, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(H.shape[0]))
    ax.set_yticklabels([f"P{p+1}" for p in range(H.shape[0])])
    ax.set_xticks(range(len(tf_ord)))
    ax.set_xticklabels(tf_ord, rotation=90, fontsize=7)
    ax.set_ylabel("Program")
    ax.set_xlabel(f"TFs (union of top {top_n_per_program} per program, "
                  f"n={len(tf_ord)})")
    ax.set_title(f"NMF program × TF loadings (k={H.shape[0]})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("H loading")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_program_position_density(modules_df_with_program: pd.DataFrame,
                                  out_stem: str):
    """For each program, KDE of module-center positions of modules dominantly
    assigned to that program. One panel per program, shared x-axis."""
    K = modules_df_with_program["dominant_program"].max()
    fig, axes = plt.subplots(K, 1, figsize=(8, 1.4 * K + 1.2),
                              sharex=True, squeeze=False)
    bins = np.linspace(-OUTER_HALF, OUTER_HALF, 121)
    centers = 0.5 * (bins[:-1] + bins[1:])
    for p in range(1, K + 1):
        ax = axes[p - 1, 0]
        sub = modules_df_with_program[
            modules_df_with_program["dominant_program"] == p]
        if not sub.empty:
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
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_program_sizes(W: np.ndarray, out_stem: str):
    K = W.shape[1]
    dom = W.argmax(axis=1)
    counts = np.bincount(dom, minlength=K)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * K + 2), 4))
    ax.bar(np.arange(K) + 1, counts, color="steelblue",
           edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Program")
    ax.set_ylabel("Modules with this program dominant")
    ax.set_title(f"Program sizes (k={K}, n_modules={W.shape[0]:,})")
    ax.set_xticks(np.arange(K) + 1)
    for i, c in enumerate(counts):
        ax.text(i + 1, c, f"{c}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_reconstruction_error(errs: dict, n_module: int, n_tf: int, nnz: int,
                              out_stem: str):
    ks = sorted(errs); vals = [errs[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, "-o", color="C3")
    ax.set_xlabel("k (n_components)")
    ax.set_ylabel("Frobenius reconstruction error")
    ax.set_title(f"NMF error vs k  (n_modules={n_module}, n_tf={n_tf}, "
                 f"nnz={nnz:,})")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_program_vs_tfcluster(H: np.ndarray, tf_names: list,
                              tf_cluster_fn: Path, top_n: int,
                              out_stem: str):
    if not tf_cluster_fn.exists():
        return
    tab = pd.read_csv(tf_cluster_fn, sep="\t")
    tf_to_cluster = dict(zip(tab["TF"], tab["cluster"]))
    clusters_present = sorted(set(tab["cluster"]))
    K = H.shape[0]
    M = np.zeros((K, len(clusters_present)), dtype=np.int32)
    for p in range(K):
        idx = np.argsort(H[p])[::-1][:top_n]
        for j in idx:
            c = tf_to_cluster.get(tf_names[j])
            if c is None:
                continue
            M[p, clusters_present.index(c)] += 1
    fig, ax = plt.subplots(figsize=(0.6 * len(clusters_present) + 4,
                                     0.4 * K + 2))
    im = ax.imshow(M, aspect="auto", cmap="Blues", interpolation="nearest")
    ax.set_xticks(range(len(clusters_present)))
    ax.set_xticklabels([f"TFc{c}" for c in clusters_present])
    ax.set_yticks(range(K))
    ax.set_yticklabels([f"P{p+1}" for p in range(K)])
    ax.set_xlabel("Filtered-K8 TF cluster")
    ax.set_ylabel("Program")
    ax.set_title(f"Program top-{top_n} TFs by TF cluster (k={K})")
    if M.max():
        for p in range(K):
            for c in range(len(clusters_present)):
                if M[p, c] > 0:
                    ax.text(c, p, str(M[p, c]), ha="center", va="center",
                            color="black" if M[p, c] < M.max() / 2 else "white",
                            fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"# of top-{top_n} TFs in cluster")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


################################################################################
# Execution ####################################################################
################################################################################
def main():
    out_root = OUT_DN / "tss_modules"
    plots_dn = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dn.mkdir(parents=True, exist_ok=True)
    (OUT_DN / "logs").mkdir(parents=True, exist_ok=True)
    log_path = OUT_DN / "logs" / f"tss_modules.{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    log, log_fh = _make_logger(str(log_path))

    log(f"WORKERS = {WORKERS}")
    log(f"OUTER_HALF = {OUTER_HALF}, KDE_BW = {KDE_BW}, "
        f"MIN_SUPPORT = {MIN_SUPPORT}, MIN_SCORE_ASSIGN = {MIN_SCORE_ASSIGN}")
    log(f"KS = {KS}")

    peaks_fn  = out_root / "peaks.parquet"
    tss_fn    = out_root / "tss_table.tsv"
    tf_fn     = out_root / "tf_index.tsv"

    if RESUME_FROM_PEAKS and all(p.exists() for p in [peaks_fn, tss_fn, tf_fn]):
        log(f"resuming peaks from {peaks_fn.name}")
        peaks_df = pd.read_parquet(peaks_fn)
        tss_df   = pd.read_csv(tss_fn, sep="\t")
        tf_names = pd.read_csv(tf_fn, sep="\t").sort_values("tf_idx")["TF"].tolist()
        n_tss = len(tss_df); n_tf = len(tf_names)
        log(f"  n_records={len(peaks_df):,}  n_tss={n_tss:,}  n_tf={n_tf}")
    else:
        # 1) Canonical TSSs
        tss_df = load_canonical_tss(GTF_FN, VALID_CHROMS)
        n_tss = len(tss_df)
        log(f"n_tss = {n_tss:,}")
        tss_df.to_csv(tss_fn, sep="\t", index=False)

        windows_df = pd.DataFrame({
            "Chromosome": tss_df["chrom"].values,
            "Start":      np.maximum(tss_df["tss"].values - OUTER_HALF, 0).astype(np.int32),
            "End":        (tss_df["tss"].values + OUTER_HALF + 1).astype(np.int32),
            "Strand":     tss_df["strand"].values,
            "tss_pos":    tss_df["tss"].values.astype(np.int32),
            "tss_id":     np.arange(n_tss, dtype=np.int32),
        })

        # 2) Discover TFs
        tf_axis = tf_name_set()
        tf_files = discover_tf_files(PER_TF_DN, tf_axis)
        n_tf = len(tf_files)
        log(f"n_tf  = {n_tf}")
        if n_tf == 0:
            raise RuntimeError("No TF files matched DNA-binding whitelist")
        tf_args = [(name, path, idx) for idx, (name, path) in enumerate(tf_files)]

        # 3) Per-TF accumulation
        t0 = time.time()
        ctx = mp.get_context("fork")
        with ctx.Pool(processes=WORKERS,
                      initializer=_init_worker,
                      initargs=(windows_df, VALID_CHROMS)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(accumulate_tf_peaks,
                                                      tf_args, chunksize=4), 1):
                results.append(r)
                if i % 100 == 0 or i == n_tf:
                    log(f"  processed {i}/{n_tf} TFs "
                        f"(last: {r['tf']} — {r['n_records']:,} records, "
                        f"{r['runtime_s']:.2f}s)")
                    if r.get("error"):
                        log(f"    [ERROR] {r['tf']}: {r['error']}")
        log(f"per-TF scan done in {time.time() - t0:.1f}s")
        results.sort(key=lambda r: r["tf_idx"])
        tf_names = [r["tf"] for r in results]

        # 4) Concat per-peak records
        tss_id_all = np.concatenate([r["tss_id"] for r in results])
        local_all  = np.concatenate([r["local"]  for r in results])
        score_all  = np.concatenate([r["score"]  for r in results])
        tf_all = np.concatenate([np.full(r["tss_id"].size, r["tf_idx"],
                                         dtype=np.int32) for r in results])
        log(f"total peak records: {tss_id_all.size:,}")

        # Sort by tss_id so we can do a single linear pass through groups
        order = np.argsort(tss_id_all, kind="stable")
        peaks_df = pd.DataFrame({
            "tss_id":  tss_id_all[order].astype(np.int32),
            "tf_idx":  tf_all[order].astype(np.int32),
            "local":   local_all[order].astype(np.int16),
            "score":   score_all[order].astype(np.int16),
        })
        peaks_df.to_parquet(peaks_fn, index=False)
        pd.Series(tf_names, name="TF").to_csv(tf_fn, sep="\t",
                                              index_label="tf_idx")

    # 5) Per-TSS module discovery (sequential — fast, no shared state)
    log("discovering modules per TSS...")
    t0 = time.time()
    tss_id_arr = peaks_df["tss_id"].to_numpy()
    tf_idx_arr = peaks_df["tf_idx"].to_numpy()
    local_arr  = peaks_df["local"].to_numpy().astype(np.int32)
    score_arr  = peaks_df["score"].to_numpy().astype(np.int32)

    # Find group boundaries via searchsorted on the sorted tss_id array
    unique_tss, group_starts = np.unique(tss_id_arr, return_index=True)
    group_ends = np.append(group_starts[1:], tss_id_arr.size)

    # Lookup table for tss meta
    tss_meta_cols = ["chrom", "tss", "strand",
                     "gene_id", "gene_name", "transcript_id"]
    tss_meta_lookup = tss_df[tss_meta_cols].to_dict(orient="records")

    all_modules = []
    all_occupancy_per_module = []   # list of arrays of tf_idx, aligned to all_modules
    n_tss_with_modules = 0

    for tss_id, gs, ge in zip(unique_tss, group_starts, group_ends):
        pk_local = local_arr[gs:ge]
        pk_tf    = tf_idx_arr[gs:ge]
        pk_score = score_arr[gs:ge]
        meta = tss_meta_lookup[int(tss_id)]
        mods, occ = process_tss_records(
            int(tss_id), meta, pk_local, pk_tf, pk_score,
            kde_bw=KDE_BW, outer_half=OUTER_HALF,
            min_support=MIN_SUPPORT, min_score_assign=MIN_SCORE_ASSIGN,
            boundary_frac=BOUNDARY_FRAC,
            min_peak_dist_bp=MIN_PEAK_DIST_BP)
        if mods:
            n_tss_with_modules += 1
            all_modules.extend(mods)
            all_occupancy_per_module.extend(occ)

    log(f"module discovery done in {time.time() - t0:.1f}s")
    log(f"  TSSs with >=1 module: {n_tss_with_modules:,} of {len(tss_df):,}")
    log(f"  total modules: {len(all_modules):,}")

    # Build modules DataFrame and assign global module_id
    modules_df = pd.DataFrame(all_modules)
    modules_df["module_id"] = np.arange(len(modules_df), dtype=np.int32)
    modules_df.to_csv(out_root / "modules.tsv", sep="\t", index=False)

    # Distribution sanity log
    log(f"  modules-per-TSS  median={modules_df.groupby('tss_id').size().median():.0f}  "
        f"mean={modules_df.groupby('tss_id').size().mean():.2f}  "
        f"max={modules_df.groupby('tss_id').size().max()}")
    log(f"  module width   median={modules_df['width'].median():.0f} bp  "
        f"p10={modules_df['width'].quantile(0.1):.0f}  "
        f"p90={modules_df['width'].quantile(0.9):.0f}")
    log(f"  module center  median={modules_df['center_offset'].median():+.0f}  "
        f"p10={modules_df['center_offset'].quantile(0.1):+.0f}  "
        f"p90={modules_df['center_offset'].quantile(0.9):+.0f}")

    # 6) Build [n_module x n_tf] sparse binary
    in_matrix_mask = modules_df["n_tfs_assigned"].to_numpy() > 0
    keep_rows = np.flatnonzero(in_matrix_mask)
    log(f"  modules in matrix (n_tfs_assigned > 0): {keep_rows.size:,}")

    rows_list, cols_list = [], []
    for new_row, orig_idx in enumerate(keep_rows):
        tfs = all_occupancy_per_module[orig_idx]
        if tfs.size == 0:
            continue
        rows_list.append(np.full(tfs.size, new_row, dtype=np.int32))
        cols_list.append(tfs.astype(np.int32))
    rows = np.concatenate(rows_list) if rows_list else np.empty(0, np.int32)
    cols = np.concatenate(cols_list) if cols_list else np.empty(0, np.int32)
    data = np.ones(rows.size, dtype=np.float32)
    n_module = keep_rows.size
    n_tf = len(tf_names)
    M = sp.coo_matrix((data, (rows, cols)),
                       shape=(n_module, n_tf)).tocsr()
    M.sum_duplicates()
    log(f"M shape = {M.shape}  nnz = {M.nnz:,}  "
        f"density = {M.nnz / (n_module * n_tf):.4f}")
    sp.save_npz(str(out_root / "occupancy.modules.npz"), M)

    # module_index.tsv: maps matrix row -> global module_id
    pd.DataFrame({
        "matrix_row": np.arange(n_module),
        "module_id":  modules_df.iloc[keep_rows]["module_id"].values,
        "tss_id":     modules_df.iloc[keep_rows]["tss_id"].values,
        "gene_name":  modules_df.iloc[keep_rows]["gene_name"].values,
        "transcript_id": modules_df.iloc[keep_rows]["transcript_id"].values,
        "module_local_idx": modules_df.iloc[keep_rows]["module_local_idx"].values,
        "center_offset": modules_df.iloc[keep_rows]["center_offset"].values,
        "width":      modules_df.iloc[keep_rows]["width"].values,
    }).to_csv(out_root / "module_index.tsv", sep="\t", index=False)

    # Sanity plots (over ALL detected modules, not just in-matrix)
    plot_module_count_distribution(modules_df, len(tss_df),
                                   str(plots_dn / "module_count_distribution"))
    plot_module_width_distribution(modules_df,
                                   str(plots_dn / "module_width_distribution"))
    plot_module_position_density(modules_df,
                                 str(plots_dn / "module_position_density"))

    # 7) NMF for each k
    errs = {}
    for k in KS:
        log(f"--- NMF k={k} ---")
        t1 = time.time()
        W, H, err = run_nmf(M, k)
        W, H = relabel_programs_by_size(W, H)
        errs[k] = err
        log(f"  k={k}: err={err:.2f}  fit_time={time.time() - t1:.1f}s")

        pd.DataFrame(W,
                     index=modules_df.iloc[keep_rows]["module_id"].values,
                     columns=[f"prog{p+1}" for p in range(k)]
        ).to_csv(out_root / f"nmf.k{k}.W.tsv.gz", sep="\t",
                 compression="gzip", index_label="module_id")
        pd.DataFrame(H,
                     index=[f"prog{p+1}" for p in range(k)],
                     columns=tf_names
        ).to_csv(out_root / f"nmf.k{k}.H.tsv.gz", sep="\t",
                 compression="gzip", index_label="program")

        # Top TFs per program
        top_rows = []
        for p in range(k):
            idx = np.argsort(H[p])[::-1][:TOP_TFS_PER_PROGRAM]
            for rank, j in enumerate(idx, 1):
                top_rows.append({"program": p + 1, "rank": rank,
                                 "tf": tf_names[j], "loading": float(H[p, j])})
        pd.DataFrame(top_rows).to_csv(out_root / f"nmf.k{k}.top_tfs.tsv",
                                       sep="\t", index=False)

        # Module → program assignment
        row_sum = W.sum(axis=1, keepdims=True)
        W_norm = W / np.where(row_sum > 0, row_sum, 1.0)
        dom = W_norm.argmax(axis=1) + 1
        mp_df = pd.DataFrame({
            "module_id":         modules_df.iloc[keep_rows]["module_id"].values,
            "tss_id":            modules_df.iloc[keep_rows]["tss_id"].values,
            "gene_name":         modules_df.iloc[keep_rows]["gene_name"].values,
            "transcript_id":     modules_df.iloc[keep_rows]["transcript_id"].values,
            "center_offset":     modules_df.iloc[keep_rows]["center_offset"].values,
            "width":             modules_df.iloc[keep_rows]["width"].values,
            "dominant_program":  dom,
            "dominant_weight":   W_norm[np.arange(W.shape[0]), dom - 1],
        })
        for p in range(k):
            mp_df[f"prog{p+1}_w"] = W_norm[:, p]
        mp_df.to_csv(out_root / f"nmf.k{k}.module_program.tsv",
                     sep="\t", index=False)

        # Per-gene configurations (module list with program assignments)
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
        gene_conf.to_csv(out_root / f"nmf.k{k}.gene_configurations.tsv",
                         sep="\t", index=False)

        # Plots
        plot_program_tf_heatmap(H, tf_names, TOP_TFS_PER_PROGRAM,
                                str(plots_dn / f"program_tf_loadings.k{k}"))
        plot_program_position_density(mp_df,
                                      str(plots_dn / f"program_position_density.k{k}"))
        plot_program_sizes(W, str(plots_dn / f"program_sizes.k{k}"))
        plot_program_vs_tfcluster(H, tf_names, TF_CLUSTER_FN,
                                   TOP_TFS_PER_PROGRAM,
                                   str(plots_dn / f"program_vs_tfcluster.k{k}"))

        # Quick log readout
        for p in range(k):
            tfs_top = [tf_names[j] for j in np.argsort(H[p])[::-1][:8]]
            n_dom = int((dom == p + 1).sum())
            sub = mp_df[mp_df["dominant_program"] == p + 1]
            med_pos = int(sub["center_offset"].median()) if len(sub) else 0
            med_wid = int(sub["width"].median()) if len(sub) else 0
            log(f"  P{p+1} (n_dom={n_dom:5d}, med_center={med_pos:+5d}, "
                f"med_width={med_wid:4d}): {', '.join(tfs_top)}")

    # Skipped entirely when HPA_KS="" -- there is no error curve to plot or
    # write, and an empty frame here would either crash the plot or leave a
    # header-only TSV that looks like a finished factorization.
    if errs:
        plot_reconstruction_error(errs, n_module, n_tf, M.nnz,
                                  str(plots_dn / "nmf_reconstruction_error"))
        pd.DataFrame({"k": list(errs.keys()),
                      "frobenius_err": list(errs.values())}).to_csv(
            out_root / "nmf_reconstruction_error.tsv", sep="\t", index=False)
    else:
        log("NMF skipped (HPA_KS empty) -- modules + peaks.parquet only")

    log("DONE")
    log_fh.close()


if __name__ == "__main__":
    main()
