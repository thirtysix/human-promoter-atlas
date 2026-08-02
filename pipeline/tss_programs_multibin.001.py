#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Multi-bin TSS-binding programs by NMF.

Same idea as tss_programs.001.py, but each TF is split into 5 strand-aware
positional bins across a wider TSS +/- 1500 bp window:

    B1.far_up      [-1500, -500)
    B2.prox_up     [-500,    -50)
    B3.core        [-50,     +50)
    B4.prox_dn     [+50,    +500)
    B5.far_dn      [+500, +1500]

Feature matrix: [n_tss x (n_tf * 5)] binary occupancy
    M[i, tf_idx*5 + bin_idx] = 1
        iff TF tf_idx has at least one peak (midpoint) for TSS i in bin bin_idx.

NMF then learns programs that can mix TFs and positions, e.g. "TBP-core +
SP1-prox_up" as one component and "TBP-core + CTCF-far_dn" as another --
something the single-window matrix cannot resolve.

Outputs (tss_programs_multibin/):
    occupancy.wide{H}_b{N}.npz     # CSR sparse [n_tss x n_tf*N_BINS]
    feature_index.tsv              # column -> (tf, bin)
    nmf.k{K}.{W,H}.tsv.gz
    nmf.k{K}.top_features.tsv      # top features per program (TF.bin)
    nmf.k{K}.tss_assignment.tsv
    nmf.k{K}.program_tf_bin.tsv    # per program: TF x bin loading grid (long form)
    plots/
        program_tf_bin.k{K}.{png,pdf}    # for each k: K horizontal panels,
                                         # each panel = top TFs x 5 bins heatmap
        program_sizes.k{K}.{png,pdf}
        nmf_reconstruction_error.{png,pdf}
        program_vs_tfcluster.k{K}.{png,pdf}
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
from sklearn.decomposition import NMF

# Machine-specific paths and build axes -> pipeline/config.py
from config import normalize_chrom, discover_tf_files, read_peak_beds, tf_name_set, DNA_BINDING_FN, GTF_FN, OUT_DN, PER_TF_DN


################################################################################
# Initiating Variables #########################################################
################################################################################
TF_CLUSTER_FN       = OUT_DN / "clustering" / "tf_cluster_table.tsv"

WIDE_HALF           = 1500       # TSS +/- WIDE_HALF
PEAK_RECENTER_HALF  = 12         # 25-nt peak block, matches main pipeline
VALID_CHROMS        = {str(c) for c in list(range(1, 23)) + ["X", "Y", "MT"]}

# Bin edges (5 bins). Right-open: searchsorted on the inner edges yields 0..N-1.
BIN_EDGES_INNER     = np.array([-500, -50, 50, 500], dtype=np.int32)
BIN_LABELS          = ["B1.far_up", "B2.prox_up", "B3.core",
                       "B4.prox_dn", "B5.far_dn"]
N_BINS              = len(BIN_LABELS)

KS                  = [8, 12, 15, 20]
NMF_MAX_ITER        = 300
NMF_RANDOM_STATE    = 0
TOP_FEATURES_PER_K  = 30
TOP_TFS_FOR_GRID    = 25         # rows in the per-program TF×bin heatmap
RESUME_FROM_NPZ     = True

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


def accumulate_tf_multibin(args) -> dict:
    """
    Per-TF worker. Returns the unique (tss_id, bin_idx) pairs bound, computed
    from each peak's recentered midpoint relative to the TSS (strand-aware).
    """
    tf_name, bed_paths, tf_idx = args
    t0 = time.time()
    pairs_tss = np.empty(0, dtype=np.int32)
    pairs_bin = np.empty(0, dtype=np.int8)
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
                # local position of peak mid relative to TSS, oriented to txn dir
                local = np.where(sp_, m - tss, tss - m).astype(np.int32)
                # bin index in {0..N_BINS-1}
                bin_ix = np.searchsorted(BIN_EDGES_INNER, local).astype(np.int8)

                tss_id = ov["tss_id"].to_numpy(np.int32)

                # Dedup unique (tss_id, bin_ix)
                key = tss_id.astype(np.int64) * N_BINS + bin_ix.astype(np.int64)
                _, idx = np.unique(key, return_index=True)
                pairs_tss = tss_id[idx]
                pairs_bin = bin_ix[idx]
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    return {"tf": tf_name, "tf_idx": tf_idx,
            "pairs_tss": pairs_tss, "pairs_bin": pairs_bin,
            "n_peaks_kept": n_peaks_kept,
            "n_pairs": int(pairs_tss.size),
            "runtime_s": time.time() - t0, "error": err}


def build_occupancy_matrix(results: list, n_tss: int, n_tf: int) -> sp.csr_matrix:
    """[n_tss x (n_tf * N_BINS)] CSR sparse, binary."""
    n_cols = n_tf * N_BINS
    rows_chunks, cols_chunks = [], []
    for r in results:
        if r["pairs_tss"].size == 0:
            continue
        rows_chunks.append(r["pairs_tss"].astype(np.int32))
        col = (r["tf_idx"] * N_BINS + r["pairs_bin"].astype(np.int32)).astype(np.int32)
        cols_chunks.append(col)
    if not rows_chunks:
        return sp.csr_matrix((n_tss, n_cols), dtype=np.float32)
    rows = np.concatenate(rows_chunks)
    cols = np.concatenate(cols_chunks)
    data = np.ones(rows.size, dtype=np.float32)
    M = sp.coo_matrix((data, (rows, cols)), shape=(n_tss, n_cols)).tocsr()
    M.sum_duplicates()
    return M


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
    dominant = W.argmax(axis=1)
    counts = np.bincount(dominant, minlength=W.shape[1])
    order = np.argsort(-counts)
    return W[:, order], H[order, :]


def feature_names(tf_names: list) -> list:
    return [f"{tf}.{lab}" for tf in tf_names for lab in BIN_LABELS]


def top_features_table(H: np.ndarray, feat_names: list, top_n: int) -> pd.DataFrame:
    rows = []
    for p in range(H.shape[0]):
        idx = np.argsort(H[p])[::-1][:top_n]
        for rank, j in enumerate(idx, 1):
            tf, bin_lab = feat_names[j].rsplit(".", 1)
            rows.append({"program": p + 1, "rank": rank,
                         "feature": feat_names[j],
                         "tf": tf, "bin": bin_lab,
                         "loading": float(H[p, j])})
    return pd.DataFrame(rows)


def program_tf_bin_long(H: np.ndarray, tf_names: list) -> pd.DataFrame:
    """Long-format DataFrame: program, tf, bin, loading."""
    K = H.shape[0]
    n_tf = len(tf_names)
    rows = []
    for p in range(K):
        for j, tf in enumerate(tf_names):
            base = j * N_BINS
            for b in range(N_BINS):
                v = float(H[p, base + b])
                if v > 0:
                    rows.append({"program": p + 1, "tf": tf,
                                 "bin": BIN_LABELS[b], "loading": v})
    return pd.DataFrame(rows)


def tss_assignment_table(W: np.ndarray, tss_df: pd.DataFrame) -> pd.DataFrame:
    K = W.shape[1]
    row_sum = W.sum(axis=1, keepdims=True)
    W_norm = W / np.where(row_sum > 0, row_sum, 1.0)
    dom = W_norm.argmax(axis=1)
    out = tss_df[["chrom", "tss", "strand", "gene_id", "gene_name",
                  "transcript_id"]].copy()
    out["dominant_program"] = dom + 1
    out["dominant_weight"]  = W_norm[np.arange(W.shape[0]), dom]
    out["total_weight"]     = W.sum(axis=1)
    for p in range(K):
        out[f"prog{p+1}_w"] = W_norm[:, p]
    return out


################################################################################
# Plotting #####################################################################
################################################################################
def plot_program_tf_bin_grid(H: np.ndarray, tf_names: list, top_tfs_n: int,
                             out_stem: str):
    """
    K-row grid of TF × bin heatmaps, one row per program.
    For each program, pick the top-N TFs by max-over-bins loading; all programs
    use the union of those TFs (sorted by aggregate loading) so columns line up
    across rows.
    """
    K = H.shape[0]
    n_tf = len(tf_names)

    # H -> per-program (n_tf, N_BINS) tensor
    H3 = H.reshape(K, n_tf, N_BINS)            # NB: feature-major: tf*N_BINS + b
    # Top-N TFs per program (by max loading across bins)
    top_set = set()
    for p in range(K):
        max_per_tf = H3[p].max(axis=1)
        top_set.update(np.argsort(max_per_tf)[::-1][:top_tfs_n].tolist())
    tf_idx_used = sorted(top_set)
    # Column order: by total loading across all programs
    total_load = H3[:, tf_idx_used, :].sum(axis=(0, 2))
    order = np.argsort(-total_load)
    tf_idx_used = [tf_idx_used[i] for i in order]
    tf_lbl = [tf_names[i] for i in tf_idx_used]
    n_show = len(tf_idx_used)

    # Shared colour scale
    vmax = float(np.quantile(H3[:, tf_idx_used, :], 0.99))
    if vmax <= 0:
        vmax = float(H3.max() or 1.0)

    fig, axes = plt.subplots(
        K, 1, figsize=(max(10, 0.18 * n_show + 2.5), 1.6 * K + 1.0),
        sharex=True, squeeze=False,
    )
    for p in range(K):
        ax = axes[p, 0]
        grid = H3[p, tf_idx_used, :].T          # shape: (N_BINS, n_show)
        im = ax.imshow(grid, aspect="auto", cmap="magma",
                       vmin=0, vmax=vmax, interpolation="nearest")
        ax.set_yticks(range(N_BINS))
        ax.set_yticklabels(BIN_LABELS, fontsize=8)
        ax.set_ylabel(f"P{p+1}", rotation=0, ha="right", va="center",
                      fontsize=11, fontweight="bold")
        if p == K - 1:
            ax.set_xticks(range(n_show))
            ax.set_xticklabels(tf_lbl, rotation=90, fontsize=7)
        else:
            ax.set_xticks([])

    cbar = fig.colorbar(im, ax=axes.ravel().tolist(),
                        fraction=0.012, pad=0.01)
    cbar.set_label("H loading")
    fig.suptitle(f"NMF program × TF × bin loadings (k={K}, "
                 f"top-{top_tfs_n} TFs/program)", y=0.995)
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def plot_program_sizes(W: np.ndarray, out_stem: str):
    K = W.shape[1]
    dom = W.argmax(axis=1)
    counts = np.bincount(dom, minlength=K)
    fig, ax = plt.subplots(figsize=(max(5, 0.6 * K + 2), 4))
    ax.bar(np.arange(K) + 1, counts, color="steelblue", edgecolor="black",
           linewidth=0.5)
    ax.set_xlabel("Program")
    ax.set_ylabel("Promoters with this program dominant")
    ax.set_title(f"Program sizes (k={K}, n_tss={W.shape[0]})")
    ax.set_xticks(np.arange(K) + 1)
    for i, c in enumerate(counts):
        ax.text(i + 1, c, f"{c}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_reconstruction_error(errs: dict, n_tss: int, n_feat: int, nnz: int,
                              out_stem: str):
    ks = sorted(errs)
    vals = [errs[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, "-o", color="C3")
    ax.set_xlabel("k (n_components)")
    ax.set_ylabel("Frobenius reconstruction error")
    ax.set_title(f"NMF error vs k  (n_tss={n_tss}, n_feat={n_feat}, "
                 f"nnz={nnz:,})")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_program_vs_tfcluster(H: np.ndarray, tf_names: list,
                              tf_cluster_fn: Path, top_n: int,
                              out_stem: str):
    """For each program, count how many of its top-N features (TF.bin) fall in
    each filtered-K8 TF cluster (by TF identity, ignoring bin)."""
    if not tf_cluster_fn.exists():
        return None
    tab = pd.read_csv(tf_cluster_fn, sep="\t")
    tf_to_cluster = dict(zip(tab["TF"], tab["cluster"]))
    clusters_present = sorted(set(tab["cluster"]))

    K = H.shape[0]
    n_tf = len(tf_names)
    M = np.zeros((K, len(clusters_present)), dtype=np.int32)
    feats = feature_names(tf_names)
    for p in range(K):
        idx = np.argsort(H[p])[::-1][:top_n]
        for j in idx:
            tf = feats[j].rsplit(".", 1)[0]
            c = tf_to_cluster.get(tf)
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
    ax.set_title(f"Program top-{top_n} features by TF cluster (k={K})")
    if M.max():
        for p in range(K):
            for c in range(len(clusters_present)):
                if M[p, c] > 0:
                    ax.text(c, p, str(M[p, c]), ha="center", va="center",
                            color="black" if M[p, c] < M.max() / 2 else "white",
                            fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"# of top-{top_n} features in cluster")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


################################################################################
# Execution ####################################################################
################################################################################
def main():
    out_root = OUT_DN / "tss_programs_multibin"
    plots_dn = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dn.mkdir(parents=True, exist_ok=True)
    (OUT_DN / "logs").mkdir(parents=True, exist_ok=True)
    log_path = OUT_DN / "logs" / f"tss_programs_multibin.{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    log, log_fh = _make_logger(str(log_path))

    log(f"WORKERS = {WORKERS}")
    log(f"wide window = +/- {WIDE_HALF} bp")
    log(f"bins = {BIN_LABELS}")
    log(f"KS = {KS}")

    npz_path     = out_root / f"occupancy.wide{WIDE_HALF}_b{N_BINS}.npz"
    feat_idx_fn  = out_root / "feature_index.tsv"
    tss_fn       = out_root / "tss_table.tsv"
    tf_fn        = out_root / "tf_index.tsv"

    if RESUME_FROM_NPZ and all(p.exists() for p in [npz_path, feat_idx_fn,
                                                     tss_fn, tf_fn]):
        log(f"resuming from {npz_path.name}")
        M = sp.load_npz(str(npz_path)).tocsr()
        tss_df = pd.read_csv(tss_fn, sep="\t")
        n_tss = len(tss_df)
        tf_names = pd.read_csv(tf_fn, sep="\t").sort_values("tf_idx")["TF"].tolist()
        n_tf = len(tf_names)
        log(f"loaded M: shape={M.shape}  nnz={M.nnz:,}")
    else:
        # 1) Canonical TSSs
        tss_df = load_canonical_tss(GTF_FN, VALID_CHROMS)
        n_tss = len(tss_df)
        log(f"n_tss = {n_tss:,}")
        tss_df.to_csv(tss_fn, sep="\t", index=False)

        # 2) Wide TSS windows
        windows_df = pd.DataFrame({
            "Chromosome": tss_df["chrom"].values,
            "Start":      np.maximum(tss_df["tss"].values - WIDE_HALF, 0).astype(np.int32),
            "End":        (tss_df["tss"].values + WIDE_HALF + 1).astype(np.int32),
            "Strand":     tss_df["strand"].values,
            "tss_pos":    tss_df["tss"].values.astype(np.int32),
            "tss_id":     np.arange(n_tss, dtype=np.int32),
        })

        # 3) Discover TFs
        tf_axis = tf_name_set()
        tf_files = discover_tf_files(PER_TF_DN, tf_axis)
        n_tf = len(tf_files)
        log(f"n_tf  = {n_tf}")
        if n_tf == 0:
            raise RuntimeError("No TF files matched DNA-binding whitelist")
        tf_args = [(name, path, idx) for idx, (name, path) in enumerate(tf_files)]

        # 4) Per-TF accumulation
        t0 = time.time()
        if WORKERS == 1:
            _init_worker(windows_df, VALID_CHROMS)
            results = [accumulate_tf_multibin(a) for a in tf_args]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=WORKERS,
                          initializer=_init_worker,
                          initargs=(windows_df, VALID_CHROMS)) as pool:
                results = []
                for i, r in enumerate(pool.imap_unordered(accumulate_tf_multibin,
                                                          tf_args, chunksize=4), 1):
                    results.append(r)
                    if i % 100 == 0 or i == n_tf:
                        log(f"  processed {i}/{n_tf} TFs "
                            f"(last: {r['tf']} — {r['n_pairs']:,} unique "
                            f"(tss,bin), {r['runtime_s']:.2f}s)")
                        if r.get("error"):
                            log(f"    [ERROR] {r['tf']}: {r['error']}")
        log(f"per-TF scan done in {time.time() - t0:.1f}s")

        results.sort(key=lambda r: r["tf_idx"])
        tf_names = [r["tf"] for r in results]

        # 5) Build sparse [n_tss x n_tf*N_BINS] occupancy matrix
        M = build_occupancy_matrix(results, n_tss, n_tf)
        log(f"M shape = {M.shape}  nnz = {M.nnz:,}  "
            f"density = {M.nnz / (M.shape[0] * M.shape[1]):.4f}")

        sp.save_npz(str(npz_path), M)
        pd.Series(tf_names, name="TF").to_csv(tf_fn, sep="\t",
                                              index_label="tf_idx")
        # feature index
        feat_rows = []
        for j, tf in enumerate(tf_names):
            for b, lab in enumerate(BIN_LABELS):
                feat_rows.append({"feature_idx": j * N_BINS + b,
                                  "tf": tf, "bin": lab})
        pd.DataFrame(feat_rows).to_csv(feat_idx_fn, sep="\t", index=False)

    nnz = M.nnz
    n_feat = M.shape[1]

    # Per-bin density (sanity)
    col_sum = np.asarray(M.sum(axis=0)).ravel()
    per_bin = np.array([col_sum.reshape(n_tf, N_BINS)[:, b].sum()
                        for b in range(N_BINS)])
    log("per-bin total occupancy across TFs:")
    for b, lab in enumerate(BIN_LABELS):
        log(f"  {lab:11s}  sum={int(per_bin[b]):,}  "
            f"mean_per_TF={per_bin[b]/n_tf:.1f}")

    # 6) NMF for each k
    feat_names = feature_names(tf_names)
    errs = {}
    for k in KS:
        log(f"--- NMF k={k} ---")
        t1 = time.time()
        W, H, err = run_nmf(M, k)
        W, H = relabel_programs_by_size(W, H)
        errs[k] = err
        log(f"  k={k}: err={err:.2f}  fit_time={time.time() - t1:.1f}s")

        # Save W, H
        pd.DataFrame(W,
                     index=tss_df["transcript_id"].values,
                     columns=[f"prog{p+1}" for p in range(k)]
        ).to_csv(out_root / f"nmf.k{k}.W.tsv.gz", sep="\t",
                 compression="gzip", index_label="transcript_id")
        pd.DataFrame(H,
                     index=[f"prog{p+1}" for p in range(k)],
                     columns=feat_names
        ).to_csv(out_root / f"nmf.k{k}.H.tsv.gz", sep="\t",
                 compression="gzip", index_label="program")

        # Top features per program
        top_features_table(H, feat_names, TOP_FEATURES_PER_K).to_csv(
            out_root / f"nmf.k{k}.top_features.tsv", sep="\t", index=False)

        # Long TF×bin loadings per program (only nonzero)
        program_tf_bin_long(H, tf_names).to_csv(
            out_root / f"nmf.k{k}.program_tf_bin.tsv.gz",
            sep="\t", compression="gzip", index=False)

        # TSS assignment
        tss_assignment_table(W, tss_df).to_csv(
            out_root / f"nmf.k{k}.tss_assignment.tsv", sep="\t", index=False)

        # Plots
        plot_program_tf_bin_grid(H, tf_names, TOP_TFS_FOR_GRID,
                                 str(plots_dn / f"program_tf_bin.k{k}"))
        plot_program_sizes(W, str(plots_dn / f"program_sizes.k{k}"))
        plot_program_vs_tfcluster(
            H, tf_names, TF_CLUSTER_FN, TOP_FEATURES_PER_K,
            str(plots_dn / f"program_vs_tfcluster.k{k}"))

        # Quick log readout: top 8 features per program
        ts = top_features_table(H, feat_names, 8)
        for p in range(k):
            tfs_p = ts.loc[ts["program"] == p + 1, "feature"].tolist()
            n_dom = int((W.argmax(axis=1) == p).sum())
            log(f"  P{p+1} (n_dom={n_dom:5d}): {', '.join(tfs_p)}")

    # 7) Error vs k
    plot_reconstruction_error(errs, n_tss, n_feat, nnz,
                              str(plots_dn / "nmf_reconstruction_error"))
    pd.DataFrame({"k": list(errs.keys()),
                  "frobenius_err": list(errs.values())}).to_csv(
        out_root / "nmf_reconstruction_error.tsv", sep="\t", index=False)

    log("DONE")
    log_fh.close()


if __name__ == "__main__":
    main()
