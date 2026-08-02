#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
TSS-binding programs by NMF on a per-TSS × per-TF binary occupancy matrix.

For every canonical protein-coding TSS we ask: which TFs (chip-atlas) have at
least one peak landing in the core promoter window (TSS +/- CORE_HALF bp)?
That gives an [n_tss x n_tf] binary matrix M. Non-negative matrix factorization
on M yields:
    W [n_tss x K]   each TSS's mixture over K binding programs
    H [K x n_tf]    each program's TF loading profile
Programs are interpreted by their top-loading TFs and their assigned promoters.

Why NMF: occupancy is non-negative, NMF gives a parts-based decomposition that
is naturally interpretable as additive programs (vs PCA's signed components).

Outputs (tss_programs/):
    occupancy.core{H}.npz             # CSR M, tss_df, tf_names
    nmf.k{K}.W.tsv.gz                 # n_tss x K  (per-TSS program weights)
    nmf.k{K}.H.tsv.gz                 # K x n_tf   (per-program TF loadings)
    nmf.k{K}.top_tfs.tsv              # top 30 TFs per program
    nmf.k{K}.tss_assignment.tsv       # gene-level dominant program + weights
    nmf.k{K}.program_summary.tsv      # per-program: n_promoters, top TFs, top genes
    plots/
        program_tf_loadings.k{K}.{png,pdf}     # H heatmap, top TFs ordered by program
        program_sizes.k{K}.{png,pdf}           # bar of dominant-program counts
        nmf_reconstruction_error.{png,pdf}     # frobenius err vs K
        program_vs_tfcluster.k{K}.{png,pdf}    # cross-tab: program top-TFs ∩ filtered TF clusters
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
# Cap BLAS threads BEFORE importing numpy/sklearn to prevent the multi-thread
# storm + futex contention seen with default OpenBLAS settings (NMF k=5 stuck at
# 1% CPU for 25 min).
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
TF_CLUSTER_FN       = OUT_DN / "clustering" / "tf_cluster_table.tsv"   # filtered K=8

CORE_HALF           = 100        # core promoter half-window
PEAK_RECENTER_HALF  = 12         # 25-nt peak block, matches main pipeline
VALID_CHROMS        = {str(c) for c in list(range(1, 23)) + ["X", "Y", "MT"]}

KS                  = [5, 8, 12, 15]
NMF_MAX_ITER        = 300
NMF_RANDOM_STATE    = 0
TOP_TFS_PER_PROGRAM = 30
TOP_GENES_PER_PROG  = 100
RESUME_FROM_NPZ     = True       # if occupancy.core{H}.npz exists, skip BED scan

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
# Base-level Functions #########################################################
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
# Task-specific Functions ######################################################
################################################################################
def load_canonical_tss(gtf_fn: str, valid_chroms: set) -> pd.DataFrame:
    """Same loader as canonical_promoter_aggregate.001.py, copied here so this
    script is self-contained."""
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
    out = pd.DataFrame({
        "chrom":         tx.loc[keep, "seqid"].astype(str).values,
        "tss":           tss,
        "strand":        tx.loc[keep, "strand"].astype(str).values,
        "gene_id":       gene_id[keep].values,
        "gene_name":     gene_name[keep].values,
        "transcript_id": transcript_id[keep].values,
    }).reset_index(drop=True)
    return out




# Worker globals (populated by _init_worker) ----------------------------------
_TSS_WINDOWS_PR = None
_VALID_CHROMS   = None


def _init_worker(tss_windows_df: pd.DataFrame, valid_chroms: set):
    global _TSS_WINDOWS_PR, _VALID_CHROMS
    _TSS_WINDOWS_PR = pr.PyRanges(tss_windows_df)
    _VALID_CHROMS = valid_chroms


def accumulate_tf_core(args) -> dict:
    """
    Per-TF worker. Returns the unique set of tss_ids bound in the core window.
    Binary occupancy: a TF binds a TSS iff at least one of its (recentered)
    peaks overlaps the TSS's core window.
    """
    tf_name, bed_paths, tf_idx = args
    t0 = time.time()
    tss_ids_bound = np.empty(0, dtype=np.int32)
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
            peaks_df["Start"] = np.maximum(mid - PEAK_RECENTER_HALF, 0).astype(np.int32)
            peaks_df["End"]   = (mid + PEAK_RECENTER_HALF + 1).astype(np.int32)

            peaks_pr = pr.PyRanges(peaks_df)
            ov = peaks_pr.join(
                _TSS_WINDOWS_PR, suffix="_w",
                strandedness=False, apply_strand_suffix=False,
            ).df
            if not ov.empty:
                tss_ids_bound = np.unique(ov["tss_id"].to_numpy(np.int32))
    except Exception as e:
        err = f"{type(e).__name__}: {e}"

    return {"tf": tf_name, "tf_idx": tf_idx, "tss_ids": tss_ids_bound,
            "n_peaks_kept": n_peaks_kept, "n_bound_tss": int(tss_ids_bound.size),
            "runtime_s": time.time() - t0, "error": err}


def build_occupancy_matrix(results: list, n_tss: int, n_tf: int) -> sp.csr_matrix:
    """COO -> CSR sparse matrix from per-TF tss_id lists."""
    row_chunks, col_chunks = [], []
    for r in results:
        ids = r["tss_ids"]
        if ids.size:
            row_chunks.append(ids)
            col_chunks.append(np.full(ids.size, r["tf_idx"], dtype=np.int32))
    if not row_chunks:
        return sp.csr_matrix((n_tss, n_tf), dtype=np.float32)
    rows = np.concatenate(row_chunks)
    cols = np.concatenate(col_chunks)
    data = np.ones(rows.size, dtype=np.float32)
    M = sp.coo_matrix((data, (rows, cols)), shape=(n_tss, n_tf)).tocsr()
    M.sum_duplicates()    # safety: all 1s anyway
    return M


def run_nmf(M, k: int) -> tuple:
    """Returns (W, H, err). MU solver + random init: avoids the dense SVD that
    nndsvd needs (which thread-thrashed with default OpenBLAS at k=5)."""
    nmf = NMF(n_components=k, init="random",
              max_iter=NMF_MAX_ITER, random_state=NMF_RANDOM_STATE,
              beta_loss="frobenius", solver="mu", tol=1e-4)
    W = nmf.fit_transform(M)
    H = nmf.components_
    return W, H, float(nmf.reconstruction_err_)


def relabel_programs_by_size(W: np.ndarray, H: np.ndarray) -> tuple:
    """Order programs from most-used to least-used (by dominant-assignment count)."""
    dominant = W.argmax(axis=1)
    counts = np.bincount(dominant, minlength=W.shape[1])
    order = np.argsort(-counts)
    return W[:, order], H[order, :]


def top_tfs_table(H: np.ndarray, tf_names: list, top_n: int) -> pd.DataFrame:
    rows = []
    for p in range(H.shape[0]):
        idx = np.argsort(H[p])[::-1][:top_n]
        for rank, j in enumerate(idx, 1):
            rows.append({"program": p + 1, "rank": rank,
                         "tf": tf_names[j], "loading": float(H[p, j])})
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


def program_summary_table(W: np.ndarray, H: np.ndarray, tf_names: list,
                          tss_df: pd.DataFrame, top_tfs_n: int = 10,
                          top_genes_n: int = 20) -> pd.DataFrame:
    K = W.shape[1]
    dom = W.argmax(axis=1)
    rows = []
    for p in range(K):
        member_mask = (dom == p)
        member_w    = W[member_mask, p]
        member_genes = tss_df.loc[member_mask, "gene_name"].fillna("").astype(str).values
        # top genes by their weight on this program
        if member_genes.size:
            order = np.argsort(member_w)[::-1][:top_genes_n]
            top_genes = ",".join(member_genes[order])
        else:
            top_genes = ""
        # top TFs by H
        idx = np.argsort(H[p])[::-1][:top_tfs_n]
        top_tfs = ",".join(tf_names[j] for j in idx)
        rows.append({
            "program": p + 1,
            "n_promoters_dominant": int(member_mask.sum()),
            "mean_loading_when_dominant": float(member_w.mean()) if member_w.size else 0.0,
            "top_tfs": top_tfs,
            "top_genes": top_genes,
        })
    return pd.DataFrame(rows)


# ---- Plotting ---------------------------------------------------------------
def plot_program_tf_heatmap(H: np.ndarray, tf_names: list,
                            top_n_per_program: int, out_stem: str):
    """Heatmap of H restricted to union(top_n) TFs per program; columns ordered
    by their dominant program then by loading."""
    top_set = set()
    for p in range(H.shape[0]):
        top_set.update(np.argsort(H[p])[::-1][:top_n_per_program].tolist())
    cols = sorted(top_set)
    H_sub = H[:, cols]                                          # K x C
    dom_prog = H_sub.argmax(axis=0)
    dom_load = H_sub[dom_prog, np.arange(H_sub.shape[1])]
    order = np.lexsort((-dom_load, dom_prog))
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
    ax.set_xlabel(f"TFs (union of top {top_n_per_program} per program, "
                  f"n={len(tf_ord)})")
    ax.set_ylabel("Program")
    ax.set_title(f"NMF program × TF loadings (k={H.shape[0]})")
    cbar = fig.colorbar(im, ax=ax, fraction=0.02, pad=0.01)
    cbar.set_label("H loading")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
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


def plot_reconstruction_error(errs: dict, n_tss: int, n_tf: int, nnz: int,
                              out_stem: str):
    ks = sorted(errs)
    vals = [errs[k] for k in ks]
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ks, vals, "-o", color="C3")
    ax.set_xlabel("k (n_components)")
    ax.set_ylabel("Frobenius reconstruction error")
    ax.set_title(f"NMF error vs k  (n_tss={n_tss}, n_tf={n_tf}, nnz={nnz:,})")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)


def plot_program_vs_tfcluster(H: np.ndarray, tf_names: list,
                              tf_cluster_fn: Path, top_n: int,
                              out_stem: str):
    """For each program, count how many of its top-N TFs fall in each
    filtered-K8 TF cluster. Heatmap of program × cluster counts."""
    if not tf_cluster_fn.exists():
        return None
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
    for p in range(K):
        for c in range(len(clusters_present)):
            if M[p, c] > 0:
                ax.text(c, p, str(M[p, c]), ha="center", va="center",
                        color="black" if M[p, c] < M.max() / 2 else "white",
                        fontsize=8)
    cbar = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(f"# of top-{top_n} program TFs in cluster")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300)
    fig.savefig(out_stem + ".pdf")
    plt.close(fig)
    return pd.DataFrame(M,
                        index=[f"P{p+1}" for p in range(K)],
                        columns=[f"TFc{c}" for c in clusters_present])


################################################################################
# Execution ####################################################################
################################################################################
def main():
    out_root = OUT_DN / "tss_programs"
    plots_dn = out_root / "plots"
    out_root.mkdir(parents=True, exist_ok=True)
    plots_dn.mkdir(parents=True, exist_ok=True)
    (OUT_DN / "logs").mkdir(parents=True, exist_ok=True)
    log_path = OUT_DN / "logs" / f"tss_programs.{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    log, log_fh = _make_logger(str(log_path))

    log(f"WORKERS = {WORKERS}")
    log(f"core window = +/- {CORE_HALF} bp")
    log(f"KS = {KS}")

    npz_path = out_root / f"occupancy.core{CORE_HALF}.npz"
    tf_index_path = out_root / "tf_index.tsv"

    if RESUME_FROM_NPZ and npz_path.exists() and tf_index_path.exists():
        log(f"resuming from {npz_path.name} (skipping BED scan)")
        M = sp.load_npz(str(npz_path)).tocsr()
        tss_df = pd.read_csv(out_root / "tss_table.tsv", sep="\t")
        n_tss = len(tss_df)
        tf_names = pd.read_csv(tf_index_path, sep="\t").sort_values("tf_idx")["TF"].tolist()
        n_tf = len(tf_names)
        log(f"loaded M: shape={M.shape}  nnz={M.nnz:,}")
    else:
        # 1) Canonical TSSs
        tss_df = load_canonical_tss(GTF_FN, VALID_CHROMS)
        n_tss = len(tss_df)
        log(f"n_tss = {n_tss:,}")
        tss_df.to_csv(out_root / "tss_table.tsv", sep="\t", index=False)

        # 2) TSS windows for join (small core window)
        windows_df = pd.DataFrame({
            "Chromosome": tss_df["chrom"].values,
            "Start":      np.maximum(tss_df["tss"].values - CORE_HALF, 0).astype(np.int32),
            "End":        (tss_df["tss"].values + CORE_HALF + 1).astype(np.int32),
            "Strand":     tss_df["strand"].values,
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

        # 4) Per-TF accumulation -> unique tss_ids bound
        t0 = time.time()
        if WORKERS == 1:
            _init_worker(windows_df, VALID_CHROMS)
            results = [accumulate_tf_core(a) for a in tf_args]
        else:
            ctx = mp.get_context("fork")
            with ctx.Pool(processes=WORKERS,
                          initializer=_init_worker,
                          initargs=(windows_df, VALID_CHROMS)) as pool:
                results = []
                for i, r in enumerate(pool.imap_unordered(accumulate_tf_core,
                                                          tf_args, chunksize=4), 1):
                    results.append(r)
                    if i % 100 == 0 or i == n_tf:
                        log(f"  processed {i}/{n_tf} TFs "
                            f"(last: {r['tf']} — {r['n_bound_tss']:,} bound TSSs, "
                            f"{r['runtime_s']:.2f}s)")
                        if r.get("error"):
                            log(f"    [ERROR] {r['tf']}: {r['error']}")
        log(f"per-TF scan done in {time.time() - t0:.1f}s")

        results.sort(key=lambda r: r["tf_idx"])
        tf_names = [r["tf"] for r in results]

        # 5) Build sparse [n_tss x n_tf] occupancy matrix
        M = build_occupancy_matrix(results, n_tss, n_tf)
        log(f"M shape = {M.shape}  nnz = {M.nnz:,}  density = {M.nnz/(n_tss*n_tf):.4f}")
        log(f"per-TSS bound-TF count: median={np.median(np.asarray(M.sum(axis=1))):.0f} "
            f"mean={float(M.sum(axis=1).mean()):.1f} "
            f"max={int(M.sum(axis=1).max())}")
        log(f"per-TF bound-TSS count: median={np.median(np.asarray(M.sum(axis=0))):.0f} "
            f"mean={float(M.sum(axis=0).mean()):.1f} "
            f"max={int(M.sum(axis=0).max())}")

        # Save sparse matrix + index
        sp.save_npz(str(npz_path), M)
        pd.Series(tf_names, name="TF").to_csv(tf_index_path,
                                              sep="\t", index_label="tf_idx")

        # Per-TF bound-TSS summary
        summary = pd.DataFrame([{
            "TF":            r["tf"],
            "tf_idx":        r["tf_idx"],
            "n_peaks_kept":  r["n_peaks_kept"],
            "n_bound_tss":   r["n_bound_tss"],
            "frac_tss":      r["n_bound_tss"] / max(n_tss, 1),
            "runtime_s":     round(r["runtime_s"], 3),
            "error":         r["error"] or "",
        } for r in results]).sort_values("n_bound_tss", ascending=False)
        summary.to_csv(out_root / f"tf_summary.core{CORE_HALF}.tsv",
                       sep="\t", index=False)

    nnz = M.nnz

    # 6) NMF for each k
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
                     columns=tf_names
        ).to_csv(out_root / f"nmf.k{k}.H.tsv.gz", sep="\t",
                 compression="gzip", index_label="program")

        # Top TFs per program
        top_tfs_table(H, tf_names, TOP_TFS_PER_PROGRAM).to_csv(
            out_root / f"nmf.k{k}.top_tfs.tsv", sep="\t", index=False)

        # TSS-level assignment
        assignment = tss_assignment_table(W, tss_df)
        assignment.to_csv(out_root / f"nmf.k{k}.tss_assignment.tsv",
                          sep="\t", index=False)

        # Per-program summary
        program_summary_table(W, H, tf_names, tss_df,
                              top_tfs_n=10, top_genes_n=20).to_csv(
            out_root / f"nmf.k{k}.program_summary.tsv", sep="\t", index=False)

        # Plots
        plot_program_tf_heatmap(
            H, tf_names, TOP_TFS_PER_PROGRAM,
            str(plots_dn / f"program_tf_loadings.k{k}"))
        plot_program_sizes(W, str(plots_dn / f"program_sizes.k{k}"))
        plot_program_vs_tfcluster(
            H, tf_names, TF_CLUSTER_FN, TOP_TFS_PER_PROGRAM,
            str(plots_dn / f"program_vs_tfcluster.k{k}"))

        # Quick log readout
        ts = top_tfs_table(H, tf_names, 8)
        for p in range(k):
            tfs_p = ts.loc[ts["program"] == p + 1, "tf"].tolist()
            n_dom = int((W.argmax(axis=1) == p).sum())
            log(f"  P{p+1} (n_dom={n_dom:5d}): {', '.join(tfs_p)}")

    # 7) Error vs k
    plot_reconstruction_error(errs, n_tss, n_tf, nnz,
                              str(plots_dn / "nmf_reconstruction_error"))
    pd.DataFrame({"k": list(errs.keys()),
                  "frobenius_err": list(errs.values())}).to_csv(
        out_root / "nmf_reconstruction_error.tsv", sep="\t", index=False)

    log("DONE")
    log_fh.close()


if __name__ == "__main__":
    main()
