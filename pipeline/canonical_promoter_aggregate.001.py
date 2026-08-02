#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Canonical-promoter TF aggregate plots.

Maps every annotated-TF ChIP-seq peakset (chip-atlas) onto the TSSs of all
canonical protein-coding transcripts (Ensembl GRCh38.114) and produces three
+/- 1000 bp aggregate plots, each in two signal flavors (binary occupancy and
raw peak score):

    1. heatmap          rows = TFs, cols = bp from TSS
    2. metaplot_per_tf  one translucent line per TF
    3. metaplot_aggregate  single line = mean across TFs (+/- SEM band)

Intermediate matrices (parquet + tsv.gz) are written so the sibling R script
can re-plot without re-scanning BEDs.
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
import time
import datetime as dt
import multiprocessing as mp
from pathlib import Path

import numpy as np
import pandas as pd
import pyranges as pr
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import gaussian_filter1d

# Machine-specific paths and build axes -> pipeline/config.py
from config import (normalize_chrom, discover_tf_files, read_peak_beds,
                    tf_name_set, TF_SET, DNA_BINDING_FN, GTF_FN, OUT_DN,
                    PER_TF_DN)


################################################################################
# Initiating Variables #########################################################
################################################################################

HALF                = 1000
LEN                 = 2 * HALF + 1
VALID_CHROMS        = {str(c) for c in list(range(1, 23)) + ["X", "Y", "MT"]}
HIGHLIGHT_TFS       = ["CTCF", "YY1", "MYC", "SP1", "NRF1", "REST", "TBP", "EP300"]
PEAK_RECENTER_HALF  = 12          # 25-nt peaks: [mid - 12, mid + 13)

# Worker count. SLURM_CPUS_PER_TASK wins on HPC: os.cpu_count() reports the whole
# node there (384 cores on a Roihu CPU node) while the cgroup owns only what was
# requested. Locally, cap at 12 to leave headroom and limit sustained thermal load.
# Override explicitly with HPA_WORKERS; set to 1 for serial debug.
WORKERS             = max(1, int(os.environ.get(
    "HPA_WORKERS",
    os.environ.get("SLURM_CPUS_PER_TASK") or min(12, (os.cpu_count() or 2) - 2))))

# rcParams (matches sibling chip-atlas scripts) --------------------------------
sns.set_style("whitegrid")
plt.rcParams["font.size"]        = 11
plt.rcParams["axes.labelsize"]   = 12
plt.rcParams["axes.titlesize"]   = 14
plt.rcParams["figure.dpi"]       = 100


################################################################################
# Base-level Functions #########################################################
################################################################################


################################################################################
# Task-specific Functions ######################################################
################################################################################
def load_canonical_tss(gtf_fn: str, valid_chroms: set) -> pd.DataFrame:
    """
    Filter GTF to Ensembl_canonical protein-coding transcripts on standard
    chromosomes and compute TSS = start (forward) or end (reverse).

    Single-pass parser — read_ensembl_gtf() collapses duplicate 'tag' attributes
    (json_normalize keeps only the last value), so tags are re-extracted directly
    from the raw attributes column with re.findall.
    """
    print(f"[{_ts()}] loading GTF: {gtf_fn}")
    raw = pd.read_csv(
        gtf_fn, sep="\t", comment="#", header=None, low_memory=False,
        names=["seqid", "source", "feature", "start", "end", "score",
               "strand", "frame", "attributes"],
        dtype={"seqid": "string", "feature": "category",
               "start": "int32", "end": "int32",
               "strand": "string", "attributes": "string"},
    )
    print(f"[{_ts()}]   {len(raw):,} GTF rows")

    tx = raw[raw["feature"] == "transcript"].copy()
    print(f"[{_ts()}]   {len(tx):,} transcript rows")

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
    print(f"[{_ts()}]   {len(out):,} canonical protein-coding TSSs retained")
    return out




# Worker globals (populated by _init_worker) ----------------------------------
_TSS_WINDOWS_PR = None      # pr.PyRanges built once and shared via fork
_VALID_CHROMS   = None


def _init_worker(tss_windows_df: pd.DataFrame, valid_chroms: set):
    global _TSS_WINDOWS_PR, _VALID_CHROMS
    _TSS_WINDOWS_PR = pr.PyRanges(tss_windows_df)
    _VALID_CHROMS = valid_chroms


def _project_local(ov: pd.DataFrame) -> tuple:
    """
    Given the result of a peaks×TSS_windows pyranges join, project each
    peak-window overlap interval into TSS-local frame [0, LEN), flipping
    negative-strand transcripts so upstream = negative offset.
    Returns (loc_lo, loc_hi) numpy arrays — half-open, suitable for
    diff-encoded cumsum via np.bincount.
    """
    gs = np.maximum(ov["Start"].to_numpy(np.int32),
                    ov["Start_w"].to_numpy(np.int32))
    ge = np.minimum(ov["End"].to_numpy(np.int32),
                    ov["End_w"].to_numpy(np.int32))     # exclusive
    tss = ov["tss_pos"].to_numpy(np.int32)
    sp  = (ov["Strand"].to_numpy() == "+")
    loc_lo = np.where(sp, gs - tss + HALF,
                          tss - (ge - 1) + HALF).astype(np.int64)
    loc_hi = np.where(sp, ge - tss + HALF,
                          tss - gs + HALF + 1).astype(np.int64)
    return np.clip(loc_lo, 0, LEN), np.clip(loc_hi, 0, LEN)


def accumulate_tf(args) -> dict:
    """
    Per-TF worker. Returns sum vectors (length LEN) for binary and score signals
    plus per-TF stats. Mean (sum / n_tss) is computed in the parent.
    """
    tf_name, bed_paths = args
    t0 = time.time()

    binary_sum   = np.zeros(LEN, dtype=np.int64)
    score_sum    = np.zeros(LEN, dtype=np.float64)
    raw_sum      = np.zeros(LEN, dtype=np.int64)
    raw1000_sum  = np.zeros(LEN, dtype=np.int64)
    n_peaks_total = 0
    n_peaks_kept  = 0
    n_overlaps    = 0
    n_peaks_1000  = 0

    try:
        peaks_df = read_peak_beds(bed_paths)
        n_peaks_total = len(peaks_df)
        peaks_df["Chromosome"] = normalize_chrom(peaks_df["Chromosome"])
        peaks_df = peaks_df[peaks_df["Chromosome"].isin(_VALID_CHROMS)].copy()
        n_peaks_kept = len(peaks_df)

        # Recenter every peak to PEAK_RECENTER_HALF*2 + 1 nt around its midpoint.
        # Many chip-atlas peaks span hundreds of bp; collapsing to a fixed-size
        # window around the peak center gives a sharper signal and avoids long
        # peaks contributing redundant bp's far from the actual binding site.
        if n_peaks_kept:
            mid = ((peaks_df["Start"].astype(np.int64)
                  + peaks_df["End"].astype(np.int64)) // 2).astype(np.int32)
            peaks_df["Start"] = np.maximum(mid - PEAK_RECENTER_HALF, 0).astype(np.int32)
            peaks_df["End"]   = (mid + PEAK_RECENTER_HALF + 1).astype(np.int32)

        if n_peaks_kept:
            peaks_pr = pr.PyRanges(peaks_df)

            # --- BINARY signal: merge overlapping peaks per TF first so each
            #     bp×TSS gets counted at most once. binary_sum / n_tss is then
            #     the fraction of TSSs with any peak at that offset.
            merged_pr = peaks_pr.merge()
            ov_b = merged_pr.join(
                _TSS_WINDOWS_PR, suffix="_w",
                strandedness=False, apply_strand_suffix=False,
            ).df
            if not ov_b.empty:
                lo, hi = _project_local(ov_b)
                bin_diff = (np.bincount(lo, minlength=LEN + 1).astype(np.int64)
                          - np.bincount(hi, minlength=LEN + 1).astype(np.int64))
                binary_sum = np.cumsum(bin_diff)[:LEN].astype(np.int64)

            # --- SCORE signal: keep all peaks, sum scores. Sum reflects total
            #     binding evidence density (more experiments / stronger peaks).
            ov_s = peaks_pr.join(
                _TSS_WINDOWS_PR, suffix="_w",
                strandedness=False, apply_strand_suffix=False,
            ).df
            if not ov_s.empty:
                n_overlaps = len(ov_s)
                lo, hi = _project_local(ov_s)
                sc = ov_s["score"].to_numpy(np.float32)
                scr_diff = (np.bincount(lo, weights=sc, minlength=LEN + 1)
                          - np.bincount(hi, weights=sc, minlength=LEN + 1)).astype(np.float64)
                score_sum = np.cumsum(scr_diff)[:LEN].astype(np.float64)
                # Raw overlap count: each (peak × TSS) overlap contributes 1
                # at every covered bp (no per-TF merging, no score weighting).
                raw_diff = (np.bincount(lo, minlength=LEN + 1).astype(np.int64)
                          - np.bincount(hi, minlength=LEN + 1).astype(np.int64))
                raw_sum = np.cumsum(raw_diff)[:LEN].astype(np.int64)

            # --- RAW @ score==1000 (chip-atlas saturated / strongest peaks):
            #     same un-merged accumulation but gated to peaks with the
            #     capped maximum score.
            peaks_1000 = peaks_df[peaks_df["score"] == 1000.0]
            n_peaks_1000 = len(peaks_1000)
            if n_peaks_1000:
                ov_1k = pr.PyRanges(peaks_1000).join(
                    _TSS_WINDOWS_PR, suffix="_w",
                    strandedness=False, apply_strand_suffix=False,
                ).df
                if not ov_1k.empty:
                    lo, hi = _project_local(ov_1k)
                    diff = (np.bincount(lo, minlength=LEN + 1).astype(np.int64)
                          - np.bincount(hi, minlength=LEN + 1).astype(np.int64))
                    raw1000_sum = np.cumsum(diff)[:LEN].astype(np.int64)
    except Exception as e:
        return {"tf": tf_name, "error": f"{type(e).__name__}: {e}",
                "binary_sum": binary_sum, "score_sum": score_sum,
                "raw_sum": raw_sum, "raw1000_sum": raw1000_sum,
                "n_peaks_total": n_peaks_total, "n_peaks_kept": n_peaks_kept,
                "n_peaks_1000": n_peaks_1000,
                "n_overlaps": n_overlaps, "runtime_s": time.time() - t0}

    return {"tf": tf_name, "error": None,
            "binary_sum": binary_sum, "score_sum": score_sum,
            "raw_sum": raw_sum, "raw1000_sum": raw1000_sum,
            "n_peaks_total": n_peaks_total, "n_peaks_kept": n_peaks_kept,
            "n_peaks_1000": n_peaks_1000,
            "n_overlaps": n_overlaps, "runtime_s": time.time() - t0}


def write_matrices(results: list, n_tss: int, out_dn: str) -> tuple:
    """
    Convert per-TF sums to means (sum / n_tss). Write parquet + tsv.gz mirrors
    plus a tf_summary.tsv. Returns (binary_df, score_df).
    """
    matrices_dn = Path(out_dn) / "matrices"
    matrices_dn.mkdir(parents=True, exist_ok=True)

    cols = np.arange(-HALF, HALF + 1, dtype=np.int32)
    tfs  = [r["tf"] for r in results]

    binary_arr  = np.stack([r["binary_sum"]  for r in results]).astype(np.float64) / max(n_tss, 1)
    score_arr   = np.stack([r["score_sum"]   for r in results]).astype(np.float64) / max(n_tss, 1)
    raw_arr     = np.stack([r["raw_sum"]     for r in results]).astype(np.int64)
    raw1000_arr = np.stack([r["raw1000_sum"] for r in results]).astype(np.int64)

    binary_df  = pd.DataFrame(binary_arr,  index=tfs, columns=cols); binary_df.index.name  = "TF"
    score_df   = pd.DataFrame(score_arr,   index=tfs, columns=cols); score_df.index.name   = "TF"
    raw_df     = pd.DataFrame(raw_arr,     index=tfs, columns=cols); raw_df.index.name     = "TF"
    raw1000_df = pd.DataFrame(raw1000_arr, index=tfs, columns=cols); raw1000_df.index.name = "TF"

    # parquet
    binary_df .reset_index().to_parquet(matrices_dn / "tf_x_position.binary.parquet",       index=False)
    score_df  .reset_index().to_parquet(matrices_dn / "tf_x_position.score.parquet",        index=False)
    raw_df    .reset_index().to_parquet(matrices_dn / "tf_x_position.raw.parquet",          index=False)
    raw1000_df.reset_index().to_parquet(matrices_dn / "tf_x_position.raw_score1000.parquet", index=False)
    # tsv.gz mirror for R
    binary_df .to_csv(matrices_dn / "tf_x_position.binary.tsv.gz",       sep="\t", compression="gzip")
    score_df  .to_csv(matrices_dn / "tf_x_position.score.tsv.gz",        sep="\t", compression="gzip")
    raw_df    .to_csv(matrices_dn / "tf_x_position.raw.tsv.gz",          sep="\t", compression="gzip")
    raw1000_df.to_csv(matrices_dn / "tf_x_position.raw_score1000.tsv.gz", sep="\t", compression="gzip")

    # per-TF summary
    summary = pd.DataFrame([{
        "TF": r["tf"],
        "n_peaks_total":  r["n_peaks_total"],
        "n_peaks_kept":   r["n_peaks_kept"],
        "n_peaks_1000":   r["n_peaks_1000"],
        "n_overlaps":     r["n_overlaps"],
        "binary_total":   float(r["binary_sum"].sum()),
        "score_total":    float(r["score_sum"].sum()),
        "raw_total":      int(r["raw_sum"].sum()),
        "raw1000_total":  int(r["raw1000_sum"].sum()),
        "binary_max":     float(r["binary_sum"].max()),
        "score_max":      float(r["score_sum"].max()),
        "raw_max":        int(r["raw_sum"].max()),
        "raw1000_max":    int(r["raw1000_sum"].max()),
        "runtime_s":      round(r["runtime_s"], 3),
        "error":          r.get("error"),
    } for r in results]).sort_values("binary_total", ascending=False)
    summary.to_csv(matrices_dn / "tf_summary.tsv", sep="\t", index=False)

    return binary_df, score_df, raw_df, raw1000_df


# ---- Plotting ---------------------------------------------------------------
def _xticks(ax):
    ax.set_xticks([-1000, -500, 0, 500, 1000])
    ax.set_xlabel("Distance from TSS (bp)")


def plot_heatmap(matrix: pd.DataFrame, signal_label: str, cmap: str,
                 out_path_stem: str):
    """rows = TFs (sorted by total signal desc), cols = bp from TSS."""
    order = matrix.sum(axis=1).sort_values(ascending=False).index
    M = matrix.loc[order]

    n_tf = len(M)
    fig_h = max(4.0, 0.012 * n_tf)
    fig, ax = plt.subplots(figsize=(8, fig_h))
    vmin = 0
    vmax = float(np.quantile(M.values, 0.99)) if M.size else 1.0
    if vmax <= 0:
        vmax = float(M.values.max() or 1.0)

    mesh = ax.imshow(
        M.values, aspect="auto", origin="upper", interpolation="nearest",
        extent=[float(M.columns[0]), float(M.columns[-1]), n_tf, 0],
        cmap=cmap, vmin=vmin, vmax=vmax, rasterized=True,
    )
    ax.set_yticks([])
    ax.axvline(0, color="white", linestyle="--", linewidth=0.8, alpha=0.7)
    _xticks(ax)
    ax.set_ylabel(f"TFs (n={n_tf}, sorted by total signal)")
    ax.set_title(f"TF binding around canonical TSSs — {signal_label}")
    cbar = fig.colorbar(mesh, ax=ax, fraction=0.025, pad=0.02)
    cbar.set_label(signal_label)
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300)
    fig.savefig(out_path_stem + ".pdf")
    plt.close(fig)

    # Companion order file
    pd.Series(order, name="TF").to_csv(out_path_stem + ".tf_order.tsv",
                                       sep="\t", index=False)


def plot_metaplot_per_tf(matrix: pd.DataFrame, signal_label: str,
                         base_color: str, out_path_stem: str):
    """One translucent line per TF; highlight a curated subset."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = matrix.columns.values

    # Background: all TFs, very translucent
    for tf, row in matrix.iterrows():
        ax.plot(x, row.values, color=base_color, alpha=0.05, linewidth=0.4)

    # Highlights
    palette = sns.color_palette("tab10", n_colors=len(HIGHLIGHT_TFS))
    for tf, c in zip(HIGHLIGHT_TFS, palette):
        if tf in matrix.index:
            ax.plot(x, matrix.loc[tf].values, color=c, linewidth=2.0,
                    label=tf, alpha=0.95)

    ax.axvline(0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    _xticks(ax)
    ax.set_ylabel(signal_label)
    ax.set_title(f"Per-TF binding profile around canonical TSSs (n_TF={len(matrix)})")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9, title="Highlighted")
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300)
    fig.savefig(out_path_stem + ".pdf")
    plt.close(fig)


def plot_overlap_histogram_5nt(raw_matrix: pd.DataFrame, out_path_stem: str,
                               bin_width: int = 5, color: str = "steelblue",
                               title_suffix: str = ""):
    """
    5-nt-binned histogram of total peak×TSS overlaps summed across all TFs and
    all canonical TSSs. raw_matrix rows = per-TF raw overlap counts; we sum
    across rows to get a length-LEN per-bp count, then collapse into bin_width
    nt bins. The 2001-bp window does not divide evenly by 5, so the bp at
    +1000 is dropped for divisibility (400 bins × 5 nt = 2000 bp spanning
    -1000..+999).
    """
    per_bp = raw_matrix.sum(axis=0).values.astype(np.int64)   # length LEN = 2001

    half = bin_width // 2
    n_bins = LEN // bin_width                                 # 400 for 2001/5
    centers = np.arange(n_bins, dtype=np.int32) * bin_width - HALF + half
    truncated = per_bp[: n_bins * bin_width]
    binned = truncated.reshape(n_bins, bin_width).sum(axis=1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(centers, binned, width=bin_width, color=color,
           edgecolor="none", align="center")
    ax.axvline(0, color="red", linestyle="--", linewidth=0.9, alpha=0.7,
               label="TSS")
    ax.set_xticks([-1000, -500, 0, 500, 1000])
    ax.set_xlabel("Distance from TSS (bp)")
    ax.set_ylabel(f"Total overlaps per {bin_width} nt bin\n"
                  f"(summed across {raw_matrix.shape[0]} TFs and all canonical TSSs)")
    title = f"Overlap density around canonical TSSs ({bin_width}-nt bins)"
    if title_suffix:
        title += f" — {title_suffix}"
    ax.set_title(title)
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300)
    fig.savefig(out_path_stem + ".pdf")
    plt.close(fig)

    # Also write the binned counts as a tsv for downstream/R use
    pd.DataFrame({
        "bin_center_bp": centers,
        "bin_start_bp":  centers - half,
        "bin_end_bp":    centers + half + 1,    # exclusive
        "overlap_sum":   binned,
    }).to_csv(out_path_stem + ".tsv", sep="\t", index=False)


def plot_metaplot_aggregate(matrix: pd.DataFrame, signal_label: str,
                            line_color: str, out_path_stem: str):
    """Single mean-of-means line, +/- 1 SEM (across TFs), Gaussian-smoothed."""
    fig, ax = plt.subplots(figsize=(8, 5))
    x = matrix.columns.values
    M = matrix.values
    n_tf = len(M)

    mean = M.mean(axis=0)
    sem  = M.std(axis=0, ddof=1) / np.sqrt(max(n_tf, 1))
    mean_s = gaussian_filter1d(mean, sigma=10)
    sem_s  = gaussian_filter1d(sem,  sigma=10)

    ax.fill_between(x, mean_s - sem_s, mean_s + sem_s,
                    color=line_color, alpha=0.25, linewidth=0)
    ax.plot(x, mean_s, color=line_color, linewidth=2.0)
    ax.axvline(0, color="red", linestyle="--", linewidth=0.9, alpha=0.7,
               label="TSS")
    _xticks(ax)
    ax.set_ylabel(f"Mean across TFs of {signal_label}")
    ax.set_title(f"Aggregate TF binding around canonical TSSs (n_TF={n_tf})")
    ax.legend(loc="upper right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300)
    fig.savefig(out_path_stem + ".pdf")
    plt.close(fig)


# ---- Verification -----------------------------------------------------------
def verify(matrix_bin: pd.DataFrame, matrix_scr: pd.DataFrame,
           tss_df: pd.DataFrame, log) -> None:
    n_tss = len(tss_df)
    n_tf  = len(matrix_bin)
    cols  = matrix_bin.columns.values

    log(f"[verify] n_tss = {n_tss}")
    log(f"[verify] n_tf  = {n_tf}")
    if not (19_000 <= n_tss <= 20_500):
        log(f"[verify][WARN] n_tss outside expected 19000-20500 range")
    if not (1_000 <= n_tf <= 1_800):
        log(f"[verify][WARN] n_tf outside expected 1000-1800 range")

    # CTCF: argmax should be near TSS
    if "CTCF" in matrix_bin.index:
        argmax_bp = int(cols[np.argmax(matrix_bin.loc["CTCF"].values)])
        log(f"[verify] CTCF binary argmax = {argmax_bp:+d} bp (expect within ±200)")
        if abs(argmax_bp) > 200:
            log(f"[verify][WARN] CTCF peak unexpectedly far from TSS — strand-flip suspect")

    # TBP: should peak ~ -30 (TATA box upstream)
    if "TBP" in matrix_bin.index:
        argmax_bp = int(cols[np.argmax(matrix_bin.loc["TBP"].values)])
        log(f"[verify] TBP  binary argmax = {argmax_bp:+d} bp (expect near -30)")
        if argmax_bp >= 0:
            log(f"[verify][WARN] TBP argmax not upstream — strand-flip suspect")

    # Score-vs-binary dynamic range
    bmax = matrix_bin.max(axis=1).replace(0, np.nan)
    smax = matrix_scr.max(axis=1).replace(0, np.nan)
    ratio = (smax / bmax).dropna()
    log(f"[verify] score_max/binary_max ratio: median={ratio.median():.1f} "
        f"p10={ratio.quantile(0.1):.1f} p90={ratio.quantile(0.9):.1f}")

    # Aggregate centering
    agg_bin = matrix_bin.values.mean(axis=0)
    agg_argmax = int(cols[np.argmax(gaussian_filter1d(agg_bin, sigma=10))])
    log(f"[verify] aggregate binary argmax = {agg_argmax:+d} bp (expect within ±100)")


################################################################################
# Helpers ######################################################################
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
# Execution ####################################################################
################################################################################
def main():
    Path(OUT_DN, "matrices").mkdir(parents=True, exist_ok=True)
    Path(OUT_DN, "plots").mkdir(parents=True, exist_ok=True)
    Path(OUT_DN, "logs").mkdir(parents=True, exist_ok=True)

    log_path = Path(OUT_DN) / "logs" / f"run.{dt.datetime.now():%Y%m%d_%H%M%S}.log"
    log, log_fh = _make_logger(str(log_path))

    log(f"WORKERS = {WORKERS}")
    log(f"window  = +/- {HALF} bp (LEN = {LEN})")

    # 1) Canonical TSS table
    tss_df = load_canonical_tss(GTF_FN, VALID_CHROMS)
    tss_df.to_csv(Path(OUT_DN, "matrices", "tss_table.tsv"),
                  sep="\t", index=False)
    log(f"TSSs written: {len(tss_df):,}")

    # 2) Build TSS windows DataFrame (handed to workers via initializer)
    windows_df = pd.DataFrame({
        "Chromosome": tss_df["chrom"].values,
        "Start":      (tss_df["tss"].values - HALF).astype(np.int32),
        "End":        (tss_df["tss"].values + HALF + 1).astype(np.int32),
        "Strand":     tss_df["strand"].values,
        "tss_pos":    tss_df["tss"].values.astype(np.int32),
    })
    # Clamp Start >= 0 (TSSs near chromosome start)
    windows_df["Start"] = windows_df["Start"].clip(lower=0)

    # 3) Discover TFs
    tf_axis = tf_name_set()
    # tf_axis is the CANDIDATE set; the realised axis is the subset with peak
    # files, logged as "TF files matched" below.
    log(f"TF candidate set ({TF_SET}): {len(tf_axis)} names")
    tf_files = discover_tf_files(PER_TF_DN, tf_axis)
    log(f"TF files matched: {len(tf_files)}")

    # 4) Run per-TF accumulation
    t_start = time.time()
    if WORKERS == 1:
        _init_worker(windows_df, VALID_CHROMS)
        results = [accumulate_tf(args) for args in tf_files]
    else:
        ctx = mp.get_context("fork")  # Linux: fork shares the windows DF cheaply
        with ctx.Pool(processes=WORKERS,
                      initializer=_init_worker,
                      initargs=(windows_df, VALID_CHROMS)) as pool:
            results = []
            for i, r in enumerate(pool.imap_unordered(accumulate_tf, tf_files,
                                                      chunksize=4), 1):
                results.append(r)
                if i % 100 == 0 or i == len(tf_files):
                    log(f"  processed {i}/{len(tf_files)} TFs "
                        f"(last: {r['tf']} — {r['n_peaks_kept']:,} peaks, "
                        f"{r['n_overlaps']:,} overlaps, {r['runtime_s']:.2f}s)")
                    if r.get("error"):
                        log(f"    [ERROR] {r['tf']}: {r['error']}")
    log(f"per-TF scan done in {time.time() - t_start:.1f}s")

    # Stable order: by TF name
    results.sort(key=lambda r: r["tf"])

    # 5) Write matrices + summary
    binary_df, score_df, raw_df, raw1000_df = write_matrices(
        results, n_tss=len(tss_df), out_dn=OUT_DN)
    log(f"matrices written: shape = {binary_df.shape}")

    # 6) Plots
    plots_dn = Path(OUT_DN) / "plots"
    log("plotting...")

    plot_heatmap(binary_df, "Mean per-bp coverage probability",
                 cmap="viridis", out_path_stem=str(plots_dn / "heatmap.binary"))
    plot_heatmap(score_df,  "Mean per-bp summed score",
                 cmap="magma",   out_path_stem=str(plots_dn / "heatmap.score"))

    plot_metaplot_per_tf(binary_df, "Mean per-bp coverage probability",
                         base_color="C0",
                         out_path_stem=str(plots_dn / "metaplot_per_tf.binary"))
    plot_metaplot_per_tf(score_df,  "Mean per-bp summed score",
                         base_color="C3",
                         out_path_stem=str(plots_dn / "metaplot_per_tf.score"))

    plot_metaplot_aggregate(binary_df, "per-bp coverage probability",
                            line_color="C0",
                            out_path_stem=str(plots_dn / "metaplot_aggregate.binary"))
    plot_metaplot_aggregate(score_df,  "per-bp summed score",
                            line_color="C3",
                            out_path_stem=str(plots_dn / "metaplot_aggregate.score"))

    plot_overlap_histogram_5nt(raw_df,
                               out_path_stem=str(plots_dn / "overlap_histogram_5nt"),
                               bin_width=5, color="steelblue",
                               title_suffix="all peaks")
    plot_overlap_histogram_5nt(raw1000_df,
                               out_path_stem=str(plots_dn / "overlap_histogram_5nt.score1000"),
                               bin_width=5, color="darkorange",
                               title_suffix="score==1000 peaks only")
    log("plots written")

    # 7) Verification
    verify(binary_df, score_df, tss_df, log)

    log("DONE")
    log_fh.close()


if __name__ == "__main__":
    main()
