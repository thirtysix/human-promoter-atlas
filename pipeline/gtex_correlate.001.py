#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
TF × target-transcript expression correlation across GTEx tissues, plus
per-(module, tissue) activity scores.

Inputs:
    tss_gtex/transcript_tissue_mean.parquet   (n_tx × n_tissue)
    tss_modules/tf_index.tsv                  (atlas TFs)
    matrices/tss_table.tsv                    (atlas tx -> gene_name)
    tss_modules/peaks.parquet                 (per-(tx, tf) module assignments)
    tss_modules/modules.tsv
    tss_modules/nmf.k10.module_program.tsv

Strategy:
    Each TF is a *gene*; a single gene has multiple transcripts. For
    correlation we collapse TF transcripts to a single "TF expression
    profile" per tissue by taking the **max** TPM across the gene's
    transcripts (proxies for the TF gene's expression in each tissue).

Outputs:
    tss_gtex/tf_target_correlation.parquet
        long: tf, target_transcript, r (rounded float16), |r| filter applied
    tss_gtex/tf_tissue_expression.parquet
        wide: tf × tissue (max-of-transcripts mean TPM, rounded 2 dec)
    tss_gtex/module_tissue_activity.parquet
        long: module_id, tissue, mean_expr_of_assigned_tfs (top-tier biology
        — which tissues each module's TFs are most expressed in)
"""

################################################################################
# Libraries ####################################################################
################################################################################
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN, K_CANONICAL

sys.stdout.reconfigure(line_buffering=True)


################################################################################
# Paths ########################################################################
################################################################################
ROOT                = OUT_DN   # config.OUT_DN
GTEX_DN = ROOT / "tss_gtex"

TX_MEAN_FN  = GTEX_DN / "transcript_tissue_mean.parquet"
TSS_FN      = ROOT / "tss_modules" / "tss_table.tsv"
TF_INDEX_FN = ROOT / "tss_modules" / "tf_index.tsv"
MODULES_FN  = ROOT / "tss_modules" / "modules.tsv"
MP_K10_FN   = ROOT / "tss_modules" / f"nmf.k{K_CANONICAL}.module_program.tsv"
PEAKS_FN    = ROOT / "tss_modules" / "peaks.parquet"

DECIMALS    = 2
MIN_ABS_R   = 0.30   # only keep correlations with |r| above this threshold
MIN_TISSUES = 5      # need at least this many non-NaN tissue means to compute r


################################################################################
# Helpers ######################################################################
################################################################################
def _ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}")


def pearson_rows(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Pearson correlation between every row of M (n_rows × n_cols) and the
    1-D vector V (n_cols,). Returns an (n_rows,) array."""
    M = M.astype(np.float64, copy=False)
    V = V.astype(np.float64, copy=False)
    M -= M.mean(axis=1, keepdims=True)
    V = V - V.mean()
    num = (M * V).sum(axis=1)
    den = np.sqrt((M * M).sum(axis=1) * (V * V).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


################################################################################
# Execution ####################################################################
################################################################################
def main():
    GTEX_DN.mkdir(parents=True, exist_ok=True)

    # 1) Load inputs
    _log(f"loading {TX_MEAN_FN.name}")
    tx_mean = pd.read_parquet(TX_MEAN_FN)
    tx_mean = tx_mean.astype(np.float32)
    _log(f"  shape = {tx_mean.shape}")
    n_tx, n_tissue = tx_mean.shape

    _log(f"loading TSS table for tx -> gene_name")
    tss = pd.read_csv(TSS_FN, sep="\t",
                       usecols=["transcript_id", "gene_name"])
    tx_to_gene = dict(zip(tss["transcript_id"], tss["gene_name"].fillna("")))

    _log(f"loading TF index")
    tf_idx = pd.read_csv(TF_INDEX_FN, sep="\t")
    tf_set = set(tf_idx["TF"].astype(str))
    _log(f"  {len(tf_set):,} TFs in atlas")

    # 2) TF tissue expression (per gene): max TPM across the gene's transcripts
    _log("collapsing TF transcripts -> TF gene tissue expression")
    keep_rows = tx_mean.index.map(lambda x: tx_to_gene.get(x, "") in tf_set)
    tf_tx = tx_mean[keep_rows]
    tf_genes = pd.Series(tf_tx.index.map(lambda x: tx_to_gene.get(x, "")),
                          index=tf_tx.index)
    tf_expr = (tf_tx.assign(_tf=tf_genes.values)
                    .groupby("_tf")
                    .max(numeric_only=True))
    tf_expr.index.name = "tf"
    tf_expr = tf_expr.astype(np.float32).round(DECIMALS)
    tf_expr.to_parquet(GTEX_DN / "tf_tissue_expression.parquet")
    _log(f"  tf_tissue_expression.parquet shape={tf_expr.shape} "
         f"({(GTEX_DN / 'tf_tissue_expression.parquet').stat().st_size/1e6:.1f} MB)")

    # 3) TF × target-transcript correlation across the 66 tissues
    _log("computing TF × target-transcript Pearson correlations…")
    target_M = tx_mean.to_numpy(np.float32)
    target_ix = tx_mean.index.values

    long_chunks = []
    t0 = time.time()
    n_kept = 0
    for tf, vec in tf_expr.iterrows():
        v = vec.to_numpy(np.float32)
        if np.isnan(v).sum() > n_tissue - MIN_TISSUES:
            continue
        # Replace NaN with column-mean for both M and V (cheap robustness)
        # Already mean-removed in pearson_rows; we just need finite values.
        finite_v = np.where(np.isnan(v), np.nanmean(v), v)
        # Fill NaNs in M with row-means (per-target)
        M = target_M.copy()
        row_means = np.nanmean(M, axis=1, keepdims=True)
        M = np.where(np.isnan(M), row_means, M)
        r = pearson_rows(M, finite_v)
        keep = np.abs(r) >= MIN_ABS_R
        if keep.sum() == 0:
            continue
        ts = target_ix[keep]
        rs = r[keep].astype(np.float16).round(2)
        long_chunks.append(pd.DataFrame({
            "tf": tf, "target_transcript": ts, "r": rs,
        }))
        n_kept += int(keep.sum())

    out_corr = pd.concat(long_chunks, ignore_index=True)
    out_corr["r"] = out_corr["r"].astype(np.float16)
    out_corr.to_parquet(GTEX_DN / "tf_target_correlation.parquet", index=False)
    _log(f"  TF × target correlations: {n_kept:,} pairs at |r|>={MIN_ABS_R}  "
         f"({time.time()-t0:.1f}s); "
         f"{(GTEX_DN / 'tf_target_correlation.parquet').stat().st_size/1e6:.1f} MB")

    # 4) Module-tissue activity = mean expression of the module's assigned TFs
    _log("computing module × tissue activity scores…")
    modules = pd.read_csv(MODULES_FN, sep="\t",
                           usecols=["module_id", "transcript_id", "tss_id"])
    # Read the (module, TF) assignments by re-deriving from peaks +
    # module bounds (modules.tsv doesn't store TFs directly; fastest is to
    # rebuild the mapping from peaks within each module's [lo, hi]).
    modules_full = pd.read_csv(MODULES_FN, sep="\t",
                                usecols=["module_id", "tss_id", "lo_offset",
                                         "hi_offset", "n_tfs_assigned"])

    _log("  loading peaks parquet…")
    peaks = pd.read_parquet(PEAKS_FN)
    peaks["tf"] = peaks["tf_idx"].map(
        dict(zip(tf_idx["tf_idx"], tf_idx["TF"])))

    # For each module, find peaks at its tss_id within [lo, hi] with score>=500
    _log("  joining peaks to modules (this is the slow step)…")
    peaks500 = peaks[peaks["score"] >= 500]
    # Index peaks by tss_id for quick subset
    peaks_by_tss = peaks500.groupby("tss_id")

    rows = []
    t0 = time.time()
    for i, m in modules_full.iterrows():
        try:
            sub = peaks_by_tss.get_group(int(m["tss_id"]))
        except KeyError:
            continue
        in_mod = sub[(sub["local"] >= int(m["lo_offset"])) &
                      (sub["local"] <= int(m["hi_offset"]))]
        tfs = in_mod["tf"].dropna().unique()
        if len(tfs) == 0:
            continue
        # Mean expression per tissue across this module's TFs
        present = [t for t in tfs if t in tf_expr.index]
        if not present:
            continue
        block = tf_expr.loc[present].to_numpy(np.float32)
        block_mean = np.nanmean(block, axis=0).round(DECIMALS).astype(np.float32)
        rows.extend({"module_id": int(m["module_id"]),
                     "tissue": tissue,
                     "mean_tf_tpm": float(v),
                     "n_tfs_in_module": int(len(present))}
                    for tissue, v in zip(tf_expr.columns, block_mean))
        if (i + 1) % 5000 == 0:
            _log(f"    processed {i+1:,}/{len(modules_full):,} modules "
                 f"({time.time()-t0:.1f}s)")

    out_act = pd.DataFrame(rows)
    out_act["mean_tf_tpm"] = out_act["mean_tf_tpm"].astype(np.float32)
    out_act["n_tfs_in_module"] = out_act["n_tfs_in_module"].astype(np.int32)
    out_act.to_parquet(GTEX_DN / "module_tissue_activity.parquet", index=False)
    _log(f"  module_tissue_activity.parquet: {len(out_act):,} rows  "
         f"({(GTEX_DN / 'module_tissue_activity.parquet').stat().st_size/1e6:.1f} MB)")

    _log("DONE")


if __name__ == "__main__":
    main()
