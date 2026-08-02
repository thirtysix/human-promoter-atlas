#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Extract per-(transcript, tissue) GTEx V11 expression statistics, restricted to
the canonical TSSs in the Human Promoter Atlas.

Inputs:
    GTEx V11 transcripts TPM    (~4.9 GB gz)
    GTEx V11 sample attributes  (SAMPID -> SMTSD)
    Atlas tss_table.tsv         (canonical transcript_ids — without version)

Outputs (analyses/canonical_promoter/tss_gtex/):
    transcript_tissue_stats.parquet
        long-format: transcript_id, tissue, n_samples, mean, median, q1, q3, std
        rounded to 2 decimals; float32. ~5 MB.
    transcript_tissue_mean.parquet
        wide: transcript_id × tissue (mean TPM, float32, 2 dec). For
        downstream correlation analysis.
    tissue_index.tsv
        list of tissues + sample counts.

Strategy: stream the TPM file in chunks of CHUNK_ROWS rows; for each chunk,
keep only rows whose transcript_id (without version) is in the atlas; melt
into long-format and aggregate by tissue using `groupby` + `describe`-style
percentile computation. Concatenate chunk results into the final parquet.

Memory ceiling: ~3 GB peak (one chunk's dense float32 frame).
"""

################################################################################
# Libraries ####################################################################
################################################################################
import sys
import time
import datetime as dt
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

# Machine-specific paths and build axes -> pipeline/config.py
from config import GTEX_DN, OUT_DN

# Force line-buffered stdout so logs flush even when redirected.
sys.stdout.reconfigure(line_buffering=True)


################################################################################
# Paths ########################################################################
################################################################################
ROOT                = OUT_DN   # config.OUT_DN
TPM_FN     = GTEX_DN / "GTEx_Analysis_2025-08-22_v11_RSEMv1.3.3_transcripts_tpm.txt.gz"
SAMP_FN    = GTEX_DN / "GTEx_Analysis_v11_Annotations_SampleAttributesDS.txt"
TSS_FN     = ROOT / "tss_modules" / "tss_table.tsv"
OUT_DN     = ROOT / "tss_gtex"

CHUNK_ROWS = 2000          # transcripts per streaming chunk
DECIMALS   = 2             # output decimals
MIN_SAMPLES_PER_TISSUE = 5 # drop tissues with too few replicates


################################################################################
# Helpers ######################################################################
################################################################################
def _ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}")


def strip_version(s: pd.Series) -> pd.Series:
    """ENST00000123.4 -> ENST00000123."""
    return s.astype("string").str.split(".").str[0]


################################################################################
# Execution ####################################################################
################################################################################
def main():
    OUT_DN.mkdir(parents=True, exist_ok=True)

    # 1) Sample attributes: SAMPID -> SMTSD (granular tissue)
    _log(f"reading sample attributes: {SAMP_FN.name}")
    samp = pd.read_csv(SAMP_FN, sep="\t", usecols=["SAMPID", "SMTS", "SMTSD"],
                        dtype="string")
    samp = samp.dropna(subset=["SMTSD"])
    sample_to_tissue = dict(zip(samp["SAMPID"], samp["SMTSD"]))
    _log(f"  {len(samp):,} samples annotated; "
         f"{samp['SMTSD'].nunique()} unique SMTSD tissues")

    # 2) Atlas transcript IDs
    _log(f"reading atlas TSS table: {TSS_FN.name}")
    tss = pd.read_csv(TSS_FN, sep="\t", usecols=["transcript_id"])
    atlas_ids = set(tss["transcript_id"].astype("string"))
    _log(f"  {len(atlas_ids):,} canonical atlas transcripts")

    # 3) Stream the TPM file. Aggregate per-tissue directly (no melt).
    _log(f"streaming TPM file: {TPM_FN.name}")
    t0 = time.time()
    chunk_iter = pd.read_csv(
        TPM_FN, sep="\t", chunksize=CHUNK_ROWS,
        compression="gzip",
        dtype={"transcript_id": "string", "gene_id": "string"},
    )
    long_chunks = []
    n_kept_total = 0
    n_seen_total = 0
    tissue_to_cols: dict[str, list[str]] | None = None

    for ci, chunk in enumerate(chunk_iter, 1):
        n_seen_total += len(chunk)
        chunk["tx_strip"] = strip_version(chunk["transcript_id"])
        keep = chunk["tx_strip"].isin(atlas_ids)
        sub = chunk[keep]
        if ci == 1 or tissue_to_cols is None:
            # On first chunk, learn which sample columns belong to each tissue
            sample_cols = [c for c in chunk.columns
                            if c not in ("transcript_id", "gene_id", "tx_strip")]
            tissue_to_cols = {}
            for s in sample_cols:
                t = sample_to_tissue.get(s)
                if t is not None:
                    tissue_to_cols.setdefault(t, []).append(s)
            _log(f"  resolved {sum(len(v) for v in tissue_to_cols.values()):,} "
                 f"samples across {len(tissue_to_cols)} tissues")
        if sub.empty:
            if ci % 10 == 0:
                _log(f"  chunk {ci}: read {n_seen_total:,}, kept "
                     f"{n_kept_total:,}  ({time.time()-t0:.1f}s)")
            continue
        n_kept_total += len(sub)

        # Per-tissue aggregation directly from the wide chunk.
        # Result: one row per (transcript, tissue) for this chunk.
        tx_ids = sub["tx_strip"].values
        for tissue, cols in tissue_to_cols.items():
            block = sub[cols].astype(np.float32).to_numpy()
            n = block.shape[1]
            agg = pd.DataFrame({
                "transcript_id": tx_ids,
                "tissue":        tissue,
                "n_samples":     np.int32(n),
                "mean":          np.nanmean(block, axis=1).astype(np.float32),
                "median":        np.nanmedian(block, axis=1).astype(np.float32),
                "q1":            np.nanquantile(block, 0.25, axis=1).astype(np.float32),
                "q3":            np.nanquantile(block, 0.75, axis=1).astype(np.float32),
                "std":           np.nanstd(block, axis=1, ddof=1).astype(np.float32),
            })
            long_chunks.append(agg)

        if ci % 5 == 0:
            _log(f"  chunk {ci}: read {n_seen_total:,}, kept "
                 f"{n_kept_total:,}  ({time.time()-t0:.1f}s)")

    _log(f"streaming done: {n_seen_total:,} rows scanned, "
         f"{n_kept_total:,} matched atlas IDs in {time.time()-t0:.1f}s")

    if not long_chunks:
        _log("ERROR: no atlas transcripts found in GTEx TPM. Aborting.")
        return

    # 4) Concat and round
    out = pd.concat(long_chunks, ignore_index=True)
    out = out.rename(columns={"tx_strip": "transcript_id"})
    for c in ("mean", "median", "q1", "q3", "std"):
        out[c] = out[c].astype(np.float32).round(DECIMALS)
    out["n_samples"] = out["n_samples"].astype(np.int32)
    _log(f"long-format frame: {len(out):,} (transcript, tissue) rows; "
         f"{out['transcript_id'].nunique():,} transcripts × "
         f"{out['tissue'].nunique()} tissues")

    # Drop tissues with too few samples (mostly singletons)
    tissue_n = (out.groupby("tissue")["n_samples"].mean()
                  .sort_values(ascending=False))
    keep_tissues = set(tissue_n[tissue_n >= MIN_SAMPLES_PER_TISSUE].index)
    before = len(out)
    out = out[out["tissue"].isin(keep_tissues)]
    _log(f"  dropped {before - len(out):,} rows in tissues with <"
         f"{MIN_SAMPLES_PER_TISSUE} samples; kept {len(out):,}")

    out.to_parquet(OUT_DN / "transcript_tissue_stats.parquet", index=False)
    _log(f"  wrote transcript_tissue_stats.parquet "
         f"({(OUT_DN / 'transcript_tissue_stats.parquet').stat().st_size/1e6:.1f} MB)")

    # 5) Wide tissue-mean matrix (transcript × tissue)
    wide = out.pivot_table(index="transcript_id", columns="tissue",
                            values="mean", aggfunc="first")
    wide = wide.astype(np.float32).round(DECIMALS)
    wide.to_parquet(OUT_DN / "transcript_tissue_mean.parquet")
    _log(f"  wrote transcript_tissue_mean.parquet shape={wide.shape} "
         f"({(OUT_DN / 'transcript_tissue_mean.parquet').stat().st_size/1e6:.1f} MB)")

    # 6) Tissue index
    tissue_index = (out.groupby("tissue")
                       .agg(n_transcripts=("transcript_id", "nunique"),
                            mean_samples=("n_samples", "mean"))
                       .reset_index()
                       .sort_values("n_transcripts", ascending=False))
    tissue_index["mean_samples"] = tissue_index["mean_samples"].round(0).astype(int)
    tissue_index.to_csv(OUT_DN / "tissue_index.tsv", sep="\t", index=False)
    _log(f"  wrote tissue_index.tsv ({len(tissue_index)} tissues)")

    _log("DONE")


if __name__ == "__main__":
    main()
