#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
DepMap CRISPR essentiality summary, restricted to atlas genes.

Reads the DepMap gene-effect (Chronos) matrix + cell-line metadata, filters
to atlas genes (the union of canonical TSS gene_names + the 1,304 atlas TFs),
and aggregates per (gene, OncotreeLineage) and per gene globally.

Inputs:
    DepMap/CRISPRGeneEffect.csv   (1187 cell lines × 18,435 genes; Chronos)
    DepMap/Model.csv              (cell line metadata; OncotreeLineage)
    tss_modules/tss_table.tsv     (atlas gene_names)
    tss_modules/tf_index.tsv      (atlas TFs)

Outputs (analyses/canonical_promoter/tss_depmap/):
    gene_lineage_essentiality.parquet
        long: gene, lineage, n_lines, mean_chronos, median_chronos,
              frac_essential (Chronos < -1).
    gene_essentiality_summary.parquet
        per gene: n_lines, median_chronos_all, frac_essential_all,
                  most_essential_lineage, most_essential_lineage_chronos.
    lineage_index.tsv
        OncotreeLineage list with cell-line counts.

Convention: Chronos < 0 means the gene is essential / a dependency in that
cell line (knockout reduces fitness). The threshold for "essential" is the
DepMap convention of -1.0 (corresponding to ~50% chance of being essential
under the Chronos prior).
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
from config import DEPMAP_DN, OUT_DN

sys.stdout.reconfigure(line_buffering=True)


################################################################################
# Paths ########################################################################
################################################################################
ROOT                = OUT_DN   # config.OUT_DN
GE_FN      = DEPMAP_DN / "CRISPRGeneEffect.csv"
MODEL_FN   = DEPMAP_DN / "Model.csv"
TSS_FN     = ROOT / "tss_modules" / "tss_table.tsv"
TF_FN      = ROOT / "tss_modules" / "tf_index.tsv"
OUT_DN     = ROOT / "tss_depmap"

DECIMALS              = 3
ESSENTIAL_THRESHOLD   = -1.0     # DepMap convention
MIN_LINES_PER_LINEAGE = 5        # drop lineages with too few lines


################################################################################
# Helpers ######################################################################
################################################################################
def _ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}")


def strip_entrez(col: str) -> str:
    """'A1BG (1)' -> 'A1BG'."""
    return col.split(" ")[0]


################################################################################
# Execution ####################################################################
################################################################################
def main():
    OUT_DN.mkdir(parents=True, exist_ok=True)

    # 1) Atlas gene set (TSS gene_names ∪ atlas TFs)
    _log(f"reading atlas gene names")
    tss = pd.read_csv(TSS_FN, sep="\t", usecols=["gene_name"])
    tf_idx = pd.read_csv(TF_FN, sep="\t")
    atlas_genes = set(tss["gene_name"].dropna().astype(str)) | \
                  set(tf_idx["TF"].astype(str))
    _log(f"  {len(atlas_genes):,} atlas genes (TSSs ∪ TFs)")

    # 2) Cell-line metadata
    _log(f"reading {MODEL_FN.name}")
    model = pd.read_csv(MODEL_FN, usecols=["ModelID", "OncotreeLineage"],
                         dtype="string")
    model = model.dropna(subset=["OncotreeLineage"])
    mid_to_lineage = dict(zip(model["ModelID"], model["OncotreeLineage"]))
    _log(f"  {len(model):,} cell lines with lineage; "
         f"{model['OncotreeLineage'].nunique()} unique lineages")

    # 3) Stream CRISPRGeneEffect.csv. The matrix is 1187 × 18435 float = ~88 MB
    #    as float32, so just load it whole. Then filter columns to atlas genes
    #    and rows to cell lines we have lineage for.
    _log(f"reading {GE_FN.name}")
    t0 = time.time()
    ge = pd.read_csv(GE_FN, index_col=0)
    _log(f"  shape = {ge.shape}  ({time.time()-t0:.1f}s)")

    # Map gene names: 'A1BG (1)' -> 'A1BG'
    ge.columns = [strip_entrez(c) for c in ge.columns]
    # Restrict to atlas genes that actually exist in DepMap
    keep_cols = [g for g in ge.columns if g in atlas_genes]
    ge = ge[keep_cols].astype(np.float32)
    _log(f"  filtered to {len(keep_cols):,} atlas genes "
         f"present in DepMap (of {len(atlas_genes):,})")

    # Add lineage as index column
    ge = ge.loc[ge.index.intersection(model["ModelID"])]
    ge["_lineage"] = ge.index.map(mid_to_lineage)
    n_lineage = ge["_lineage"].nunique()
    n_lines   = len(ge)
    _log(f"  {n_lines:,} cell lines × {len(keep_cols):,} genes; "
         f"{n_lineage} lineages")

    # 4) Per (gene, lineage) aggregation
    _log("aggregating per (gene, lineage) — melt + groupby")
    t0 = time.time()
    melted = ge.melt(id_vars=["_lineage"], var_name="gene",
                      value_name="chronos")
    _log(f"  melted shape = {len(melted):,} rows ({time.time()-t0:.1f}s)")
    grp = melted.groupby(["gene", "_lineage"], dropna=False)
    agg = grp["chronos"].agg(
        n_lines        = "size",
        mean_chronos   = "mean",
        median_chronos = "median",
        frac_essential = lambda s: float((s < ESSENTIAL_THRESHOLD).mean()),
    ).reset_index().rename(columns={"_lineage": "lineage"})
    _log(f"  per-(gene, lineage) rows: {len(agg):,} "
         f"({time.time()-t0:.1f}s)")

    # Round + types
    agg["mean_chronos"]   = agg["mean_chronos"].astype(np.float32).round(DECIMALS)
    agg["median_chronos"] = agg["median_chronos"].astype(np.float32).round(DECIMALS)
    agg["frac_essential"] = agg["frac_essential"].astype(np.float32).round(DECIMALS)
    agg["n_lines"]        = agg["n_lines"].astype(np.int32)

    # Drop low-N lineages
    before = len(agg)
    keep_lineages = (agg.groupby("lineage")["n_lines"].sum()
                       .loc[lambda s: s >= MIN_LINES_PER_LINEAGE].index)
    agg = agg[agg["lineage"].isin(keep_lineages)]
    _log(f"  dropped {before - len(agg):,} rows in low-N lineages "
         f"(<{MIN_LINES_PER_LINEAGE} lines)")
    agg.to_parquet(OUT_DN / "gene_lineage_essentiality.parquet", index=False)
    _log(f"  -> gene_lineage_essentiality.parquet "
         f"({(OUT_DN / 'gene_lineage_essentiality.parquet').stat().st_size/1e6:.1f} MB)")

    # 5) Per-gene global summary
    _log("computing per-gene global summary")
    chronos_only = ge.drop(columns=["_lineage"])
    gene_summary = pd.DataFrame({
        "gene":            chronos_only.columns,
        "n_lines":         chronos_only.notna().sum(axis=0).values,
        "median_chronos":  chronos_only.median(axis=0).values,
        "frac_essential":  (chronos_only < ESSENTIAL_THRESHOLD).mean(axis=0).values,
    })
    # Most-essential lineage per gene
    me_idx = (agg.sort_values("median_chronos")
                  .groupby("gene")
                  .head(1)
                  .set_index("gene")[["lineage", "median_chronos"]]
                  .rename(columns={"lineage":         "most_essential_lineage",
                                   "median_chronos": "most_essential_chronos"}))
    gene_summary = gene_summary.merge(me_idx, left_on="gene",
                                        right_index=True, how="left")
    gene_summary["median_chronos"]  = gene_summary["median_chronos"].astype(np.float32).round(DECIMALS)
    gene_summary["frac_essential"]  = gene_summary["frac_essential"].astype(np.float32).round(DECIMALS)
    gene_summary["most_essential_chronos"] = gene_summary["most_essential_chronos"].astype(np.float32).round(DECIMALS)
    gene_summary["n_lines"]         = gene_summary["n_lines"].astype(np.int32)
    gene_summary.to_parquet(OUT_DN / "gene_essentiality_summary.parquet",
                              index=False)
    _log(f"  -> gene_essentiality_summary.parquet "
         f"({(OUT_DN / 'gene_essentiality_summary.parquet').stat().st_size/1e6:.1f} MB)")

    # 6) Lineage index
    lin = (ge["_lineage"].value_counts()
              .reset_index()
              .rename(columns={"_lineage": "lineage", "count": "n_cell_lines"}))
    lin.to_csv(OUT_DN / "lineage_index.tsv", sep="\t", index=False)
    _log(f"  -> lineage_index.tsv ({len(lin)} lineages)")

    # 7) Quick sanity readout
    _log("top 5 most essential atlas genes (by median Chronos):")
    for _, r in gene_summary.sort_values("median_chronos").head(5).iterrows():
        _log(f"    {r['gene']:8s}  median={r['median_chronos']:+.3f}  "
             f"frac_essential={r['frac_essential']:.2f}  "
             f"top_lineage={r['most_essential_lineage']}")

    _log("DONE")


if __name__ == "__main__":
    main()
