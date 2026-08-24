#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Exon structure for protein-coding transcripts, near canonical TSSs.

The per-transcript view plots TF binding on a bp axis with nothing to read it
against: a peak at +900 could be in the first exon, the first intron, or inside
a neighbouring gene, and the plot cannot say which. This extracts the exon
models so the view can.

SCOPE: only what a +/-OUTER_HALF window can show. Genome-wide protein-coding
exons are ~1.2 M rows; restricted to windows around the 19,745 canonical TSSs
it is a small fraction, and everything outside is unplottable anyway.

NEIGHBOURING GENES ARE KEPT ON PURPOSE. A promoter often sits inside or beside
another gene, and that is exactly what a reader needs to see -- dropping
everything except the focal transcript would hide the most interesting cases
(bidirectional promoters, nested genes, readthrough).

Coordinates are stored ABSOLUTE. The window offset depends on the TSS and the
strand, and baking it in would mean re-extracting whenever the window changes;
the app converts at query time, where it already knows both.

Usage:
    python data/build_gene_structure.py
"""

################################################################################
# Libraries ####################################################################
################################################################################
import gzip
import os
import re
import sys
from pathlib import Path

import duckdb
import pandas as pd

APP_DN = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(APP_DN / "data"))

DATA_DN = APP_DN / "data"
DUCKDB_FN = DATA_DN / "canonical_promoter.duckdb"
OUT_FN = DATA_DN / "gene_structure.parquet"

# Half-width to extract around each canonical TSS. Deliberately wider than the
# viewer's OUTER_HALF (1500) so a later widening of the window does not require
# re-running this, and so a feature starting just outside is still drawn
# entering the frame rather than clipped to nothing.
EXTRACT_HALF = 5000

FEATURES = ("exon", "CDS", "five_prime_utr", "three_prime_utr")
_ATTR = re.compile(r'(\w+) "([^"]*)"')


def _log(m):
    print(f"[gene_structure] {m}", flush=True)


def main() -> int:
    gtf = os.environ.get("HPA_GTF")
    if not gtf or not Path(gtf).exists():
        raise SystemExit(f"HPA_GTF not found: {gtf}")
    if not DUCKDB_FN.exists():
        raise SystemExit(f"{DUCKDB_FN} not found; run data/build_app_db.py first")

    con = duckdb.connect(str(DUCKDB_FN), read_only=True)
    tss = con.execute("SELECT chrom, tss FROM tss").df()
    con.close()
    # per chromosome, the sorted TSS positions -- used to keep only features
    # near one, without a row-by-row interval join over the whole GTF
    import numpy as np
    by_chrom = {str(c): np.sort(g.tss.to_numpy())
                for c, g in tss.groupby("chrom")}
    _log(f"{len(tss):,} canonical TSSs across {len(by_chrom)} chromosomes")

    rows = []
    kept = seen = 0
    with gzip.open(gtf, "rt") as fh:
        for line in fh:
            if line[0] == "#":
                continue
            f = line.rstrip("\n").split("\t")
            if len(f) < 9 or f[2] not in FEATURES:
                continue
            seen += 1
            arr = by_chrom.get(f[0])
            if arr is None:
                continue
            start, end = int(f[3]), int(f[4])
            # near any canonical TSS?
            i = np.searchsorted(arr, start)
            near = False
            for j in (i - 1, i):
                if 0 <= j < len(arr) and (start - EXTRACT_HALF <= arr[j]
                                          <= end + EXTRACT_HALF):
                    near = True
                    break
            if not near:
                continue
            a = dict(_ATTR.findall(f[8]))
            if a.get("gene_biotype") != "protein_coding":
                continue
            # gene_id, because 1.3% of features have no gene_name at all --
            # Ensembl novel genes like ENSG00000285171, which overlaps IL2RG's
            # promoter. Labelling those "(unnamed)" told a reader nothing and
            # looked like a bug; the accession is at least lookupable.
            # transcript_biotype, because gene_biotype protein_coding admits
            # nonsense_mediated_decay transcripts, and drawing those as if
            # they were coding overstates them.
            rows.append((f[0], start, end, f[6], f[2],
                         a.get("gene_name", ""), a.get("gene_id", ""),
                         a.get("transcript_id", ""),
                         a.get("transcript_biotype", ""),
                         int(a.get("exon_number", 0) or 0)))
            kept += 1
            if kept % 200000 == 0:
                _log(f"  kept {kept:,} of {seen:,} scanned")

    df = pd.DataFrame(rows, columns=["chrom", "start", "end", "strand",
                                     "feature", "gene_name", "gene_id",
                                     "transcript_id", "transcript_biotype",
                                     "exon_number"])
    # one display label per gene, never blank
    df["label"] = df.gene_name.where(df.gene_name.astype(bool), df.gene_id)
    df = df.sort_values(["chrom", "start"]).reset_index(drop=True)
    df.to_parquet(OUT_FN, index=False, compression="zstd")
    _log(f"{len(df):,} features from {seen:,} scanned "
         f"({df.gene_name.nunique():,} genes, "
         f"{df.transcript_id.nunique():,} transcripts)")
    _log(f"  by feature: "
         + ", ".join(f"{k} {v:,}" for k, v in df.feature.value_counts().items()))
    _log(f"wrote {OUT_FN} ({OUT_FN.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
