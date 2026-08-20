#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Attach gene identity to genome-wide elements, for the gene-centric front door.

genome_modules.py labels each element with its distance to the nearest
canonical TSS but not with WHICH TSS -- it sorts positions per chromosome
(``np.sort(g["tss"])``) and drops the gene columns before searching. So
elements carry dist_to_tss and stratum, and "show me the elements for SOX2"
cannot be answered. This adds the missing link.

Applied to an EXISTING elements.tsv rather than folded into a re-run, because
the factorization is aligned to the current element_id ordering: re-running
discovery to gain three columns would risk silently re-indexing the rows that
nmf.k*.W.npz and element_program.tsv already point at.

PROXIMITY IS NOT REGULATION
---------------------------
The nearest gene to a distal element is frequently not its target -- enhancers
skip over genes, act across hundreds of kb, and contact promoters through loops
that linear distance does not see. These columns support a lookup ("what lies
near this gene") and must not be presented as an assignment ("this element
regulates this gene"). For promoter-stratum elements (<=1 kb) the nearest-TSS
link is dependable; for proximal it is usually reasonable; for distal it is a
locator, nothing more. n_tss_within is reported alongside so an element sitting
in a gene-dense neighbourhood is visibly ambiguous rather than silently
assigned to whichever TSS happened to be one base closer.

Usage:
    python pipeline/genome_annotate_genes.py --genome-dir <dir>
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT_DN

AMBIGUITY_BP = 10_000        # fixed window, interpretable near promoters
AMBIGUITY_SCALE = 2.0        # ...and a scale-relative one that works at any
                             # distance. A fixed 10 kb window is VACUOUS for
                             # distal elements: "distal" is defined as >10 kb
                             # from any TSS, so the count is structurally 0
                             # for all 353,550 of them -- exactly the set
                             # whose gene link is weakest (median 72 kb).


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def annotate(el: pd.DataFrame, tss: pd.DataFrame) -> pd.DataFrame:
    """Add nearest-TSS gene identity, keeping the existing dist_to_tss intact."""
    cols = ["nearest_tss_id", "nearest_gene_name", "nearest_gene_id",
            "nearest_transcript_id"]
    out = {c: np.full(len(el), "", dtype=object) for c in cols}
    dist = np.full(len(el), 1 << 30, np.int64)
    n_near = np.zeros(len(el), np.int32)
    n_comp = np.zeros(len(el), np.int32)

    tss = tss.copy()
    tss["tss_id"] = tss.get("tss_id",
                            tss.gene_id.astype(str) + ":" + tss.tss.astype(str))
    for c, sub in el.groupby("chrom"):
        g = tss[tss.chrom.astype(str) == str(c)]
        if g.empty:
            continue
        g = g.sort_values("tss")
        arr = g.tss.to_numpy()
        p = sub["peak"].to_numpy()
        j = np.searchsorted(arr, p)
        lo = np.clip(j - 1, 0, len(arr) - 1)
        hi = np.clip(j, 0, len(arr) - 1)
        pick = np.where(np.abs(arr[lo] - p) <= np.abs(arr[hi] - p), lo, hi)
        idx = sub.index.to_numpy()
        dist[idx] = p - arr[pick]
        for c_out, c_src in zip(cols, ["tss_id", "gene_name", "gene_id",
                                       "transcript_id"]):
            out[c_out][idx] = g[c_src].to_numpy()[pick]
        # how many TSSs sit within AMBIGUITY_BP -- crowding, made visible
        n_near[idx] = (np.searchsorted(arr, p + AMBIGUITY_BP, "right")
                       - np.searchsorted(arr, p - AMBIGUITY_BP, "left"))
        # how many are within AMBIGUITY_SCALE x the nearest distance, i.e.
        # how many genes are COMPARABLY close. Meaningful at every scale.
        w = np.maximum(np.abs(arr[pick] - p) * AMBIGUITY_SCALE, 1).astype(np.int64)
        n_comp[idx] = (np.searchsorted(arr, p + w, "right")
                       - np.searchsorted(arr, p - w, "left"))

    el = el.copy()
    for c in cols:
        el[c] = out[c]
    el["n_tss_within_10kb"] = n_near
    el["n_tss_comparably_close"] = n_comp
    # The recomputed distance must reproduce what discovery recorded; if it does
    # not, the TSS table has changed under the build and every label is suspect.
    if "dist_to_tss" in el.columns:
        bad = int((el["dist_to_tss"].to_numpy() != dist).sum())
        if bad:
            raise SystemExit(
                f"{bad:,} of {len(el):,} elements disagree with the stored "
                f"dist_to_tss. The tss_table.tsv used here is not the one "
                f"discovery used, so these gene labels would be wrong.")
    else:
        el["dist_to_tss"] = dist
    return el


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--tss-table", default=None,
                    help="default: <OUT_DN>/tss_modules/tss_table.tsv")
    ap.add_argument("--out", default=None,
                    help="default: <genome-dir>/elements.genes.tsv")
    args = ap.parse_args()

    root = Path(args.genome_dir)
    tpath = Path(args.tss_table) if args.tss_table else (
        OUT_DN / "tss_modules" / "tss_table.tsv")
    el = pd.read_csv(root / "elements.tsv", sep="\t", dtype={"chrom": str})
    tss = pd.read_csv(tpath, sep="\t", dtype={"chrom": str})
    _log(f"{len(el):,} elements, {len(tss):,} canonical TSSs")

    el = annotate(el, tss)
    out = Path(args.out) if args.out else root / "elements.genes.tsv"
    el.to_csv(out, sep="\t", index=False)

    _log("=== nearest-gene link ===")
    for s in ("promoter", "proximal", "distal"):
        sub = el[el.stratum == s]
        if not len(sub):
            continue
        amb = float((sub.n_tss_within_10kb >= 2).mean() * 100)
        cmp_ = float((sub.n_tss_comparably_close >= 2).mean() * 100)
        print(f"  {s:>9}: {len(sub):>7,} elements, "
              f"{sub.nearest_gene_name.nunique():>6,} genes, "
              f"median |dist| {int(sub.dist_to_tss.abs().median()):>7,} bp, "
              f"{amb:5.1f}% >=2 TSS/10kb, "
              f"{cmp_:5.1f}% have a rival within 2x")
    print(f"\n  genes with >=1 element : {el.nearest_gene_name.nunique():,} "
          f"of {tss.gene_name.nunique():,}")
    print(f"  wrote {out}")
    print("\n  NOTE: nearest-gene is a LOCATOR, not a regulatory assignment.")
    print("  Dependable at the promoter stratum; a hint at best for distal.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
