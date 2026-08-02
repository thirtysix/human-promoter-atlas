#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Per-k program summary tables for tss_modules NMF outputs.

For each k in KS, reads:
    tss_modules/nmf.k{K}.H.tsv.gz
    tss_modules/nmf.k{K}.module_program.tsv

and writes:
    tss_modules/nmf.k{K}.summary.tsv
        program, n_modules, median_center, median_width,
        mean_dom_weight, top_tfs, reading

The 'reading' column is auto-filled with the top-3 TFs joined; intended to be
hand-curated afterwards.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN

ROOT    = OUT_DN / "tss_modules"
KS      = [8, 12, 15, 20]
TOP_N   = 8


def main():
    for k in KS:
        H_fn  = ROOT / f"nmf.k{k}.H.tsv.gz"
        mp_fn = ROOT / f"nmf.k{k}.module_program.tsv"
        if not (H_fn.exists() and mp_fn.exists()):
            print(f"k={k}: missing inputs, skipping")
            continue

        H = pd.read_csv(H_fn, sep="\t", index_col=0)
        mp = pd.read_csv(mp_fn, sep="\t")

        rows = []
        for p in range(1, k + 1):
            sub = mp[mp["dominant_program"] == p]
            n_dom = len(sub)
            med_center = int(sub["center_offset"].median()) if n_dom else 0
            med_width  = int(sub["width"].median())         if n_dom else 0
            mean_w     = float(sub["dominant_weight"].mean()) if n_dom else 0.0

            row = H.loc[f"prog{p}"]
            top_idx = np.argsort(row.values)[::-1][:TOP_N]
            top_tfs = [row.index[i] for i in top_idx]
            reading = ", ".join(top_tfs[:3])
            rows.append({
                "program":         p,
                "n_modules":       n_dom,
                "median_center":   med_center,
                "median_width":    med_width,
                "mean_dom_weight": round(mean_w, 4),
                "top_tfs":         ",".join(top_tfs),
                "reading":         reading,
            })
        out = pd.DataFrame(rows).sort_values("n_modules", ascending=False)
        out.to_csv(ROOT / f"nmf.k{k}.summary.tsv", sep="\t", index=False)
        print(f"k={k}: wrote nmf.k{k}.summary.tsv ({len(out)} programs)")


if __name__ == "__main__":
    main()
