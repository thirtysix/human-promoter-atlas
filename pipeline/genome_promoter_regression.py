#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Regression gate: does annotation-free discovery reproduce the promoter atlas?

genome_modules.py finds elements without reference to genes. If that pipeline
is correct, then wherever it and the gene-centric promoter atlas are looking at
the same DNA they must find the same things. This checks that, and it is a
PREREQUISITE for believing anything the genome-wide run says about distal
elements -- a distal program from a buggy pipeline is just a bug with a
biological-sounding name.

The comparison has one trap. The atlas was built at its own support floor of 2,
the genome run at the calibrated floor of 11 (see genome_support_null.py). Most
atlas modules therefore CANNOT be recovered, by construction, and a naive
overall recovery rate is dominated by them and means nothing. Recovery is
reported stratified by each atlas module's own n_tfs_assigned, and the gate is
read off the rows at or above the genome floor.

Coordinates. modules.tsv stores offsets relative to the TSS, so on the minus
strand they run opposite to genome coordinates and lo/hi swap. Peak positions
live on the 25 bp recentering grid (RECENTER_HALF=12), so a median offset near
12 bp is half a bin -- agreement to the limit of the representation, not a
discrepancy to explain.

Counts do not match one-to-one and should not. The atlas is indexed per TSS, so
one genomic locus appears once per nearby TSS; annotation-free discovery finds
it once. The collapse factor and how many collapsed elements span more than one
gene are reported, because that difference IS the composite-identity problem
the gene-centric schema has and this one does not.

Usage:
    python pipeline/genome_promoter_regression.py --genome-dir <dir>
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")
os.environ.setdefault("OMP_NUM_THREADS", "4")

import argparse
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

from config import OUT_DN, TIER, TF_SET, analysis_dir, write_analysis_readme

MATCH_BP    = 500       # peak-to-peak distance counted as the same element
FLOOR       = 11        # the calibrated genome-wide support floor
SUPPORT_BINS = [(0, 4), (5, 10), (11, 20), (21, 50), (51, 100), (101, 10**9)]


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _absolute(m):
    """TSS-relative offsets -> genome coordinates (minus strand runs backwards)."""
    plus = m.strand.isin(["+", 1, "1"])
    m = m.copy()
    m["abs_lo"] = np.where(plus, m.tss + m.lo_offset, m.tss - m.hi_offset)
    m["abs_hi"] = np.where(plus, m.tss + m.hi_offset, m.tss - m.lo_offset)
    m["abs_pk"] = np.where(plus, m.tss + m.center_offset, m.tss - m.center_offset)
    assert (m.abs_hi >= m.abs_lo).all(), "minus-strand bounds did not swap"
    return m


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True,
                    help="a genome_modules.py output directory")
    ap.add_argument("--modules", default=None,
                    help="default: <OUT_DN>/tss_modules/modules.tsv")
    ap.add_argument("--floor", type=int, default=FLOOR)
    ap.add_argument("--match-bp", type=int, default=MATCH_BP)
    ap.add_argument("--out", default=None)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    gdir = Path(args.genome_dir)
    mods = Path(args.modules) if args.modules else OUT_DN / "tss_modules" / "modules.tsv"
    out_dn = (Path(args.out) if args.out else
              analysis_dir("regression", sup=args.floor, match=args.match_bp))
    if out_dn.exists() and any(out_dn.iterdir()) and not args.force:
        raise SystemExit(f"{out_dn} exists and is not empty; pass --force")
    out_dn.mkdir(parents=True, exist_ok=True)

    g = pd.read_csv(gdir / "elements.tsv", sep="\t", dtype={"chrom": str})
    m = pd.read_csv(mods, sep="\t", dtype={"chrom": str})
    _log(f"atlas modules {len(m):,}   genome elements {len(g):,}")
    m = _absolute(m)
    m = m[m.chrom.isin(set(g.chrom))].reset_index(drop=True)
    _log(f"  atlas modules on the genome run's chromosomes: {len(m):,}")

    # nearest element peak, and whether any element overlaps the module span
    hit = np.zeros(len(m), bool)
    dist = np.full(len(m), 1 << 30, np.int64)
    eid = np.full(len(m), -1, np.int64)
    for c, sub in m.groupby("chrom"):
        e = g[g.chrom == c].sort_values("peak")
        pk, ids = e.peak.values, e.element_id.values
        j = np.clip(np.searchsorted(pk, sub.abs_pk.values), 0, len(pk) - 1)
        j2 = np.maximum(j - 1, 0)
        near = np.where(np.abs(pk[j] - sub.abs_pk.values)
                        <= np.abs(pk[j2] - sub.abs_pk.values), j, j2)
        d = np.abs(pk[near] - sub.abs_pk.values)
        dist[sub.index] = d
        eid[sub.index] = np.where(d <= args.match_bp, ids[near], -1)
        # span overlap, independent of peak distance
        es = e.sort_values("start")
        s, t = es.start.values, es.end.values
        i = np.searchsorted(t, sub.abs_lo.values, "left")
        hit[sub.index] = (i < len(s)) & (s[np.minimum(i, len(s) - 1)]
                                         <= sub.abs_hi.values)
    m["hit"], m["dist"], m["eid"] = hit, dist, eid

    # ---- recovery stratified by the atlas module's own support --------------
    rows = []
    for lo, hi in SUPPORT_BINS:
        s = m[(m.n_tfs_assigned >= lo) & (m.n_tfs_assigned <= hi)]
        if not len(s):
            continue
        rows.append(dict(support=f"{lo}-{hi}" if hi < 10**9 else f"{lo}+",
                         modules=len(s), recovered_pct=s.hit.mean() * 100,
                         median_peak_offset_bp=float(np.median(s["dist"])),
                         at_or_above_floor=lo >= args.floor))
    rec = pd.DataFrame(rows)
    rec.to_csv(out_dn / "recovery_by_support.tsv", sep="\t", index=False,
               float_format="%.2f")

    above = m[m.n_tfs_assigned >= args.floor]
    below = m[m.n_tfs_assigned < args.floor]
    matched = above[above.eid >= 0]
    n_elem = matched.eid.nunique()
    per = matched.groupby("eid").size()
    genes = matched.groupby("eid")["gene_name"].nunique()
    strat = g.set_index("element_id").loc[matched.eid.unique(), "stratum"]

    matched[["module_id", "chrom", "abs_pk", "gene_name", "n_tfs_assigned",
             "eid", "dist"]].to_csv(out_dn / "module_to_element.tsv",
                                    sep="\t", index=False)

    gate_pct = above.hit.mean() * 100
    stats = {
        "atlas modules compared": int(len(m)),
        "genome elements": int(len(g)),
        f"atlas modules >= floor {args.floor}": int(len(above)),
        "RECOVERED (the gate)": f"{gate_pct:.1f}%",
        "median peak offset (recovered)": f"{np.median(above['dist']):.0f} bp",
        f"atlas modules < floor {args.floor}": int(len(below)),
        "  of those, recovered": f"{below.hit.mean()*100:.1f}%",
        "distinct elements matched": int(n_elem),
        "collapse factor": f"{len(matched)/max(n_elem,1):.2f} modules/element",
        "elements absorbing 2+ modules": int((per >= 2).sum()),
        "  spanning >1 gene": int((genes >= 2).sum()),
        "matched element strata":
            " / ".join(f"{k} {v:,}" for k, v in strat.value_counts().items()),
    }

    print()
    _log("=== recovery by the atlas module's own assigned-TF count ===")
    print(rec.to_string(index=False, float_format=lambda x: f"{x:,.1f}"))
    print()
    for k, v in stats.items():
        print(f"  {k:<38} {v}")

    write_analysis_readme(
        out_dn,
        title="Regression gate: annotation-free discovery vs the promoter atlas",
        rationale=(
            "genome_modules.py discovers elements without reference to genes. "
            "Wherever it and the gene-centric atlas examine the same DNA they "
            "must agree; if they do not, the genome-wide pipeline is wrong and "
            "nothing it reports about distal elements can be believed.\n\n"
            "**Read the stratified table, not the overall rate.** The atlas was "
            "built at a support floor of 2 and the genome run at the calibrated "
            "floor of 11, so most atlas modules cannot be recovered by "
            "construction and an overall percentage is dominated by them. The "
            "gate is the recovery rate for atlas modules at or above the genome "
            "floor.\n\n"
            "Peaks are stored on the 25 bp recentering grid (RECENTER_HALF=12), "
            "so a median offset near 12 bp is half a bin -- agreement to the "
            "limit of the representation.\n\n"
            "Module and element counts are NOT expected to match. The atlas is "
            "indexed per TSS, so one locus appears once per nearby TSS while "
            "annotation-free discovery finds it once. The collapse factor below, "
            "and how many collapsed elements span more than one gene, measure "
            "exactly the composite-identity problem the per-TSS schema has and "
            "this one does not."),
        params={"floor (n_tfs_assigned)": args.floor,
                "match_bp (peak-to-peak)": args.match_bp,
                "support_bins": str(SUPPORT_BINS)},
        inputs={"genome_dir": str(gdir), "modules": str(mods),
                "tier": TIER, "tf_set": TF_SET},
        stats=stats)
    _log(f"wrote {out_dn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
