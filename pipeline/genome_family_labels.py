#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Reproducible names for the program families.

Until now the families carried hand-written names -- "cohesin/CTCF", "PRC2",
"AP-1/BAF/TEAD" -- read off their TF lists by a human. Those are an
interpretation, not an output: nothing regenerates them, they go stale the
moment the clustering changes, and there is no statistic to report.

This replaces them with enrichment against a versioned reference. Each family's
member TFs are tested against every GO-CC set by Fisher's exact test, corrected
across all tests with Benjamini-Hochberg, and labelled with the best surviving
term. Same input gives the same label, the reference is citable, and every
label carries an FDR.

WHY GO-CC WORKS HERE WHEN IT FAILED AS A RANK CRITERION
-------------------------------------------------------
nmf_complex_recovery.py scored GO-CC by AMI and it was uninformative (~0.03,
within noise). That asked a much harder question: does the global 140-way
partition agree with GO-CC's global partition, over a reference where 169 of
351 used labels are singletons. This asks whether ONE family's TF set is
over-represented in ONE term. The complex terms that matter -- COHESIN_COMPLEX,
ESC_E_Z_COMPLEX, PRC1_COMPLEX, NURD_COMPLEX, SWI_SNF_COMPLEX,
INTEGRATOR_COMPLEX, SET1C_COMPASS_COMPLEX -- have enough members to answer it.

THE BACKGROUND IS THE ASSAYED TFs, NOT THE GENOME
-------------------------------------------------
Enrichment is computed against the 1,793 TFs in the occupancy matrix. Using all
protein-coding genes would make every family look enriched for
transcription-related terms, since the whole population is transcription
factors to begin with.

Families with no term surviving FDR are labelled by their top TFs and flagged
`named=False`, rather than being given the best non-significant hit -- an
unnamed family is a fact about the data, not a gap to paper over.

Usage:
    python pipeline/genome_family_labels.py --genome-dir <dir> --k 140 \\
        --cc data/msigdb/c5.go.cc.v2026.1.Hs.json
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import argparse
import datetime as dt
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

from config import TIER, TF_SET, write_analysis_readme

TOP_TFS_PER_PROGRAM = 20     # member TFs = union of each program's top N
FDR_ALPHA = 0.05
MIN_SET_SIZE = 3             # a 1-2 member set cannot be meaningfully enriched
# A 362-member term like TRANSCRIPTION_REGULATOR_COMPLEX is significant for
# almost every family and says nothing -- these are all transcription
# regulators. At 400 it won 9 of 28 families. Cap at a size that can still
# name a complex: cohesin is 6, Sin3 15, PcG 49.
MAX_SET_SIZE = 150
MIN_OVERLAP = 3              # 2/5 produced "protein folding chaperone" for
                             # the TFAP2/SOX10 family


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _pretty(term: str) -> str:
    """GOCC_ESC_E_Z_COMPLEX -> 'ESC E Z complex'."""
    t = re.sub(r"^GO[A-Z]{2}_", "", term).replace("_", " ")
    return t[:1] + t[1:].lower()


def bh(p):
    """Benjamini-Hochberg q-values."""
    p = np.asarray(p, float)
    n = p.size
    o = np.argsort(p)
    q = np.empty(n)
    q[o] = np.minimum.accumulate((p[o] * n / (np.arange(n) + 1))[::-1])[::-1]
    return np.clip(q, 0, 1)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--cc", required=True, help="MSigDB GO-CC json")
    ap.add_argument("--alpha", type=float, default=FDR_ALPHA)
    args = ap.parse_args()
    root = Path(args.genome_dir)
    fdir = root / f"program_families.k{args.k}"

    pf = pd.read_csv(fdir / "program_family.tsv", sep="\t")
    tt = pd.read_csv(root / f"nmf.k{args.k}.top_tfs.tsv", sep="\t")
    tf_all = set(pd.read_csv(root / "tf_index.tsv", sep="\t").TF)
    cc = json.load(open(args.cc))
    sets = {k: set(v["geneSymbols"]) & tf_all for k, v in cc.items()}
    # COMPLEX terms only. GO-CC also annotates TFs to compartments they pass
    # through or were incidentally observed in, and at small overlaps those
    # win on odds: the nuclear-receptor family was labelled "presynaptic
    # cytosol" (3/5) and the erythroid family "golgi membrane" (3/13). A
    # compartment is not a name for a set of co-binding TFs.
    sets = {k: v for k, v in sets.items()
            if "COMPLEX" in k.upper()
            and MIN_SET_SIZE <= len(v) <= MAX_SET_SIZE}
    _log(f"{len(tf_all):,} assayed TFs; {len(sets):,} usable GO-CC sets "
         f"(of {len(cc):,})")

    top_by_prog = {p: set(g.nlargest(TOP_TFS_PER_PROGRAM, "loading").tf)
                   for p, g in tt.groupby("program")}
    members = {}
    for fam, g in pf.groupby("family"):
        s = set()
        for p in g.program:
            s |= top_by_prog.get(int(p), set())
        members[int(fam)] = s & tf_all

    rows = []
    for fam, mem in members.items():
        if not mem:
            continue
        for name, st in sets.items():
            a = len(mem & st)
            if a < MIN_OVERLAP:
                continue
            b, c = len(mem) - a, len(st) - a
            d = len(tf_all) - a - b - c
            odds, p = fisher_exact([[a, b], [c, d]], alternative="greater")
            rows.append(dict(family=fam, term=name, overlap=a,
                             set_size=len(st), family_size=len(mem),
                             odds=float(odds), p=float(p)))
    e = pd.DataFrame(rows)
    if e.empty:
        raise SystemExit("no testable family/term overlaps")
    e["q"] = bh(e.p.to_numpy())
    e = e.sort_values(["family", "q", "p"])
    e.to_csv(fdir / "family_enrichment.tsv", sep="\t", index=False,
             float_format="%.6g")

    fs = pd.read_csv(fdir / "family_summary.tsv", sep="\t")
    # Labels must be UNIQUE. Without this three families all came back "NF
    # kappaB complex" and three more "beta catenin tcf complex", which is a
    # vocabulary that cannot distinguish its own entries. Terms are claimed
    # greedily by the family with the strongest evidence for them.
    claimed = set()
    order = fs.sort_values("n_elements", ascending=False)
    lab = []
    for _, r in order.iterrows():
        fam = int(r.family)
        # Among terms that clear FDR, take the most SPECIFIC (highest odds),
        # not the smallest p. p-value ranks by evidence strength, which
        # structurally favours large generic sets; a label needs the term
        # that most distinguishes this family from the background.
        hits = (e[(e.family == fam) & (e.q <= args.alpha)
                  & (~e.term.isin(claimed))]
                .sort_values(["odds", "q"], ascending=[False, True]))
        if len(hits):
            best = hits.iloc[0]
            claimed.add(best.term)
            lab.append(dict(family=fam, label=_pretty(best.term),
                            term=best.term, q=float(best.q),
                            overlap=int(best.overlap),
                            set_size=int(best.set_size), named=True,
                            n_terms_sig=int(len(hits))))
        else:
            # No term survives. Fall back to the top TFs and SAY SO, rather
            # than promoting a non-significant hit to a name.
            lab.append(dict(family=fam,
                            label=" / ".join(str(r.top_tfs).split(", ")[:3]),
                            term="", q=np.nan, overlap=0, set_size=0,
                            named=False, n_terms_sig=0))
    L = pd.DataFrame(lab)
    L.to_csv(fdir / "family_labels.tsv", sep="\t", index=False,
             float_format="%.3g")

    n_named = int(L.named.sum())
    print()
    _log(f"=== family labels (FDR <= {args.alpha}) ===")
    m = L.merge(fs[["family", "n_elements", "top_tfs"]], on="family")
    for _, r in m.sort_values("n_elements", ascending=False).iterrows():
        flag = "" if r.named else "   [unnamed]"
        q = f"q={r.q:.1e} {r.overlap}/{r.set_size}" if r.named else ""
        print(f"  fam {int(r.family):>3} ({int(r.n_elements):>7,} el)  "
              f"{r.label[:44]:<44} {q}{flag}")
    print(f"\n  named by enrichment: {n_named}/{len(L)}")
    print(f"  wrote {fdir/'family_labels.tsv'}")

    write_analysis_readme(
        fdir / "labels",
        title="Reproducible family labels by GO-CC enrichment",
        rationale=(
            "Hand-written family names are an interpretation, not an output: "
            "nothing regenerates them and they go stale when the clustering "
            "changes. Each family's member TFs are tested against every GO-CC "
            "set by Fisher's exact test, corrected with Benjamini-Hochberg, "
            "and labelled with the best surviving term, so the same input "
            "gives the same label and every label carries an FDR.\n\n"
            "GO-CC was uninformative as a RANK criterion (AMI ~0.03) because "
            "that asked whether the global 140-way partition matched GO-CC's, "
            "against a reference where half the used labels are singletons. "
            "Asking whether one family's TFs are over-represented in one term "
            "is a far easier question and the complex terms have enough "
            "members to answer it.\n\n"
            "Background is the assayed TFs, not all genes -- against a genomic "
            "background every family would look enriched for transcription "
            "terms. Families with nothing surviving FDR are labelled by top "
            "TFs and flagged named=False; an unnamed family is a fact, not a "
            "gap to fill with the best non-significant hit."),
        params={"k": args.k, "alpha": args.alpha,
                "top_tfs_per_program": TOP_TFS_PER_PROGRAM,
                "set_size_range": f"{MIN_SET_SIZE}-{MAX_SET_SIZE}",
                "reference": Path(args.cc).name},
        inputs={"genome_dir": str(root), "tier": TIER, "tf_set": TF_SET,
                "assayed_tfs": len(tf_all), "usable_sets": len(sets)},
        stats={"families": len(L), "named by enrichment": n_named,
               "unnamed": int((~L.named).sum())})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
