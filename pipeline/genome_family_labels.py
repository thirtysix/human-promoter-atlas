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
# BP is 3,492 usable sets against CC's 131, so it always contains some
# 3-member term with a perfect overlap, and a 3/3 hit has enormous odds.
# Pooling the libraries let those outbid real complexes: the cohesin family
# lost "mitotic cohesin complex" to "positive regulation of T helper 17 cell
# lineage", and the myeloid family became "mammary gland involution". A
# floor on BP set size keeps the terms broad enough to be names.
MIN_SET_SIZE_BP = 10
# A COMPLEX that names a set of co-binding TFs should itself be a TF
# complex. The labels that came out wrong were terms about something else
# that happen to contain a few TFs: apical junction complex is 7 TFs of 157
# genes (0.04), ER protein containing complex 4 of 131 (0.03), golgi
# membrane 13 of 712 (0.02). Every correct label sits above 0.68 -- AP-1
# 1.00, ESC/E(Z) 0.94, PcG 0.83, cohesin 0.75, bBAF 0.70, Sin3 0.68 -- so
# the two groups separate cleanly. Applied to CC ONLY: a biological PROCESS
# legitimately involves non-TFs, and this filter would erase all of BP.
MIN_TF_FRACTION_CC = 0.5


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
    ap.add_argument("--bp", default=None,
                    help="MSigDB GO-BP json. CC names COMPLEXES, which the "
                         "structural/lineage families have none of -- BP "
                         "names them by the process they drive "
                         "(erythrocyte differentiation, stem cell "
                         "maintenance, oxidative stress response).")
    ap.add_argument("--mf", default=None, help="MSigDB GO-MF json")
    ap.add_argument("--alpha", type=float, default=FDR_ALPHA)
    args = ap.parse_args()
    root = Path(args.genome_dir)
    fdir = root / f"program_families.k{args.k}"

    pf = pd.read_csv(fdir / "program_family.tsv", sep="\t")
    tt = pd.read_csv(root / f"nmf.k{args.k}.top_tfs.tsv", sep="\t")
    tf_all = set(pd.read_csv(root / "tf_index.tsv", sep="\t").TF)
    sets, prov, meta = {}, {}, {}
    for lib, path in (("CC", args.cc), ("BP", args.bp), ("MF", args.mf)):
        if not path:
            continue
        raw = json.load(open(path))
        got = 0
        for k, v in raw.items():
            st = set(v["geneSymbols"]) & tf_all
            # CC is restricted to COMPLEX terms: it also annotates TFs to
            # compartments they merely pass through, and at small overlaps
            # those win on odds -- the nuclear-receptor family came back
            # "presynaptic cytosol" (3/5), the erythroid one "golgi
            # membrane" (3/13). BP and MF have no such failure mode: a
            # process or an activity IS a reasonable name for a TF set.
            if lib == "CC":
                if "COMPLEX" not in k.upper():
                    continue
                n_all = len(set(v["geneSymbols"]))
                if n_all and len(st) / n_all < MIN_TF_FRACTION_CC:
                    continue
            floor = MIN_SET_SIZE_BP if lib in ("BP", "MF") else MIN_SET_SIZE
            if floor <= len(st) <= MAX_SET_SIZE:
                sets[k] = st
                prov[k] = lib
                meta[k] = (v.get("exactSource", ""), v.get("msigdbURL", ""))
                got += 1
        _log(f"  {lib}: {got:,} usable sets of {len(raw):,}")
    _log(f"{len(tf_all):,} assayed TFs; {len(sets):,} usable sets total")

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
            rows.append(dict(family=fam, term=name, lib=prov[name], overlap=a,
                             set_size=len(st), family_size=len(mem),
                             odds=float(odds), p=float(p),
                             go_id=meta[name][0], url=meta[name][1],
                             # WHICH member TFs drove the call. More useful
                             # than a prose blurb: it is the evidence, and it
                             # lets a reader judge a 3/7 hit for themselves.
                             overlap_tfs=", ".join(sorted(mem & st))))
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
        # TIERED, not pooled: a complex is a better name for a set of
        # co-binding TFs than a process, so CC is exhausted before BP/MF is
        # consulted. Pooling them let narrow BP terms displace correct
        # complex labels.
        avail = e[(e.family == fam) & (e.q <= args.alpha)
                  & (~e.term.isin(claimed))]
        hits = avail[avail.lib == "CC"].sort_values(
            ["odds", "q"], ascending=[False, True])
        if not len(hits):
            hits = avail[avail.lib != "CC"].sort_values(
                ["odds", "q"], ascending=[False, True])
        if len(hits):
            best = hits.iloc[0]
            claimed.add(best.term)
            lab.append(dict(family=fam, label=_pretty(best.term),
                            term=best.term, lib=str(best.lib),
                            q=float(best.q),
                            overlap=int(best.overlap),
                            set_size=int(best.set_size), named=True,
                            n_terms_sig=int(len(hits))))
        else:
            # No term survives. Fall back to the top TFs and SAY SO, rather
            # than promoting a non-significant hit to a name.
            lab.append(dict(family=fam,
                            label=" / ".join(str(r.top_tfs).split(", ")[:3]),
                            term="", lib="", q=np.nan, overlap=0, set_size=0,
                            named=False, n_terms_sig=0))
    L = pd.DataFrame(lab)
    L.to_csv(fdir / "family_labels.tsv", sep="\t", index=False,
             float_format="%.3g")

    # Every significant term per family, not just the winner. One label is a
    # summary; a family is usually enriched for several related terms and the
    # UI should be able to show them.
    terms = (e[e.q <= args.alpha]
             .sort_values(["family", "q", "odds"], ascending=[True, True, False])
             .copy())
    terms["label"] = terms.term.map(_pretty)
    terms["rank"] = terms.groupby("family").cumcount() + 1
    # The list is ranked by q (evidence) but the PRIMARY label is chosen by
    # tier-then-odds, so row 1 is often not the label. Mark it, or the UI shows
    # a headline that does not match the top row beneath it.
    chosen = set(zip(L.family, L.term))
    terms["is_label"] = [(f, t) in chosen
                         for f, t in zip(terms.family, terms.term)]
    terms[["family", "rank", "is_label", "label", "term", "lib", "go_id", "url",
           "overlap", "set_size", "odds", "q", "overlap_tfs"]].to_csv(
        fdir / "family_terms.tsv", sep="\t", index=False, float_format="%.4g")
    per_family = terms.groupby("family").size()
    _log(f"family_terms.tsv: {len(terms):,} significant terms across "
         f"{terms.family.nunique()} families "
         f"(median {int(per_family.median())} each, max {int(per_family.max())})")

    n_named = int(L.named.sum())
    print()
    _log(f"=== family labels (FDR <= {args.alpha}) ===")
    m = L.merge(fs[["family", "n_elements", "top_tfs"]], on="family")
    for _, r in m.sort_values("n_elements", ascending=False).iterrows():
        flag = "" if r.named else "   [unnamed]"
        q = (f"[{r.lib}] q={r.q:.1e} {r.overlap}/{r.set_size}"
             if r.named else "")
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
