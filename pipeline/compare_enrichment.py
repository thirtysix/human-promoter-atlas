#!/usr/bin/env python3
"""
Compare GO-BP enrichment quality between builds, as an external arbiter for
MIN_SCORE_ASSIGN.

Why this and not similarity to the incumbent
--------------------------------------------
Agreement with the existing q1e-50 atlas cannot decide the threshold: the whole
premise of loosening it is that the incumbent under-counts sparse factors, so
optimising for agreement guarantees we can never improve on it. GO coherence is
external to both builds -- if the modules recovered by a lower threshold carry
real signal, program enrichment should get MORE SPECIFIC; if they are noise,
enrichment should wash out toward generic terms.

Metrics, and what each is guarding against
------------------------------------------
programs_enriched   programs with >=1 term at q<0.05. Falling = programs losing
                    coherent biology entirely.
median_top_q        strength of each program's best term. Necessary but NOT
                    sufficient -- a huge generic term scores well here.
median_top_or       odds ratio of the best term. Specificity: a program defined
                    by a real, restricted process has a high OR; one defined by
                    "everything binds housekeeping genes" has an OR near 1.
distinct_top_terms  distinct terms across all programs' top-5. Falling = several
                    programs collapsing onto the same generic term, i.e. the
                    programs stop being different from each other.
generic_share       fraction of top terms that are very large GO sets
                    (>=1000 genes). Rising = drifting toward CELL_CYCLE-style
                    catch-alls, the signature of dilution.

Usage:
    python pipeline/compare_enrichment.py LABEL=DIR [LABEL=DIR ...] [--k 12]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

GENERIC_MIN_SET = 1000
SIG_Q = 0.05


def summarise(enrich_dn: Path, k: int) -> dict | None:
    kdn = enrich_dn / f"k{k}"
    if not kdn.is_dir():
        return None
    tops, n_enriched, n_prog = [], 0, 0
    for f in sorted(kdn.glob("program*.gobp.tsv")):
        n_prog += 1
        df = pd.read_csv(f, sep="\t")
        if df.empty:
            continue
        sig = df[df["q_value"] < SIG_Q]
        if sig.empty:
            continue
        n_enriched += 1
        sig = sig.sort_values("q_value")
        for _, r in sig.head(5).iterrows():
            tops.append({"program": f.stem, "term": r["term"],
                         "q": r["q_value"], "or": r["odds_ratio"],
                         "bg": r["set_size_in_bg"]})
    if not tops:
        return {"programs": n_prog, "enriched": 0}
    t = pd.DataFrame(tops)
    best = t.groupby("program").first()
    import numpy as np
    return {
        "programs": n_prog,
        "enriched": n_enriched,
        "median_top_q": float(np.median(-np.log10(best["q"].clip(lower=1e-300)))),
        "median_top_or": float(best["or"].median()),
        "distinct_top_terms": int(t["term"].nunique()),
        "possible_top_terms": len(t),
        "generic_share": float((t["bg"] >= GENERIC_MIN_SET).mean()),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("builds", nargs="+", help="LABEL=DIR")
    ap.add_argument("--k", type=int, default=12)
    args = ap.parse_args()

    rows = []
    for spec in args.builds:
        label, _, d = spec.partition("=")
        s = summarise(Path(d) / "enrichment_msigdb_gobp_modules", args.k)
        if s is None:
            print(f"  {label}: no enrichment for k={args.k} under {d}")
            continue
        s["label"] = label
        rows.append(s)
    if not rows:
        return 1

    print(f"GO-BP enrichment, k={args.k}   (sig q<{SIG_Q}; generic = GO set >={GENERIC_MIN_SET} genes)\n")
    hdr = (f"{'build':>8}{'progs':>7}{'enriched':>10}{'med -log10q':>13}"
           f"{'med OR':>9}{'distinct top5':>15}{'generic share':>15}")
    print(hdr); print("-" * len(hdr))
    for r in rows:
        if not r.get("enriched"):
            print(f"{r['label']:>8}{r['programs']:>7}{0:>10}{'-':>13}{'-':>9}{'-':>15}{'-':>15}")
            continue
        print(f"{r['label']:>8}{r['programs']:>7}{r['enriched']:>10}"
              f"{r['median_top_q']:>13.1f}{r['median_top_or']:>9.2f}"
              f"{r['distinct_top_terms']:>8}/{r['possible_top_terms']:<6}"
              f"{r['generic_share']:>14.0%}")
    print("\n  higher median OR + higher distinct-top5 + LOWER generic share = "
          "more specific programs")
    print("  a build that gains modules but loses OR / distinctness is recovering noise")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
