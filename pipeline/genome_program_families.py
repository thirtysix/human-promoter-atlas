#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Group the k=140 genome programs into named families.

140 numbered programs is not a vocabulary. Clustering them by the TFs they load
on collapses PRC2, PRC1.1 and PRC1.6 into one Polycomb family, CTCF/cohesin
into an architectural family, and so on -- a browsable hierarchy for the app
and ~15 nameable families for the paper instead of 140 integers.

THIS IS A PRESENTATION LAYER, NOT A GENE-LEVEL CLAIM
----------------------------------------------------
Families do NOT rescue gene archetypes. diag_gene_coherence.py measured that
gene identity carries no information about an element's program once genomic
distance is controlled: same-gene element pairs are slightly LESS similar than
different-gene pairs at matched separation, in all seven separation bins
(overall 0.2006 vs 0.2243). Similarity tracks proximity, not gene membership.
Aggregating families per gene would therefore produce archetypes by arithmetic
rather than recovering diluted structure, so this script deliberately stops at
the program level and emits nothing per gene.

SIMILARITY IS CO-OCCURRENCE, NOT COMPOSITION
--------------------------------------------
Clustering the H rows by cosine -- do two programs use the same TFs -- was
tried first and does not work. NMF drives components onto disjoint factor sets,
so measured here the median between-program cosine is 0.0002, only 10 of 9,730
pairs exceed 0.3, none exceed 0.5, and the silhouette peaks at 0.008: no
structure at all, and the clustering degenerates to one family holding 132 of
140 programs.

That failure is informative rather than technical. PRC2 (EZH2/SUZ12/JARID2),
PRC1.1 (BCOR/KDM2B) and PRC1.6 (E2F6/L3MBTL2/MGA) are the family this was meant
to recover, and they share almost NO subunits -- cos(PRC2, PRC1.1) = 0.009.
Composition cannot group them because they are not compositionally alike.

What they share is location: Polycomb complexes co-occupy the same repressed
domains. Correlating programs across elements finds exactly that, with
corr(PRC2, PRC1.1) = +0.256 against a median of +0.107, and it also reunites
the AP-1 programs that k=140 split (p3 ~ p14, r = +0.664). Co-occurrence is
therefore the default; --similarity composition is kept for comparison.

Clustering is hierarchical average linkage -- deterministic, so no seed can
change the vocabulary between runs. The family count is chosen by silhouette
over a swept range and the whole curve is reported, because a silhouette peak
on a smooth curve is a weak optimum and readers should see how weak.

Usage:
    python pipeline/genome_program_families.py --genome-dir <dir> --k 140
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
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score

from config import TIER, TF_SET, is_substantive, write_analysis_readme

M_RANGE = range(5, 41)
TOP_TFS_PER_FAMILY = 8


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--families", type=int, default=None,
                    help="override the silhouette choice")
    ap.add_argument("--similarity", choices=["cooccurrence", "composition"],
                    default="cooccurrence",
                    help="cooccurrence (default): correlate programs across "
                         "elements -- do they occupy the same places. "
                         "composition: cosine on TF loadings -- do they use "
                         "the same factors. Composition does not work here; "
                         "see the module docstring.")
    args = ap.parse_args()
    root = Path(args.genome_dir)

    H = pd.read_csv(root / f"nmf.k{args.k}.H.tsv.gz", sep="\t", index_col=0)
    s = pd.read_csv(root / f"nmf.k{args.k}.summary.tsv", sep="\t")
    if "substantive" not in s.columns:
        s["substantive"] = is_substantive(s.n_elements, s.median_cosine)
    tf_names = list(H.columns)
    A = H.to_numpy(dtype=np.float64)
    A = A / np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
    if args.similarity == "composition":
        S = A @ A.T
    else:
        z = np.load(root / f"nmf.k{args.k}.W.npz")
        Wc = z["W"].astype(np.float64)
        Wc = Wc - Wc.mean(0, keepdims=True)
        Wc /= np.maximum(np.linalg.norm(Wc, axis=0, keepdims=True), 1e-12)
        S = Wc.T @ Wc
    D = np.clip(1.0 - S, 0.0, None)
    np.fill_diagonal(D, 0.0)
    D = (D + D.T) / 2.0                          # enforce exact symmetry
    Z = linkage(squareform(D, checks=False), method="average")
    off = S[np.triu_indices(args.k, 1)]
    _log(f"{args.k} programs, {len(tf_names):,} TFs; {args.similarity} "
         f"similarity median {np.median(off):+.4f}, p99 "
         f"{np.quantile(off, .99):+.4f}, max {off.max():+.4f}")

    sil = []
    for M in M_RANGE:
        lab = fcluster(Z, M, criterion="maxclust")
        if len(np.unique(lab)) < 2:
            continue
        sil.append((M, float(silhouette_score(D, lab, metric="precomputed"))))
    sil_df = pd.DataFrame(sil, columns=["n_families", "silhouette"])
    best = int(sil_df.loc[sil_df.silhouette.idxmax(), "n_families"])
    M = args.families or best
    _log(f"silhouette peak at M={best}"
         + (f"; using M={M} (override)" if args.families else ""))

    lab = fcluster(Z, M, criterion="maxclust")
    fam = pd.DataFrame({"program": H.index.astype(str), "family": lab})
    fam["program_num"] = fam.program.str.replace("prog", "", regex=False).astype(int)
    fam = fam.merge(s[["program", "n_elements", "median_cosine", "substantive",
                       "distal_log2FE_matched", "promoter_log2FE_matched"]],
                    left_on="program_num", right_on="program", suffixes=("", "_y"))

    rows = []
    for f, g in fam.groupby("family"):
        idx = g.program_num.to_numpy() - 1
        mean_load = A[idx].mean(axis=0)
        top = [tf_names[i] for i in np.argsort(mean_load)[::-1][:TOP_TFS_PER_FAMILY]]
        rows.append(dict(
            family=int(f), n_programs=len(g),
            n_substantive=int(g.substantive.sum()),
            n_elements=int(g.n_elements.sum()),
            median_stability=float(g.median_cosine.median()),
            distal_log2FE_matched=float(np.average(
                g.distal_log2FE_matched, weights=np.maximum(g.n_elements, 1))),
            promoter_log2FE_matched=float(np.average(
                g.promoter_log2FE_matched, weights=np.maximum(g.n_elements, 1))),
            top_tfs=", ".join(top)))
    fsum = pd.DataFrame(rows).sort_values("n_elements", ascending=False)

    out_dn = root / f"program_families.k{args.k}"
    out_dn.mkdir(parents=True, exist_ok=True)
    sil_df.to_csv(out_dn / "silhouette.tsv", sep="\t", index=False,
                  float_format="%.6f")
    fam[["program_num", "family", "n_elements", "median_cosine", "substantive"]
        ].rename(columns={"program_num": "program"}).to_csv(
        out_dn / "program_family.tsv", sep="\t", index=False)
    fsum.to_csv(out_dn / "family_summary.tsv", sep="\t", index=False,
                float_format="%.4f")

    print()
    _log(f"=== {M} families over {args.k} programs ===")
    print(f"{'fam':>4}{'prog':>6}{'subst':>7}{'elements':>10}{'stab':>7}"
          f"{'distFEm':>9}{'promFEm':>9}  top TFs")
    for _, r in fsum.iterrows():
        print(f"{int(r.family):>4}{int(r.n_programs):>6}{int(r.n_substantive):>7}"
              f"{int(r.n_elements):>10,}{r.median_stability:>7.3f}"
              f"{r.distal_log2FE_matched:>9.2f}{r.promoter_log2FE_matched:>9.2f}"
              f"  {r.top_tfs[:52]}")
    sd = sil_df.set_index("n_families").silhouette
    print(f"\n  silhouette: peak {sd.max():.3f} at M={best}; "
          f"range over M in [{M_RANGE.start},{M_RANGE.stop-1}] "
          f"= {sd.min():.3f}..{sd.max():.3f}")
    print("  A shallow peak means the family count is a presentation choice, "
          "not a\n  discovered quantity -- read the families, not the number.")

    write_analysis_readme(
        out_dn,
        title=f"Program families over the k={args.k} genome programs",
        rationale=(
            "140 numbered programs is not a vocabulary. Clustering them by the "
            "TFs they load on gives a browsable hierarchy for the app and a "
            "nameable set for the paper.\n\n"
            "**Families are a presentation layer and do not license gene-level "
            "archetypes.** diag_gene_coherence.py measured that gene identity "
            "carries no information about an element's program once genomic "
            "distance is controlled -- same-gene pairs are slightly LESS "
            "similar than different-gene pairs at matched separation, in all "
            "seven bins. Aggregating families per gene would manufacture "
            "structure rather than recover it, so nothing per gene is emitted "
            "here.\n\n"
            "Clustering is deterministic (hierarchical, average linkage on "
            "cosine distance), so no seed can change the vocabulary between "
            "runs. The silhouette curve is reported in full because its peak "
            "is shallow: the family count is a choice about readability, not a "
            "discovered quantity."),
        params={"k": args.k, "n_families": M,
                "silhouette_peak_at": best,
                "selection": "silhouette" if not args.families else "override",
                "similarity": args.similarity,
                "linkage": "average linkage"},
        inputs={"genome_dir": str(root), "tier": TIER, "tf_set": TF_SET},
        stats={"families": M,
               "largest family (programs)": int(fsum.n_programs.max()),
               "singleton families": int((fsum.n_programs == 1).sum()),
               "silhouette at chosen M":
                   f"{float(sd.loc[M]):.3f}" if M in sd.index else "n/a"})
    _log(f"wrote {out_dn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
