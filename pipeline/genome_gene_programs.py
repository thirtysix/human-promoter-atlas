#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 #############################################################
"""
Per-gene program composition from genome-wide elements.

The gene-centric front door and the genome-wide analysis share ONE program
vocabulary: gene pages show genome programs on the elements near that gene,
rather than a second factorization of promoter modules. This builds the
gene x program matrix that makes that possible -- the analogue of
tss_archetypes.001.py's [n_gene x 10] module-count matrix, with elements in
place of modules and genome programs in place of promoter programs.

Two outputs:
    occupancy.gene_x_program.npz  [n_gene x k] counts, for archetype NMF
    gene_program_composition.tsv  per-gene summary, for the gene page

WHICH ELEMENTS COUNT TOWARD A GENE
----------------------------------
Default is promoter + proximal (<=10 kb). Distal elements are EXCLUDED by
default, and the reason is measured rather than assumed: 56.6% of distal
elements have a rival TSS within twice the distance to their nearest one
(genome_annotate_genes.py). Counting them would attribute roughly half of
~353,550 enhancers to the wrong gene and bake that into every gene's program
composition -- and then into archetypes derived from it, where the error
becomes invisible.

--strata all is available for the analysis that wants it, but the default is
the one that can be defended per gene rather than only in aggregate.

Usage:
    python pipeline/genome_gene_programs.py --genome-dir <dir> --k 140
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
import scipy.sparse as sp

from config import TIER, TF_SET, write_analysis_readme, is_substantive

STRATA_DEFAULT = ("promoter", "proximal")


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--strata", default="promoter,proximal",
                    help="'all' or a comma list; default excludes distal "
                         "because its nearest-gene link is a coin flip for "
                         "56.6% of elements")
    ap.add_argument("--aggregate", choices=["soft", "hard"], default="soft",
                    help="soft (default) sums each element's full program "
                         "weight vector; hard counts elements by their argmax "
                         "program. Hard is degenerate at k=140: a gene has ~5 "
                         "nearby elements and 140 programs, so 55.3%% of genes "
                         "end up with every program holding exactly one "
                         "element and the argmax is an arbitrary tie-break.")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    root = Path(args.genome_dir)

    el = pd.read_csv(root / "elements.genes.tsv", sep="\t", dtype={"chrom": str})
    ep = pd.read_csv(root / f"nmf.k{args.k}.element_program.tsv.gz", sep="\t",
                     dtype={"chrom": str})
    prog = pd.read_csv(root / f"nmf.k{args.k}.summary.tsv", sep="\t")
    el = el.merge(ep[["element_id", "dominant_program", "dominant_weight"]],
                  on="element_id", how="left")
    if el.dominant_program.isna().any():
        raise SystemExit("elements without a program assignment -- "
                         "elements.genes.tsv and the factorization disagree")

    strata = (list(STRATA_DEFAULT) if args.strata == "" else
              (["promoter", "proximal", "distal"] if args.strata == "all"
               else [s.strip() for s in args.strata.split(",") if s.strip()]))
    # Empty gene labels round-trip through TSV as NaN, not "" -- filtering on
    # the empty string alone leaves floats in a string column and np.sort dies
    # comparing float to str.
    has_gene = el.nearest_gene_name.notna() & (el.nearest_gene_name.astype(str) != "")
    sel = el[el.stratum.isin(strata) & has_gene].copy()
    sel["nearest_gene_name"] = sel.nearest_gene_name.astype(str)
    dropped = int((el.stratum.isin(strata) & ~has_gene).sum())
    _log(f"{len(el):,} elements; {len(sel):,} in strata {strata}"
         + (f"; {dropped:,} dropped for having no gene label" if dropped else ""))

    genes = np.sort(sel.nearest_gene_name.unique())
    gidx = {g: i for i, g in enumerate(genes)}
    r = sel.nearest_gene_name.map(gidx).to_numpy()
    if args.aggregate == "hard":
        c = sel.dominant_program.to_numpy().astype(int) - 1
        M = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)),
                          shape=(len(genes), args.k)).tocsr()
        M.sum_duplicates()
    else:
        # Sum each element's ROW-NORMALISED weight vector, so every element
        # contributes total mass 1 spread over the programs it actually loads
        # on. Equivalent to hard counting when an element loads on one program,
        # and unlike hard counting it does not manufacture ties when it does
        # not.
        z = np.load(root / f"nmf.k{args.k}.W.npz")
        W, wid = z["W"], z["element_id"]
        pos = pd.Series(np.arange(len(wid)), index=wid)
        take = pos.reindex(sel.element_id.to_numpy()).to_numpy()
        if np.isnan(take).any():
            raise SystemExit("elements missing from W.npz -- not the same build")
        Wsel = W[take.astype(np.int64)]
        rs = Wsel.sum(axis=1, keepdims=True)
        Wsel = Wsel / np.where(rs > 0, rs, 1.0)
        M = np.zeros((len(genes), args.k), np.float32)
        np.add.at(M, r, Wsel)
        M = sp.csr_matrix(M)
        M.eliminate_zeros()

    out_dn = Path(args.out) if args.out else root / f"gene_programs.k{args.k}"
    out_dn.mkdir(parents=True, exist_ok=True)
    sp.save_npz(str(out_dn / "occupancy.gene_x_program.npz"), M)
    pd.DataFrame({"gene_name": genes, "gene_idx": range(len(genes))}).to_csv(
        out_dn / "gene_index.tsv", sep="\t", index=False)

    # per-gene summary for the gene page
    counts = np.asarray(M.sum(1)).ravel()
    dom = np.asarray(M.argmax(1)).ravel() + 1
    domn = np.asarray(M[np.arange(M.shape[0]), dom - 1]).ravel()
    # "programs present" needs a floor under soft aggregation, or numerical
    # dust counts as a program.
    Mb = M.copy(); Mb.data = (Mb.data >= 0.05).astype(np.float32)
    Mb.eliminate_zeros()
    nprog = np.diff(Mb.indptr)
    # summary.tsv may predate the substantive column; derive it from the
    # single definition in config rather than re-stating the rule here.
    if "substantive" not in prog.columns:
        prog["substantive"] = is_substantive(prog.n_elements,
                                             prog.median_cosine)
    sub = prog.set_index("program")["substantive"].to_dict()
    # NOT a single "dominant program". Measured at k=140: under hard counting
    # 55.3% of genes are a pure tie (every program holds exactly one element),
    # and under soft weighting 98.6% have a top program below 25% with a median
    # share of 0.054. A gene's elements genuinely belong to different programs;
    # an argmax over them reports an arbitrary winner as if it were a finding.
    # Report the top three and the spread instead, so a flat composition is
    # visible as flat.
    D = np.asarray(M.todense())
    tot = D.sum(1, keepdims=True)
    P = D / np.where(tot > 0, tot, 1.0)
    order = np.argsort(P, axis=1)[:, ::-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        ent = -(P * np.log(np.where(P > 0, P, 1.0))).sum(1)
    fields = {"gene_name": genes,
              "n_elements": counts.astype(int) if args.aggregate == "hard"
                            else np.rint(counts).astype(int),
              "n_programs": nprog.astype(int),
              # normalised entropy: 0 = one program holds everything,
              # 1 = mass spread evenly over all k
              "spread": ent / np.log(args.k)}
    for j in range(3):
        pj = order[:, j] + 1
        fields[f"top{j+1}_program"] = pj
        fields[f"top{j+1}_frac"] = P[np.arange(len(genes)), order[:, j]]
        fields[f"top{j+1}_substantive"] = [bool(sub.get(int(x), False)) for x in pj]
    comp = pd.DataFrame(fields)
    comp.to_csv(out_dn / "gene_program_composition.tsv", sep="\t", index=False,
                float_format="%.4f")

    _log(f"=== gene x program (k={args.k}, strata {'+'.join(strata)}) ===")
    print(f"  genes                    : {len(genes):,}")
    print(f"  elements counted         : {int(counts.sum()):,}")
    print(f"  median elements per gene : {np.median(counts):.0f}")
    print(f"  median programs per gene : {np.median(nprog):.0f}")
    print(f"  top program is substantive: {int(comp.top1_substantive.sum()):,} "
          f"({comp.top1_substantive.mean()*100:.1f}%)")
    print(f"  top program's share: median {comp.top1_frac.median():.3f}, "
          f"p90 {comp.top1_frac.quantile(.9):.3f}")
    print(f"  composition spread : median {comp.spread.median():.3f} "
          f"(0 = one program, 1 = even over all {args.k})")
    print(f"  matrix {M.shape}  nnz={M.nnz:,}  "
          f"density={M.nnz/np.prod(M.shape):.2%}")
    print("\n  most program-diverse genes:")
    print("\n  genes with the most CONCENTRATED composition:")
    for _, r_ in comp.nsmallest(6, "spread").iterrows():
        print(f"    {r_.gene_name:<12} {int(r_.n_elements):>3} elements, "
              f"top p{int(r_.top1_program):<3} {r_.top1_frac:5.1%}, "
              f"spread {r_.spread:.3f}"
              + ("  [substantive]" if r_.top1_substantive else ""))

    write_analysis_readme(
        out_dn,
        title=f"Per-gene composition over genome programs (k={args.k})",
        rationale=(
            "The gene-centric view and the genome-wide analysis share one "
            "program vocabulary: gene pages show genome programs on nearby "
            "elements rather than a second factorization of promoter modules. "
            "The 98.2% regression gate is what makes that sound -- both see "
            "the same promoters.\n\n"
            "**Distal elements are excluded by default.** 56.6% of them have a "
            "rival TSS within twice the distance to their nearest, so counting "
            "them would assign roughly half of ~353,550 enhancers to the wrong "
            "gene, and any archetype built on this matrix would inherit that "
            "error invisibly. Pass --strata all to include them deliberately."),
        params={"k": args.k, "strata": "+".join(strata)},
        inputs={"genome_dir": str(root), "tier": TIER, "tf_set": TF_SET,
                "elements_total": int(len(el)), "elements_counted": int(len(sel))},
        stats={"genes": int(len(genes)),
               "median elements per gene": float(np.median(counts)),
               "median programs per gene": float(np.median(nprog)),
               "top program substantive":
                   f"{comp.top1_substantive.mean()*100:.1f}%",
               "top program median share": f"{comp.top1_frac.median():.3f}",
               "median composition spread": f"{comp.spread.median():.3f}"})
    _log(f"wrote {out_dn}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
