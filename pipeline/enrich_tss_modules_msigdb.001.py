#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
GO BP enrichment per NMF program for the tss_modules pipeline.

For each k in KS, reads tss_modules/nmf.k{K}.module_program.tsv, groups modules
by dominant_program -> unique gene_names, and runs hypergeometric enrichment
against the MSigDB c5.go.bp gene-set collection. Background is the MSigDB
gene universe (genome_bg) — the standard biological interpretation; tf_bg-style
foreground/background restriction is not meaningful here since modules can
belong to any of ~18k genes.

Tests:
    foreground = genes whose canonical transcript has >=1 module with
                 dominant_program == p
    background = MSigDB GO BP gene universe (~18k genes)
    one-sided upper-tail hypergeometric p-value, BH-FDR within program

Outputs (enrichment_msigdb_gobp_modules/k{K}/):
    program{P}.gobp.tsv    full table with q_value, odds_ratio
    program_top_terms.tsv  per-program top-N q<0.05 terms (for summary table)
    dotplot.{png,pdf}      top-5 terms per program, deduped union
"""

################################################################################
# Libraries ####################################################################
################################################################################
import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom

# Machine-specific paths and build axes -> pipeline/config.py
from config import MSIGDB_FN, OUT_DN, K_CANONICAL


################################################################################
# Initiating Variables #########################################################
################################################################################
MODULES_DN = OUT_DN / "tss_modules"
ENRICH_DN  = OUT_DN / "enrichment_msigdb_gobp_modules"

# Always include the canonical rank, or the app's GO tab silently has no
# terms for the rank it actually displays.
KS         = sorted({8, 10, 12, 15, 20, K_CANONICAL})
TOP_N_TERMS = 5         # for dotplot
TOP_N_PER_PROGRAM = 15  # for summary table
QVAL_CUTOFF = 0.05
MIN_TERM_BG = 3

sns.set_style("whitegrid")
plt.rcParams["font.size"] = 10


################################################################################
# Functions ####################################################################
################################################################################
def load_msigdb_terms(path: str) -> pd.DataFrame:
    with open(path) as f:
        d = json.load(f)
    rows = []
    for term, meta in d.items():
        rows.append({"term": term,
                     "go_id": meta.get("exactSource", ""),
                     "genes": set(meta.get("geneSymbols", []))})
    return pd.DataFrame(rows)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.clip(q, 0, 1)
    return out


def enrich_program(fg_set: set, bg_set: set, terms_df: pd.DataFrame,
                   min_term_bg: int = MIN_TERM_BG) -> pd.DataFrame:
    N = len(bg_set)
    n = len(fg_set & bg_set)
    rows = []
    for _, t in terms_df.iterrows():
        term_genes = t["genes"] & bg_set
        K = len(term_genes)
        if K < min_term_bg:
            continue
        overlap = term_genes & fg_set
        k = len(overlap)
        if k == 0:
            continue
        p = hypergeom.sf(k - 1, N, K, n)
        a = k + 0.5; b = (n - k) + 0.5
        c = (K - k) + 0.5; d_ = ((N - K) - (n - k)) + 0.5
        odds = (a / b) / (c / d_)
        rows.append({"term": t["term"], "go_id": t["go_id"],
                     "set_size_in_bg": K,
                     "fg_in": k, "fg_total": n,
                     "bg_in": K, "bg_total": N,
                     "odds_ratio": odds, "p_value": p,
                     "genes_in_overlap": ",".join(sorted(overlap))})
    if not rows:
        return pd.DataFrame(columns=["term","go_id","set_size_in_bg","fg_in",
                                     "fg_total","bg_in","bg_total","odds_ratio",
                                     "p_value","q_value","genes_in_overlap"])
    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    return out.sort_values(["q_value", "p_value"]).reset_index(drop=True)


def plot_dotplot(per_program_results: dict, out_path_stem: str,
                 top_n: int = TOP_N_TERMS, qcut: float = QVAL_CUTOFF):
    long_rows = []
    for p, df in per_program_results.items():
        if df.empty:
            continue
        for _, r in df.head(top_n).iterrows():
            long_rows.append({"program": p, "term": r["term"],
                              "q_value": r["q_value"],
                              "odds_ratio": r["odds_ratio"]})
    if not long_rows:
        print(f"  no terms to plot for {out_path_stem}")
        return
    long_df = pd.DataFrame(long_rows)
    sig_terms = set(long_df.loc[long_df["q_value"] < qcut, "term"])
    if not sig_terms:
        sig_terms = set(long_df["term"])

    full_rows = []
    for p, df in per_program_results.items():
        if df.empty:
            continue
        sub = df[df["term"].isin(sig_terms)]
        for _, r in sub.iterrows():
            full_rows.append({"program": p, "term": r["term"],
                              "q_value": r["q_value"],
                              "odds_ratio": r["odds_ratio"]})
    full = pd.DataFrame(full_rows)
    full["neg_log10_q"] = -np.log10(full["q_value"].clip(lower=1e-15)).clip(upper=10)
    full["log2_or"]     = np.log2(full["odds_ratio"].clip(lower=1e-3))

    term_best = full.loc[full.groupby("term")["q_value"].idxmin()][["term","program","q_value"]]
    term_best = term_best.sort_values(["program", "q_value"])
    term_order = term_best["term"].tolist()
    program_order = sorted(full["program"].unique())

    full["term"]    = pd.Categorical(full["term"],    categories=term_order,    ordered=True)
    full["program"] = pd.Categorical(full["program"], categories=program_order, ordered=True)

    height = max(4, 0.25 * len(term_order) + 1.5)
    width  = max(5, 0.55 * len(program_order) + 4)
    fig, ax = plt.subplots(figsize=(width, height))
    vmax_or = max(abs(full["log2_or"].min()), abs(full["log2_or"].max()), 1.0)
    sc = ax.scatter(
        full["program"].cat.codes, full["term"].cat.codes,
        s=full["neg_log10_q"] * 30 + 5,
        c=full["log2_or"], cmap="RdBu_r",
        vmin=-vmax_or, vmax=vmax_or,
        edgecolors="black", linewidth=0.3,
    )
    ax.set_xticks(range(len(program_order)))
    ax.set_xticklabels([f"P{p}" for p in program_order])
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels([t.replace("GOBP_", "").replace("_", " ").lower()
                        for t in term_order], fontsize=8)
    ax.set_xlabel("Program")
    ax.set_title(f"GO BP enrichment per program (top {top_n} per program, q<{qcut})")
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("log2(odds ratio)")
    sizes = [1, 3, 5, 10]
    handles = [plt.scatter([], [], s=s * 30 + 5, color="grey",
                            edgecolors="black", linewidth=0.3,
                            label=f"-log10 q = {s}") for s in sizes]
    ax.legend(handles=handles, bbox_to_anchor=(1.18, 1), loc="upper left",
              fontsize=8, frameon=True, title="Significance")
    fig.tight_layout()
    fig.savefig(out_path_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_path_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


################################################################################
# Execution ####################################################################
################################################################################
def main():
    print(f"loading MSigDB GO BP from {MSIGDB_FN}")
    terms_df = load_msigdb_terms(MSIGDB_FN)
    universe = set().union(*terms_df["genes"])
    print(f"  {len(terms_df):,} terms; gene universe = {len(universe):,}")

    ENRICH_DN.mkdir(parents=True, exist_ok=True)

    for k in KS:
        mp_fn = MODULES_DN / f"nmf.k{k}.module_program.tsv"
        if not mp_fn.exists():
            print(f"k={k}: missing {mp_fn.name}, skipping")
            continue
        print(f"\n=== k={k} ===")
        mp = pd.read_csv(mp_fn, sep="\t")
        out_k = ENRICH_DN / f"k{k}"
        out_k.mkdir(parents=True, exist_ok=True)

        per_program = {}
        top_rows_all = []
        for p in sorted(mp["dominant_program"].unique()):
            sub = mp[mp["dominant_program"] == p]
            fg = set(sub["gene_name"].dropna().astype(str)) & universe
            n_modules = len(sub)
            res = enrich_program(fg, universe, terms_df, MIN_TERM_BG)
            per_program[int(p)] = res
            n_sig = int((res["q_value"] < QVAL_CUTOFF).sum()) if not res.empty else 0
            print(f"  P{p}: n_modules={n_modules:5d}  fg_genes={len(fg):4d}  "
                  f"terms_tested={len(res):4d}  q<{QVAL_CUTOFF}: {n_sig}")
            res.to_csv(out_k / f"program{p}.gobp.tsv", sep="\t", index=False)
            for rank, (_, r) in enumerate(res.head(TOP_N_PER_PROGRAM).iterrows(),
                                          1):
                top_rows_all.append({
                    "program": int(p), "rank": rank,
                    "term": r["term"], "go_id": r["go_id"],
                    "fg_in": r["fg_in"], "set_size_in_bg": r["set_size_in_bg"],
                    "odds_ratio": round(float(r["odds_ratio"]), 2),
                    "p_value": r["p_value"], "q_value": r["q_value"],
                    "genes_in_overlap": r["genes_in_overlap"],
                })
        pd.DataFrame(top_rows_all).to_csv(
            out_k / "program_top_terms.tsv", sep="\t", index=False)
        plot_dotplot(per_program, str(out_k / "dotplot"))
        print(f"  -> {out_k}/")

    print("\nDONE")


if __name__ == "__main__":
    main()
