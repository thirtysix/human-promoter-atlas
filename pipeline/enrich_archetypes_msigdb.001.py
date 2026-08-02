#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""GO BP enrichment per gene archetype (mirrors enrich_tss_modules_msigdb.001.py
but at the gene-archetype level)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import hypergeom

# Machine-specific paths and build axes -> pipeline/config.py
from config import MSIGDB_FN, OUT_DN


ROOT                = OUT_DN   # config.OUT_DN
ARCH_DN    = ROOT / "tss_archetypes"
OUT_DN     = ROOT / "enrichment_msigdb_gobp_archetypes"

A_VALUES   = [4, 5, 6, 7, 8, 9]   # match tss_archetypes.001.py
TOP_N_TERMS_PLOT = 5
TOP_N_PER_ARCH    = 15
QVAL_CUTOFF = 0.05
MIN_TERM_BG = 3

sns.set_style("whitegrid")


def load_msigdb_terms(path: str) -> pd.DataFrame:
    with open(path) as f:
        d = json.load(f)
    rows = []
    for term, meta in d.items():
        rows.append({"term": term, "go_id": meta.get("exactSource", ""),
                     "genes": set(meta.get("geneSymbols", []))})
    return pd.DataFrame(rows)


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    p = np.asarray(p, dtype=float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    q = ranked * n / np.arange(1, n + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty_like(q); out[order] = np.clip(q, 0, 1)
    return out


def enrich(fg_set: set, bg_set: set, terms_df: pd.DataFrame) -> pd.DataFrame:
    N = len(bg_set); n = len(fg_set & bg_set)
    rows = []
    for _, t in terms_df.iterrows():
        term_genes = t["genes"] & bg_set
        K = len(term_genes)
        if K < MIN_TERM_BG: continue
        overlap = term_genes & fg_set
        k = len(overlap)
        if k == 0: continue
        p = hypergeom.sf(k - 1, N, K, n)
        a = k + 0.5; b = (n - k) + 0.5
        c = (K - k) + 0.5; d_ = ((N - K) - (n - k)) + 0.5
        odds = (a / b) / (c / d_)
        rows.append({"term": t["term"], "go_id": t["go_id"],
                     "set_size_in_bg": K, "fg_in": k, "fg_total": n,
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


def plot_dotplot(per_arch_results: dict, out_stem: str,
                  top_n: int = TOP_N_TERMS_PLOT, qcut: float = QVAL_CUTOFF):
    long_rows = []
    for a, df in per_arch_results.items():
        if df.empty: continue
        for _, r in df.head(top_n).iterrows():
            long_rows.append({"archetype": a, "term": r["term"],
                              "q_value": r["q_value"],
                              "odds_ratio": r["odds_ratio"]})
    if not long_rows: return
    long_df = pd.DataFrame(long_rows)
    sig = set(long_df.loc[long_df["q_value"] < qcut, "term"])
    if not sig: sig = set(long_df["term"])
    full = []
    for a, df in per_arch_results.items():
        if df.empty: continue
        sub = df[df["term"].isin(sig)]
        for _, r in sub.iterrows():
            full.append({"archetype": a, "term": r["term"],
                         "q_value": r["q_value"], "odds_ratio": r["odds_ratio"]})
    full = pd.DataFrame(full)
    full["neg_log10_q"] = -np.log10(full["q_value"].clip(lower=1e-15)).clip(upper=10)
    full["log2_or"]     = np.log2(full["odds_ratio"].clip(lower=1e-3))
    term_best = full.loc[full.groupby("term")["q_value"].idxmin()][["term","archetype","q_value"]]
    term_order = term_best.sort_values(["archetype", "q_value"])["term"].tolist()
    arch_order = sorted(full["archetype"].unique())
    full["term"]      = pd.Categorical(full["term"], categories=term_order, ordered=True)
    full["archetype"] = pd.Categorical(full["archetype"], categories=arch_order, ordered=True)
    height = max(4, 0.25 * len(term_order) + 1.5)
    width  = max(5, 0.55 * len(arch_order) + 4)
    fig, ax = plt.subplots(figsize=(width, height))
    vmax_or = max(abs(full["log2_or"].min()), abs(full["log2_or"].max()), 1.0)
    sc = ax.scatter(full["archetype"].cat.codes, full["term"].cat.codes,
                     s=full["neg_log10_q"] * 30 + 5, c=full["log2_or"],
                     cmap="RdBu_r", vmin=-vmax_or, vmax=vmax_or,
                     edgecolors="black", linewidth=0.3)
    ax.set_xticks(range(len(arch_order))); ax.set_xticklabels([f"A{a}" for a in arch_order])
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels([t.replace("GOBP_", "").replace("_", " ").lower()
                         for t in term_order], fontsize=8)
    ax.set_xlabel("archetype"); ax.invert_yaxis()
    ax.set_title(f"GO BP enrichment per archetype (top {top_n}, q<{qcut})")
    fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02).set_label("log2(odds ratio)")
    fig.tight_layout()
    fig.savefig(out_stem + ".png", dpi=300, bbox_inches="tight")
    fig.savefig(out_stem + ".pdf", bbox_inches="tight")
    plt.close(fig)


def main():
    print(f"loading MSigDB GO BP from {MSIGDB_FN}")
    terms_df = load_msigdb_terms(MSIGDB_FN)
    universe = set().union(*terms_df["genes"])
    print(f"  {len(terms_df):,} terms; universe = {len(universe):,}")
    OUT_DN.mkdir(parents=True, exist_ok=True)

    for A in A_VALUES:
        ga_fn = ARCH_DN / f"nmf.A{A}.gene_archetype.tsv"
        if not ga_fn.exists():
            print(f"A={A}: missing {ga_fn.name}, skipping")
            continue
        print(f"\n=== A={A} ===")
        ga = pd.read_csv(ga_fn, sep="\t")
        out_a = OUT_DN / f"A{A}"
        out_a.mkdir(parents=True, exist_ok=True)
        per_arch = {}
        top_rows_all = []
        for a in sorted(ga["dominant_archetype"].unique()):
            sub = ga[ga["dominant_archetype"] == a]
            fg = set(sub["gene_name"].dropna().astype(str)) & universe
            res = enrich(fg, universe, terms_df)
            per_arch[int(a)] = res
            n_sig = int((res["q_value"] < QVAL_CUTOFF).sum()) if not res.empty else 0
            print(f"  A{a}: n_genes={len(sub):5d}  fg={len(fg):4d}  "
                  f"terms={len(res):4d}  q<{QVAL_CUTOFF}: {n_sig}")
            res.to_csv(out_a / f"archetype{a}.gobp.tsv", sep="\t", index=False)
            for rank, (_, r) in enumerate(res.head(TOP_N_PER_ARCH).iterrows(), 1):
                top_rows_all.append({
                    "archetype": int(a), "rank": rank,
                    "term": r["term"], "go_id": r["go_id"],
                    "fg_in": int(r["fg_in"]),
                    "set_size_in_bg": int(r["set_size_in_bg"]),
                    "odds_ratio": round(float(r["odds_ratio"]), 2),
                    "p_value": float(r["p_value"]),
                    "q_value": float(r["q_value"]),
                    "genes_in_overlap": r["genes_in_overlap"],
                })
        pd.DataFrame(top_rows_all).to_csv(
            out_a / "archetype_top_terms.tsv", sep="\t", index=False)
        plot_dotplot(per_arch, str(out_a / "dotplot"))

    print("\nDONE")


if __name__ == "__main__":
    main()
