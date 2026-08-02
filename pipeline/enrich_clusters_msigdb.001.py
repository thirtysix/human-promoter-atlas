#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
GO BP enrichment per TF cluster, using the local MSigDB c5.go.bp JSON.

For each cluster table (filtered, no_filter), iterate clusters and run a
one-sided hypergeometric test (Fisher's exact, equivalent in this setup):

    foreground = TFs in cluster
    background = all TFs in the same cluster table (intersected with the
                 union of genes annotated in the MSigDB GO BP universe)

Output:
    enrichment_msigdb_gobp/<table>/cluster<C>.gobp.tsv
        Term, GO_ID, set_size_in_bg, fg_in, fg_total, bg_in, bg_total,
        odds_ratio, p_value, q_value, genes_in_overlap
    plots/enrichment_msigdb_gobp.<table>.dotplot.{png,pdf}
        x = cluster, y = top-5 q<0.05 terms per cluster (deduped union)
        size = -log10(q), color = log2(odds_ratio)

Hypergeometric enrichment factors:
    K = bg TFs in term, n = foreground size, k = fg in term, N = bg size
    p = sum_{i=k..min(n,K)} C(K, i) C(N-K, n-i) / C(N, n)   (upper tail)
    odds_ratio = (k / (n - k)) / ((K - k) / ((N - K) - (n - k)))   (with +0.5 cells)
BH-FDR for q-values (across all terms tested per cluster).
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
from config import MSIGDB_FN, OUT_DN


################################################################################
# Initiating Variables #########################################################
################################################################################

TABLES = {
    "filtered":  OUT_DN / "clustering"           / "tf_cluster_table.tsv",
    "no_filter": OUT_DN / "clustering_no_filter" / "tf_cluster_table.tsv",
}
ENRICH_DN  = OUT_DN / "enrichment_msigdb_gobp"
PLOTS_DN   = OUT_DN / "plots"
TOP_N_TERMS = 5
QVAL_CUTOFF = 0.05
MIN_TERM_BG = 3        # term must hit at least this many bg TFs to be testable

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
        rows.append({
            "term":   term,
            "go_id":  meta.get("exactSource", ""),
            "genes":  set(meta.get("geneSymbols", [])),
        })
    df = pd.DataFrame(rows)
    return df


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


def enrich_cluster(fg_set: set, bg_set: set, terms_df: pd.DataFrame,
                   min_term_bg: int = 3) -> pd.DataFrame:
    """Hypergeometric enrichment of fg vs bg using terms in terms_df."""
    N = len(bg_set)
    n = len(fg_set & bg_set)        # fg restricted to bg universe
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
        # one-sided upper-tail p-value: P(X >= k)
        p = hypergeom.sf(k - 1, N, K, n)
        # odds ratio with +0.5 continuity (Haldane–Anscombe)
        a = k + 0.5
        b = (n - k) + 0.5
        c = (K - k) + 0.5
        d = ((N - K) - (n - k)) + 0.5
        odds = (a / b) / (c / d)
        rows.append({
            "term":     t["term"],
            "go_id":    t["go_id"],
            "set_size_in_bg": K,
            "fg_in":    k,
            "fg_total": n,
            "bg_in":    K,
            "bg_total": N,
            "odds_ratio": odds,
            "p_value":  p,
            "genes_in_overlap": ",".join(sorted(overlap)),
        })
    if not rows:
        return pd.DataFrame(columns=["term","go_id","set_size_in_bg","fg_in",
                                     "fg_total","bg_in","bg_total","odds_ratio",
                                     "p_value","q_value","genes_in_overlap"])
    out = pd.DataFrame(rows)
    out["q_value"] = benjamini_hochberg(out["p_value"].to_numpy())
    out = out.sort_values(["q_value", "p_value"]).reset_index(drop=True)
    return out


def plot_dotplot(per_cluster_results: dict, out_path_stem: str,
                 top_n: int = 5, qval_cutoff: float = 0.05):
    """
    Build a long-format DataFrame of (cluster, term) -> (q, OR), keep terms
    that are top_n per cluster AND have q<qval_cutoff in at least one cluster,
    plot as size/color dotplot.
    """
    long_rows = []
    for c, df in per_cluster_results.items():
        if df.empty:
            continue
        sub = df.head(top_n)
        for _, r in sub.iterrows():
            long_rows.append({"cluster": c, "term": r["term"],
                              "q_value": r["q_value"],
                              "odds_ratio": r["odds_ratio"],
                              "fg_in": r["fg_in"]})
    if not long_rows:
        print(f"  no enriched terms to plot for {out_path_stem}")
        return
    long_df = pd.DataFrame(long_rows)

    # union of top-N terms across clusters; keep term if q<cutoff in any cluster
    sig_terms = set(long_df.loc[long_df["q_value"] < qval_cutoff, "term"])
    if not sig_terms:
        sig_terms = set(long_df["term"])
    long_df = long_df[long_df["term"].isin(sig_terms)]

    # Re-pull q & OR for ALL clusters for these terms (even those past their top_n)
    full_rows = []
    for c, df in per_cluster_results.items():
        if df.empty:
            continue
        sub = df[df["term"].isin(sig_terms)]
        for _, r in sub.iterrows():
            full_rows.append({"cluster": c, "term": r["term"],
                              "q_value": r["q_value"],
                              "odds_ratio": r["odds_ratio"],
                              "fg_in": r["fg_in"]})
    full = pd.DataFrame(full_rows)

    full["neg_log10_q"]   = -np.log10(full["q_value"].clip(lower=1e-15)).clip(upper=10)
    full["log2_or"]       = np.log2(full["odds_ratio"].clip(lower=1e-3))

    # Order terms: pick the cluster of best q for each term, then sort by cluster, then by best q
    term_best = full.loc[full.groupby("term")["q_value"].idxmin()][["term","cluster","q_value"]]
    term_best = term_best.sort_values(["cluster", "q_value"])
    term_order = term_best["term"].tolist()
    cluster_order = sorted(full["cluster"].unique())

    full["term"]    = pd.Categorical(full["term"],    categories=term_order,    ordered=True)
    full["cluster"] = pd.Categorical(full["cluster"], categories=cluster_order, ordered=True)

    height = max(4, 0.25 * len(term_order) + 1.5)
    width  = max(5, 0.5 * len(cluster_order) + 4)
    fig, ax = plt.subplots(figsize=(width, height))
    sc = ax.scatter(
        full["cluster"].cat.codes, full["term"].cat.codes,
        s=full["neg_log10_q"] * 30 + 5,
        c=full["log2_or"], cmap="RdBu_r",
        vmin=-max(abs(full["log2_or"].min()), abs(full["log2_or"].max()), 1.0),
        vmax= max(abs(full["log2_or"].min()), abs(full["log2_or"].max()), 1.0),
        edgecolors="black", linewidth=0.3,
    )
    ax.set_xticks(range(len(cluster_order)))
    ax.set_xticklabels([f"C{c}" for c in cluster_order])
    ax.set_yticks(range(len(term_order)))
    ax.set_yticklabels([t.replace("GOBP_", "").replace("_", " ").lower() for t in term_order],
                       fontsize=8)
    ax.set_xlabel("Cluster")
    ax.set_title(f"GO BP enrichment per cluster (top {top_n} per cluster, q<{qval_cutoff})")
    ax.invert_yaxis()
    ax.grid(True, linestyle=":", alpha=0.4)
    cb = fig.colorbar(sc, ax=ax, fraction=0.025, pad=0.02)
    cb.set_label("log2(odds ratio)")

    # size legend
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
    msigdb_universe = set().union(*terms_df["genes"])
    print(f"  {len(terms_df):,} GO BP terms; gene universe = {len(msigdb_universe):,}")

    ENRICH_DN.mkdir(exist_ok=True)
    PLOTS_DN.mkdir(exist_ok=True)

    for tname, tpath in TABLES.items():
        print(f"\n=== {tname} ({tpath.name}) ===")
        tab = pd.read_csv(tpath, sep="\t")
        tf_bg     = set(tab["TF"])           & msigdb_universe
        genome_bg = msigdb_universe           # all 18k+ MSigDB-annotated genes
        print(f"  table TFs: {len(set(tab['TF']))}; "
              f"|tf_bg|={len(tf_bg)}; |genome_bg|={len(genome_bg)}")

        for bg_name, bg_set in [("tf_bg", tf_bg), ("genome_bg", genome_bg)]:
            out_table_dn = ENRICH_DN / tname / bg_name
            out_table_dn.mkdir(parents=True, exist_ok=True)

            per_cluster = {}
            for c in sorted(tab["cluster"].unique()):
                fg_set = set(tab.loc[tab["cluster"] == c, "TF"]) & msigdb_universe
                res = enrich_cluster(fg_set, bg_set, terms_df, min_term_bg=MIN_TERM_BG)
                per_cluster[c] = res
                n_sig = int((res["q_value"] < QVAL_CUTOFF).sum()) if not res.empty else 0
                print(f"  [{bg_name}] cluster {c}: |fg|={len(fg_set):3d}  "
                      f"terms_tested={len(res):4d}  q<{QVAL_CUTOFF}: {n_sig}")
                res.to_csv(out_table_dn / f"cluster{c}.gobp.tsv", sep="\t", index=False)

            plot_stem = str(PLOTS_DN / f"enrichment_msigdb_gobp.{tname}.{bg_name}.dotplot")
            plot_dotplot(per_cluster, plot_stem, top_n=TOP_N_TERMS,
                         qval_cutoff=QVAL_CUTOFF)
            print(f"  [{bg_name}] dotplot -> {plot_stem}.png/.pdf")

    print("\nDONE")


if __name__ == "__main__":
    main()
