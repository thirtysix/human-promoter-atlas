#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Build a single multi-tab Excel summary file covering the entire
canonical-promoter analysis. Each tab is one logical table.

Tabs (in order):
    00_README                 — text overview of what's in the workbook
    01_TF_clusters_filtered   — K=8 cluster assignments (filtered TF set)
    02_TF_clusters_nofilter   — K=12 cluster assignments (no_filter TF set)
    03_TFcluster_GO_filtered  — top-5 GO BP terms per filtered TF cluster
    04_TSS_programs_k8        — single-window (±100 bp) k=8 summary
    05_modules_k_selection    — ARI / cophenetic / scree across k
    06_modules_k10            — canonical k=10 program summary (algorithmic k)
    07_modules_k10_GO         — top-10 GO BP terms per k=10 program
    08_modules_k8             — alt k=8 program summary (modules)
    09_modules_k12            — alt k=12
    10_modules_k15            — alt k=15
    11_modules_k20            — alt k=20
    12_top_gene_configs_k10   — most frequent program_paths (gene archetypes)
"""

################################################################################
# Libraries ####################################################################
################################################################################
import json
from pathlib import Path

import numpy as np
import pandas as pd

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN, K_CANONICAL


################################################################################
# Paths ########################################################################
################################################################################
ROOT                = OUT_DN   # config.OUT_DN
DOCS = ROOT / "docs"
OUT  = DOCS / "summary_tables.xlsx"

CLUSTERING_FILT     = ROOT / "clustering"           / "tf_cluster_table.tsv"
CLUSTERING_NOFILT   = ROOT / "clustering_no_filter" / "tf_cluster_table.tsv"
TFCLUSTER_GO_FILT_DN = ROOT / "enrichment_msigdb_gobp" / "filtered" / "genome_bg"
PROGRAMS_K8_SUM     = ROOT / "tss_programs"         / "nmf.k8.program_summary.tsv"
MODULES_DN          = ROOT / "tss_modules"
MODULE_GO_DN        = ROOT / "enrichment_msigdb_gobp_modules"
MODULES_KSEL_DN     = MODULES_DN / "k_selection"


################################################################################
# Loaders ######################################################################
################################################################################
def load_tf_cluster_table(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    return pd.read_csv(path, sep="\t")


def load_tfcluster_top_go(go_dn: Path, top_n: int = 5) -> pd.DataFrame:
    """Read every clusterX.gobp.tsv in go_dn, keep top_n by q_value per cluster."""
    if not go_dn.exists():
        return pd.DataFrame()
    out_rows = []
    for fp in sorted(go_dn.glob("cluster*.gobp.tsv")):
        cluster_id = int(fp.stem.replace("cluster", "").replace(".gobp", ""))
        df = pd.read_csv(fp, sep="\t")
        if df.empty:
            continue
        df = df.sort_values("q_value").head(top_n)
        for rank, (_, r) in enumerate(df.iterrows(), 1):
            out_rows.append({
                "cluster":       cluster_id,
                "rank":          rank,
                "term":          r["term"].replace("GOBP_", "").replace("_", " ").lower(),
                "go_id":         r.get("go_id", ""),
                "fg_in":         int(r["fg_in"]),
                "set_size_in_bg": int(r["set_size_in_bg"]),
                "odds_ratio":    round(float(r["odds_ratio"]), 2),
                "q_value":       float(r["q_value"]),
            })
    return pd.DataFrame(out_rows)


def load_tss_programs_summary(path: Path) -> pd.DataFrame:
    """tss_programs.001.py emits program_summary.tsv with top_genes; clean it."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, sep="\t")
    keep = ["program", "n_promoters_dominant", "mean_loading_when_dominant",
            "top_tfs", "top_genes"]
    df = df[[c for c in keep if c in df.columns]]
    df["mean_loading_when_dominant"] = df["mean_loading_when_dominant"].round(4)
    # Truncate top_genes for readability
    if "top_genes" in df.columns:
        df["top_genes"] = df["top_genes"].astype(str).str.slice(0, 250)
    return df


def load_modules_summary(k: int) -> pd.DataFrame:
    fn = MODULES_DN / f"nmf.k{k}.summary.tsv"
    if not fn.exists():
        return pd.DataFrame()
    return pd.read_csv(fn, sep="\t")


def load_k_selection() -> pd.DataFrame:
    """Combine ari_stability + cophenetic + scree into one wide table."""
    rows = {}
    ari_fn = MODULES_KSEL_DN / "ari_stability.tsv"
    if ari_fn.exists():
        ari = pd.read_csv(ari_fn, sep="\t").set_index("k")
        for k, r in ari.iterrows():
            rows.setdefault(int(k), {}).update({
                "median_ARI":  round(r["median_ari"],  3),
                "p25_ARI":     round(r["p25"],         3),
                "p75_ARI":     round(r["p75"],         3),
            })
    coph_fn = MODULES_KSEL_DN / "cophenetic.tsv"
    if coph_fn.exists():
        coph = pd.read_csv(coph_fn, sep="\t").set_index("k")
        for k, r in coph.iterrows():
            rows.setdefault(int(k), {}).update({
                "cophenetic":  round(r["cophenetic"], 4),
                "dispersion":  round(r["dispersion"], 4),
            })
    scree_fn = MODULES_KSEL_DN / "scree.tsv"
    if scree_fn.exists():
        scree = pd.read_csv(scree_fn, sep="\t").set_index("k")
        for k, r in scree.iterrows():
            rows.setdefault(int(k), {}).update({
                "mean_err":   round(r["mean_err"], 2) if pd.notna(r["mean_err"]) else np.nan,
                "std_err":    round(r["std_err"],  2) if pd.notna(r["std_err"])  else np.nan,
            })
    if not rows:
        return pd.DataFrame()
    out = pd.DataFrame.from_dict(rows, orient="index").reset_index()
    out = out.rename(columns={"index": "k"}).sort_values("k")
    out["preferred"] = out["median_ARI"] == out["median_ARI"].max()
    return out


def load_modules_k_top_go(k: int, top_n: int = 10) -> pd.DataFrame:
    """Keep top_n GO terms per program at given k. Drop the long
    genes_in_overlap column for readability — it's preserved in the per-program
    .tsv files."""
    fn = MODULE_GO_DN / f"k{k}" / "program_top_terms.tsv"
    if not fn.exists():
        return pd.DataFrame()
    df = pd.read_csv(fn, sep="\t")
    df = df[df["rank"] <= top_n].copy()
    df["term"] = df["term"].str.replace("GOBP_", "").str.replace("_", " ").str.lower()
    if "genes_in_overlap" in df.columns:
        df = df.drop(columns=["genes_in_overlap"])
    df["odds_ratio"] = df["odds_ratio"].round(2)
    df["p_value"] = df["p_value"].apply(lambda x: f"{x:.2e}")
    df["q_value"] = df["q_value"].apply(lambda x: f"{x:.2e}")
    return df


def load_top_gene_configs(k: int = K_CANONICAL, top_n: int = 30) -> pd.DataFrame:
    fn = MODULES_DN / f"nmf.k{k}.gene_configurations.tsv"
    if not fn.exists():
        return pd.DataFrame()
    gc = pd.read_csv(fn, sep="\t")
    grp = (gc.groupby(["n_modules", "program_path"])
             .agg(n_genes=("gene_name", "size"),
                  example_genes=("gene_name",
                                 lambda s: ",".join(sorted(s)[:10])))
             .reset_index()
             .sort_values("n_genes", ascending=False)
             .head(top_n))
    return grp


def make_readme() -> pd.DataFrame:
    lines = [
        ("Sheet", "Contents"),
        ("00_README",                "This sheet."),
        ("01_TF_clusters_filtered",  "Filtered K=8 TF cluster assignments (184 TFs)."),
        ("02_TF_clusters_nofilter",  "No-filter K=12 TF cluster assignments (1207 TFs)."),
        ("03_TFcluster_GO_filtered", "Top-5 GO BP terms per filtered TF cluster (genome_bg)."),
        ("04_TSS_programs_k8",       "Single-window (±100 bp) NMF k=8 program summary, top TFs and top genes."),
        ("05_modules_k_selection",   "Algorithmic k selection: median ARI, Brunet cophenetic ρ, dispersion, mean Frobenius error per k. 'preferred' marks the k with peak stability."),
        (f"06_modules_k{K_CANONICAL}", f"Canonical k={K_CANONICAL} program summary (algorithmically selected): n_modules, median position, median width, mean dominant weight, top TFs, auto reading."),
        (f"07_modules_k{K_CANONICAL}_GO", f"Top-10 GO BP terms per k={K_CANONICAL} program. genome_bg = MSigDB universe."),
        ("08_modules_k8",            "Alt k=8 program summary for comparison."),
        ("09_modules_k12",           "Alt k=12 program summary."),
        ("10_modules_k15",           "Alt k=15 program summary."),
        ("11_modules_k20",           "Alt k=20 program summary."),
        (f"12_top_gene_configs_k{K_CANONICAL}", "Most frequent gene configurations: program_path = ordered list of dominant programs across a gene's modules. Top 30 by gene count."),
        ("",                         ""),
        ("Notes",                    ""),
        ("",                         "Per-program GO term files with full genes_in_overlap lists are at enrichment_msigdb_gobp_modules/k{K}/program{P}.gobp.tsv"),
        ("",                         "Per-gene module assignments with all program weights are at tss_modules/nmf.k{K}.module_program.tsv"),
        ("",                         "Per-TF cluster GO term files are at enrichment_msigdb_gobp/{filtered,no_filter}/{tf_bg,genome_bg}/cluster{C}.gobp.tsv"),
        ("",                         ""),
        ("Pipeline scripts",         ""),
        ("canonical_promoter_aggregate.001.py", "1: BED -> TF×position matrices."),
        ("cluster_tfs.001.py",       "2: hierarchical clustering of TFs by aggregate profile shape (filtered, K=8)."),
        ("cluster_tfs.no_filter.001.py", "2alt: same with no signal filter (K=12)."),
        ("enrich_clusters_msigdb.001.py", "3: GO BP enrichment per TF cluster."),
        ("tss_programs.001.py",      "4: per-TSS NMF, single window."),
        ("tss_programs_multibin.001.py", "4alt: per-TSS NMF with positional bins."),
        ("tss_modules.001.py",       "5: per-gene KDE -> regulatory modules -> NMF on module×TF."),
        ("tss_modules_select_k.001.py", "6: ARI + Brunet cophenetic for algorithmic k selection."),
        ("tss_modules_k10.py",       "7: canonical k=10 NMF run + tables + plots."),
        ("enrich_tss_modules_msigdb.001.py", "8: GO BP enrichment per program at each k."),
        ("build_summary_tables.001.py", "9: this workbook."),
    ]
    return pd.DataFrame(lines[1:], columns=lines[0])


################################################################################
# Builder ######################################################################
################################################################################
def main():
    DOCS.mkdir(parents=True, exist_ok=True)
    print(f"writing {OUT}")
    with pd.ExcelWriter(OUT, engine="openpyxl") as xw:
        sheets = {
            "00_README":               make_readme(),
            "01_TF_clusters_filtered": load_tf_cluster_table(CLUSTERING_FILT),
            "02_TF_clusters_nofilter": load_tf_cluster_table(CLUSTERING_NOFILT),
            "03_TFcluster_GO_filtered": load_tfcluster_top_go(TFCLUSTER_GO_FILT_DN, top_n=5),
            "04_TSS_programs_k8":      load_tss_programs_summary(PROGRAMS_K8_SUM),
            "05_modules_k_selection":  load_k_selection(),
            f"06_modules_k{K_CANONICAL}":    load_modules_summary(K_CANONICAL),
            f"07_modules_k{K_CANONICAL}_GO": load_modules_k_top_go(K_CANONICAL, top_n=10),
            "08_modules_k8":           load_modules_summary(8),
            "09_modules_k12":          load_modules_summary(12),
            "10_modules_k15":          load_modules_summary(15),
            "11_modules_k20":          load_modules_summary(20),
            f"12_top_gene_configs_k{K_CANONICAL}": load_top_gene_configs(K_CANONICAL, top_n=30),
        }
        for name, df in sheets.items():
            if df is None or df.empty:
                print(f"  [skip] {name}  (empty)")
                continue
            df.to_excel(xw, sheet_name=name, index=False)
            print(f"  wrote {name}: {df.shape[0]:>4d} rows × {df.shape[1]} cols")

        # Auto-fit column widths (best-effort)
        for sh_name, df in sheets.items():
            if df is None or df.empty:
                continue
            ws = xw.book[sh_name]
            for col_idx, col in enumerate(df.columns, 1):
                max_len = max(
                    len(str(col)),
                    df[col].astype(str).map(len).max() if len(df) else 0,
                )
                # Cap at 60 to keep headers readable
                ws.column_dimensions[ws.cell(row=1, column=col_idx).column_letter].width = min(max_len + 2, 60)
            # Freeze header row
            ws.freeze_panes = ws["A2"]

    print("\nDONE")


if __name__ == "__main__":
    main()
