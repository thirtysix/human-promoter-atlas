"""Archetypes tab: gene-level NMF on the program-presence vector. Each gene
gets one of A=8 dominant archetypes; each archetype has a program signature
and its own GO BP enrichment."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from app.lib import db, plotting, ui


def render() -> None:
    ui.intro_card(
        title="Archetypes — gene-level summary of promoter composition",
        what="Each gene is a 10-vector counting how many of its modules "
             "belong to each k=10 program. NMF on the "
             "[18,055 genes × 10] matrix produces **A=8 archetypes** — "
             "the gene-level analog of programs.",
        objective="Answer 'what kind of promoter does this gene have?' "
                   "with a single label, not a list of modules.",
        significance="The natural endpoint of the analysis: "
                      "TFs → modules → programs → **archetypes**. They "
                      "surface biology the per-program view misses — e.g. "
                      "A6 (P5 cohesin-dominated) lights up homophilic "
                      "cell-cell adhesion at OR=7.5 (the protocadherin / "
                      "cohesin-anchored signal).",
    )

    arches = db.get_archetypes()

    # Merge per-archetype tau (computed at runtime from GTEx + gene_archetypes)
    if db.gtex_available():
        arch_tau = db.gtex_archetype_tissue_specificity()
        if not arch_tau.empty:
            arches = arches.merge(
                arch_tau[["archetype", "tau", "top_tissue",
                          "top_tissue_mean_tpm"]],
                on="archetype", how="left",
            )

    # ---- Archetype summary ------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "### Archetype summary",
            help="One row per archetype. `top programs` = top 5 k=10 "
                 "programs by H loading; `top loadings` are the H values "
                 "in the same order. `mean_modules_per_gene` summarizes "
                 "promoter complexity within the archetype. `tau` is the "
                 "Yanai 2005 tissue-specificity index (0 = broadly "
                 "expressed across tissues, 1 = single-tissue) computed "
                 "over the mean GTEx TPM of the archetype's member "
                 "transcripts; `top tissue` is the GTEx tissue where "
                 "those transcripts express most strongly.",
        )
        cols = ["archetype", "n_genes", "frac_genes", "mean_modules_per_gene",
                "top_programs", "top_loadings"]
        if "tau" in arches.columns:
            cols += ["tau", "top_tissue", "top_tissue_mean_tpm"]
        st.dataframe(
            arches[cols],
            column_config={
                "archetype":             st.column_config.TextColumn(
                    "A",
                    help="Archetype id — a coarse cluster over genes "
                         "based on their full vector of program loadings."),
                "n_genes":               st.column_config.NumberColumn(
                    "# genes",
                    help="# canonical-promoter genes assigned to this "
                         "archetype (dominant_archetype = this one)."),
                "frac_genes":            st.column_config.NumberColumn(
                    "frac. of genome", format="%.2f",
                    help="n_genes / total canonical-promoter genes."),
                "mean_modules_per_gene": st.column_config.NumberColumn(
                    "mean modules / gene", format="%.2f",
                    help="Average # NMF-discovered modules per gene "
                         "within this archetype. Higher = more "
                         "regulatory complexity."),
                "top_programs":          st.column_config.TextColumn(
                    "top programs",
                    help="The top-loading programs that DEFINE this "
                         "archetype (comma-joined). These are the "
                         "programs that genes in this archetype tend "
                         "to be enriched for."),
                "top_loadings":          st.column_config.TextColumn(
                    "top loadings (H)",
                    help="NMF H-coefficients for the top programs."),
                "tau":                   st.column_config.NumberColumn(
                    "tau (tissue specificity)",
                    format="%.3f",
                    help="Yanai 2005 tau over mean GTEx TPM of the "
                         "archetype's member transcripts. Tissues with "
                         "<40 donor samples excluded "
                         "(MIN_SAMPLES_FOR_TAU=40)."),
                "top_tissue":            st.column_config.TextColumn(
                    "top tissue",
                    help="GTEx tissue where the archetype's genes "
                         "show peak mean TPM."),
                "top_tissue_mean_tpm":   st.column_config.NumberColumn(
                    "top tissue mean TPM", format="%.1f",
                    help="Mean TPM of the archetype's transcripts in "
                         "its top tissue."),
            },
            hide_index=True, width="stretch",
        )

    # ---- Archetype × program H heatmap -----------------------------------
    with st.container(border=True):
        st.markdown(
            "### Archetype × program loadings (H)",
            help="The NMF H matrix at the gene level. Each cell is how "
                 "much program p contributes to archetype A. Reading this "
                 "matrix tells you what programs each archetype is "
                 "made of.",
        )
        ap = db.get_archetype_program_loadings()
        wide = ap.pivot(index="archetype", columns="program", values="loading")
        # Ensure full 1..10 program order
        wide = wide.reindex(columns=range(1, 11), fill_value=0.0)
        z = wide.values
        fig = go.Figure(go.Heatmap(
            z=z,
            x=[f"P{p}" for p in wide.columns],
            y=[f"A{a}" for a in wide.index],
            colorscale="magma", zmin=0,
            zmax=float(np.quantile(z, 0.99) or 1.0),
            colorbar=dict(title="H loading", thickness=12),
            hovertemplate="archetype: %{y}<br>program: %{x}<br>"
                          "loading: %{z:.3f}<extra></extra>",
        ))
        # Numeric overlay
        for i, a in enumerate(wide.index):
            for j, p in enumerate(wide.columns):
                v = z[i, j]
                if v > 0.05:
                    fig.add_annotation(
                        x=f"P{p}", y=f"A{a}", text=f"{v:.2f}",
                        showarrow=False, font=dict(
                            size=10,
                            color="white" if v > z.mean() else "black",
                        ),
                    )
        fig.update_layout(
            title=f"Archetype × program loadings (A={len(wide)}, "
                  f"k={len(wide.columns)})",
            margin=dict(l=60, r=60, t=60, b=50),
            height=80 + 32 * len(wide), width=800,
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, width="content")

    # ---- Drill-down --------------------------------------------------------
    st.markdown("### Drill into an archetype")
    sel = st.selectbox(
        "Archetype",
        options=arches["archetype"].tolist(),
        format_func=lambda a: (
            f"A{a} — top programs: "
            f"{arches.loc[arches['archetype']==a, 'top_programs'].iloc[0]}"
            f"  (n={int(arches.loc[arches['archetype']==a, 'n_genes'].iloc[0]):,})"
        ),
        index=0,
        help="Pick an archetype to see its top GO BP terms and a sample "
             "of the genes it labels.",
    )

    # Tissue-specificity chip — colored bar + three metric cards to match
    # the pattern on the Per-transcript tab.
    if "tau" in arches.columns:
        row = arches.loc[arches["archetype"] == int(sel)].iloc[0]
        if pd.notna(row.get("tau")) and row.get("top_tissue"):
            st.markdown(
                f"<div style='height:6px;background:"
                f"{plotting.PROGRAM_COLORS[(int(sel) - 1) % 10]};"
                f"border-radius:3px;margin:6px 0 8px 0'></div>",
                unsafe_allow_html=True,
            )
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Genes", f"{int(row['n_genes']):,}",
                       help="# canonical-promoter genes whose dominant "
                            "archetype is this one.")
            m2.metric("τ (tissue specificity)", f"{row['tau']:.3f}",
                       help="Yanai 2005 index over the archetype's mean "
                            "GTEx TPM. 0 = uniform across tissues, 1 = "
                            "single-tissue. Tissues with <40 donors "
                            "excluded (MIN_SAMPLES_FOR_TAU = 40).")
            m3.metric("Top tissue", str(row["top_tissue"]),
                       help="GTEx tissue where this archetype's member "
                            "transcripts express most strongly.")
            m4.metric("Top tissue TPM",
                       f"{row['top_tissue_mean_tpm']:.1f}",
                       help="Mean TPM of the archetype's transcripts in "
                            "the top tissue.")

    col_go, col_genes = st.columns([3, 2])
    with col_go:
        with st.container(border=True):
            st.markdown("#### Top GO BP terms (genome bg)",
                         help="Hypergeometric enrichment for the gene set "
                              "of this archetype against the MSigDB "
                              "c5.go.bp universe. q < 0.05 = significant.")
            go_df = db.get_archetype_top_go(int(sel), limit=15)
            if go_df.empty:
                st.info("No GO terms for this archetype.")
            else:
                go_df["term"] = (go_df["term"]
                                  .str.replace("GOBP_", "")
                                  .str.replace("_", " ").str.lower())
                st.dataframe(
                    go_df[["rank", "term", "fg_in", "set_size_in_bg",
                           "odds_ratio", "q_value"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "rank":  st.column_config.NumberColumn(
                            "rank", help="Rank by q-value within this "
                                         "archetype."),
                        "term":  st.column_config.TextColumn(
                            "term", help="GO BP term (prettified)."),
                        "fg_in": st.column_config.NumberColumn(
                            "fg", help="# archetype genes annotated to "
                                       "this GO term."),
                        "set_size_in_bg": st.column_config.NumberColumn(
                            "term size",
                            help="Total # MSigDB genes annotated to this "
                                 "GO term."),
                        "odds_ratio": st.column_config.NumberColumn(
                            "OR", format="%.2f",
                            help="Odds ratio of archetype-gene set vs "
                                 "MSigDB background."),
                        "q_value":    st.column_config.NumberColumn(
                            "q", format="%.1e",
                            help="Hypergeometric p adjusted by BH-FDR."),
                    },
                )
    with col_genes:
        with st.container(border=True):
            st.markdown(
                f"#### Top genes in A{int(sel)}",
                help="Genes whose dominant archetype is this one, sorted by "
                     "the dominant-archetype weight (how cleanly the gene "
                     "fits this archetype).",
            )
            top = db.get_genes_in_archetype(int(sel), limit=100)
            st.dataframe(
                top[["gene_name", "transcript_id", "n_modules",
                     "dominant_weight"]],
                hide_index=True, width="stretch",
                column_config={
                    "gene_name": st.column_config.TextColumn(
                        "gene",
                        help="Gene symbol."),
                    "transcript_id": st.column_config.TextColumn(
                        "transcript_id",
                        help="Ensembl ID of the gene's canonical transcript."),
                    "n_modules": st.column_config.NumberColumn(
                        "modules",
                        help="# NMF-discovered modules at this gene's "
                             "canonical promoter."),
                    "dominant_weight": st.column_config.NumberColumn(
                        "weight", format="%.3f",
                        help="Dominant-archetype weight — how cleanly "
                             "this gene fits the archetype. 1.0 = pure."),
                },
            )
            st.download_button(
                f"Top {len(top)} genes for A{int(sel)} (TSV)",
                data=top.to_csv(sep="\t", index=False).encode(),
                file_name=f"A{int(sel)}_top_genes.tsv",
                mime="text/tab-separated-values",
            )
