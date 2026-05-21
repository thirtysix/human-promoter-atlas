"""Per-TF explorer: aggregate profile, program loadings, top TSSs."""
from __future__ import annotations

import math

import pandas as pd
import streamlit as st

from app.lib import db, plotting, ui


HELP_TF = (
    "Pick a TF (chip-atlas) to see its profile across all canonical TSSs, "
    "which k=10 programs it participates in, its membership in the "
    "filtered-K8 / no-filter-K12 TF clustering, and where it binds most."
)
HELP_PROFILE = (
    "Mean binary coverage probability for this TF across ~19,700 canonical "
    "protein-coding TSSs. The peak position tells you where this TF "
    "characteristically binds in the average promoter."
)
HELP_LOADINGS = (
    "How heavily this TF loads on each of the 10 programs in the NMF H "
    "matrix (top-30-per-program subset). A TF can load on multiple programs "
    "if it participates in multiple regulatory contexts."
)
HELP_CLUSTERS = (
    "TF clusters were computed by hierarchical (Ward / Euclidean) clustering "
    "on each TF's peak-normalized aggregate profile shape — independent of "
    "the per-promoter NMF programs."
)
HELP_TOP_TSS = (
    "Canonical TSSs with the most chip-atlas peaks at score ≥ 500 within "
    "their ±1.5 kb window. `# peaks` counts recentered 25-nt blocks for this "
    "TF at this TSS — a TF can have multiple peaks at one promoter."
)


def render() -> None:
    ui.intro_card(
        title="Per-TF view — what does this transcription factor do?",
        what="Everything the atlas knows about one TF: aggregate binding "
             "profile, loading on each k=10 program, TF-cluster membership "
             "(K=8 / K=12 hierarchical), DepMap essentiality, GTEx "
             "expression, top co-binding partners, and top bound TSSs.",
        objective="Trace a TF's role through the regulatory hierarchy: "
                   "raw peaks → mean-of-promoters profile → NMF programs "
                   "→ co-binding partners → target genes.",
        significance="Many TFs participate in more than one program — CTCF "
                      "loads on two (cohesin-anchored at TSS *and* "
                      "chromatin-anchored downstream); EP300 spans several. "
                      "The program-loading bar makes this multi-role "
                      "behavior explicit.",
    )

    st.subheader("Per-TF view", help=HELP_TF)

    with st.container(border=True):
        tf = st.selectbox(
            "TF",
            options=[""] + db.list_tfs(),
            index=0,
            placeholder="Type to search…",
            key="tf_select",
            help="Filename stem in chip-atlas/per_TF/, intersected with "
                 "the curated DNA-binding gene list.",
        )

    if not tf:
        st.info("Pick a TF above to load its profile.")
        return

    meta = db.get_tf_meta(tf)
    if meta is None:
        st.error(f"TF '{tf}' not in the index.")
        return

    # ---- Quick stats card --------------------------------------------------
    with st.container(border=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Peaks (chip-atlas, recentered)",
                  f"{int(meta.get('n_peaks_kept') or 0):,}",
                  help="Peaks across all chip-atlas experiments aggregated "
                       "for this TF that fall on standard chromosomes after "
                       "25-nt recentering.")
        c2.metric("Bound TSSs (score≥500, ±100 bp core)",
                  f"{int(meta.get('n_bound_tss_core') or 0):,}",
                  help="Number of canonical TSSs this TF binds at the core "
                       "promoter (±100 bp) with at least one score≥500 peak.")
        c3.metric("Cluster (filtered K=8)",
                  f"C{int(meta['cluster_filtered'])}"
                  if meta.get("cluster_filtered") is not None
                     and not _is_nan(meta["cluster_filtered"])
                  else "—",
                  help=HELP_CLUSTERS)
        c4.metric("Cluster (no_filter K=12)",
                  f"C{int(meta['cluster_no_filter'])}"
                  if meta.get("cluster_no_filter") is not None
                     and not _is_nan(meta["cluster_no_filter"])
                  else "—",
                  help=HELP_CLUSTERS)

    # ---- Aggregate profile + program loadings (paired row) ----------------
    prof_col, load_col = st.columns([2, 1])
    with prof_col:
        with st.container(border=True):
            st.markdown("#### Aggregate binding profile", help=HELP_PROFILE)
            # All profile-related controls live with the plot.
            c_clu, c_show, c_cmp = st.columns([1, 1, 2])
            with c_clu:
                show_cluster_mean = st.checkbox(
                    "Overlay TF-cluster mean",
                    value=False,
                    help="Plot the mean ± SEM profile of all TFs in this "
                         "TF's filtered K=8 (or no_filter K=12) cluster — "
                         "see if this TF is a typical or atypical member "
                         "of its shape-cluster.",
                )
            with c_show:
                cluster_set = st.radio(
                    "Cluster set",
                    ["filtered K=8", "no_filter K=12"],
                    horizontal=True, index=0,
                    disabled=not show_cluster_mean,
                    help="Which hierarchical TF-shape clustering to draw "
                         "the cluster mean from. **filtered K=8** uses "
                         "only score≥500 peaks (8 clusters), "
                         "**no_filter K=12** uses all peaks (12 finer "
                         "clusters).",
                ) if show_cluster_mean else "filtered K=8"
            with c_cmp:
                compare = st.multiselect(
                    "Compare to (faded dotted lines)",
                    options=db.list_tfs(),
                    default=[],
                    help="Show these TFs as light comparison traces on "
                         "the profile plot below.",
                    key=f"tf_compare_{tf}",
                )

            cluster_members = []
            cluster_label = ""
            if show_cluster_mean:
                tag = ("filtered" if cluster_set.startswith("filtered")
                       else "no_filter")
                cid = (meta.get("cluster_filtered") if tag == "filtered"
                       else meta.get("cluster_no_filter"))
                if cid is not None and not _is_nan(cid):
                    cluster_members = db.get_cluster_members(tag, int(cid))
                    cluster_label = f"cluster {tag}-C{int(cid)}"

            matrix = db.load_aggregate_matrix("binary")
            if not matrix.empty and tf in matrix.index:
                st.plotly_chart(
                    plotting.fig_tf_aggregate_profile(
                        matrix, tf, list(compare),
                        cluster_members=cluster_members,
                        cluster_label=cluster_label,
                    ),
                    width="stretch",
                )
            else:
                st.warning(f"Aggregate profile for {tf} not available.")

    with load_col:
        with st.container(border=True):
            st.markdown("#### Loading on each k=10 program",
                         help=HELP_LOADINGS)
            loadings = db.get_tf_program_loadings(tf)
            if loadings.empty:
                st.info(f"{tf} is not in the top-30 of any k=10 program "
                        f"(its loadings across programs are below that "
                        f"cutoff).")
            else:
                st.plotly_chart(
                    plotting.fig_tf_program_loadings(loadings, tf),
                    width="stretch",
                )

    # ---- DepMap essentiality ----------------------------------------------
    if db.depmap_available():
        with st.container(border=True):
            st.markdown(
                "#### DepMap CRISPR essentiality",
                help="Median Chronos gene-effect score per OncotreeLineage "
                     "across DepMap CRISPR screens (1,184 cancer cell lines, "
                     "30 lineages). More-negative = more essential — knockout "
                     "of this TF reduces fitness in that lineage. The dashed "
                     "red line marks the conventional essentiality threshold "
                     "(Chronos = −1).",
            )
            ess_df = db.depmap_gene_lineage(tf)
            sumr = db.depmap_gene_summary((tf,))
            if sumr.empty:
                st.info(f"{tf} not in DepMap CRISPR screen.")
            else:
                row = sumr.iloc[0]
                c1, c2, c3 = st.columns(3)
                c1.metric("Median Chronos (all lines)",
                           f"{row['median_chronos']:+.2f}",
                           help="More negative = more essential. "
                                "Pan-essential genes have median ≈ −1 to −2.")
                c2.metric("Fraction essential (<−1)",
                           f"{row['frac_essential']:.0%}",
                           help="Fraction of cell lines where this gene's "
                                "knockout drops fitness past the −1 threshold.")
                c3.metric("Most essential lineage",
                           str(row['most_essential_lineage']) if pd.notna(row['most_essential_lineage']) else "—",
                           help=f"Lineage with the most-negative median Chronos "
                                f"for this gene "
                                f"({row['most_essential_chronos']:+.2f}).")
                st.plotly_chart(plotting.fig_depmap_lineage_bar(ess_df, tf),
                                width="stretch")

    # ---- GTEx expression --------------------------------------------------
    if db.gtex_available():
        with st.container(border=True):
            st.markdown("#### GTEx tissue expression",
                         help="Max TPM across this TF gene's transcripts "
                              "per tissue. The tissue with the highest "
                              "value is where the TF is most active.")
            tf_expr = db.gtex_tf_expression(tf)
            if tf_expr.empty:
                st.info(f"{tf} not in the GTEx tissue-expression matrix.")
            else:
                expr_stats = tf_expr.copy()
                expr_stats.columns = ["tissue", "mean"]
                expr_stats["median"] = expr_stats["mean"]
                expr_stats["q1"] = expr_stats["mean"]
                expr_stats["q3"] = expr_stats["mean"]
                expr_stats["std"] = 0.0
                expr_stats["n_samples"] = 0
                st.plotly_chart(
                    plotting.fig_gtex_expression_bar(
                        expr_stats, title_prefix=f"{tf} — "),
                    width="stretch",
                )

    # ---- Co-binding partners ----------------------------------------------
    with st.container(border=True):
        st.markdown(
            "#### Top co-binding partner TFs",
            help=(
                "Ranks every other TF by the number of regulatory modules "
                "in which it co-occurs with the focal TF. Both directional "
                "percentages are shown on hover:\n\n"
                "**`pct_of_partner_modules`** = what fraction of the "
                "partner's modules also contain the focal TF. High values "
                "(e.g. RAD21 → 87% when focal=CTCF) name obligate "
                "partners.\n\n"
                "**`pct_of_focal_modules`** = what fraction of the focal "
                "TF's modules also contain the partner. Tells you how "
                "broadly the partner reaches across the focal's footprint."
            ),
        )
        c1, c2 = st.columns([1, 5])
        with c1:
            n_part = st.slider(
                "Top N partners", 10, 50, 20, step=5,
                help="Sorted by # shared modules (descending).",
                key=f"tf_cobind_n_{tf}",
            )
        partners = db.get_tf_cobinding_partners(tf, limit=int(n_part))
        if partners.empty:
            st.info(f"No co-binding partners found for {tf} "
                    "(check that this TF appears in module_tf evidence).")
        else:
            with c2:
                st.plotly_chart(
                    plotting.fig_tf_cobinding_partners(partners, tf),
                    width="stretch",
                )
            with st.expander("Co-binding partners table", expanded=False):
                st.dataframe(
                    partners, hide_index=True, width="stretch",
                    column_config={
                        "partner":                st.column_config.TextColumn(
                            "partner TF",
                            help="Other TF that shares modules with the "
                                 "focal TF."),
                        "n_shared":               st.column_config.NumberColumn(
                            "# shared modules",
                            help="# modules in the atlas in which BOTH "
                                 f"the partner AND {tf} appear (from the "
                                 "module-TF evidence table)."),
                        "partner_total":          st.column_config.NumberColumn(
                            "# partner's modules",
                            help="Total atlas-wide modules containing "
                                 "the partner TF."),
                        "focal_total":            st.column_config.NumberColumn(
                            f"# {tf}'s modules",
                            help=f"Total atlas-wide modules containing "
                                 f"{tf}."),
                        "pct_of_partner_modules": st.column_config.NumberColumn(
                            "% of partner", format="%.1f",
                            help=f"What fraction of the partner's modules "
                                 f"ALSO contain {tf}. High = obligate "
                                 f"partner of {tf}."),
                        "pct_of_focal_modules":   st.column_config.NumberColumn(
                            f"% of {tf}", format="%.1f",
                            help=f"What fraction of {tf}'s modules ALSO "
                                 "contain the partner. High = partner "
                                 f"reaches broadly across {tf}'s "
                                 "footprint."),
                        "jaccard":                st.column_config.NumberColumn(
                            "Jaccard", format="%.3f",
                            help="Symmetric overlap = n_shared / "
                                 "(n_focal + n_partner − n_shared). "
                                 "1.0 = identical module sets, "
                                 "0 = disjoint."),
                    },
                )
                st.download_button(
                    f"{tf} co-binding partners — TSV",
                    data=partners.to_csv(sep="\t", index=False).encode(),
                    file_name=f"{tf}_cobinding_partners.tsv",
                    mime="text/tab-separated-values",
                    key=f"tf_cobind_dl_{tf}",
                )

    # ---- Top TSSs ----------------------------------------------------------
    with st.container(border=True):
        st.markdown("### Top TSSs bound by this TF (score ≥ 500)",
                     help=HELP_TOP_TSS)
        n_top = st.slider("# of top TSSs", 25, 500, 100, step=25,
                           help="How many of the most-bound canonical TSSs "
                                "to show.")
        top = db.get_top_tss_for_tf(tf, limit=n_top)
        if top.empty:
            st.info("No score≥500 peaks for this TF in the canonical-TSS "
                    "windows.")
        else:
            st.dataframe(
                top, hide_index=True, width="stretch",
                column_config={
                    "transcript_id": st.column_config.TextColumn(
                        "transcript_id",
                        help="Ensembl transcript ID of the bound canonical TSS."),
                    "gene_name": st.column_config.TextColumn(
                        "gene",
                        help="Gene symbol of that transcript."),
                    "n_peaks_500": st.column_config.NumberColumn(
                        "# peaks (score≥500)",
                        help="Number of recentered 25-nt peaks for this "
                             "TF at this TSS, score ≥ 500."),
                    "min_offset":  st.column_config.NumberColumn(
                        "earliest bp",
                        help="Most upstream peak position (txn-oriented "
                             "bp from TSS) for this TF at this TSS."),
                    "max_offset":  st.column_config.NumberColumn(
                        "latest bp",
                        help="Most downstream peak position (txn-oriented "
                             "bp from TSS) for this TF at this TSS."),
                },
            )
            st.download_button(
                f"Top {len(top)} TSSs for {tf} (TSV)",
                data=top.to_csv(sep="\t", index=False).encode(),
                file_name=f"{tf}_top_tss.tsv",
                mime="text/tab-separated-values",
            )


def _is_nan(x) -> bool:
    return isinstance(x, float) and math.isnan(x)
