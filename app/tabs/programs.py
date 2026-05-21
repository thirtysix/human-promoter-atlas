"""Programs & Modules tab: browse the 10 NMF programs."""
from __future__ import annotations

import streamlit as st

from app.lib import db, nav, plotting, ui


HELP_PROGRAMS = (
    "A 'program' is a recurring TF-by-position archetype across human "
    "promoters, discovered by NMF on the [n_module × n_TF] occupancy "
    "matrix. k=10 was chosen algorithmically by ARI stability + Brunet "
    "cophenetic correlation. Each program has a TF signature and a "
    "characteristic position relative to the TSS."
)
HELP_TOP_TFS = (
    "TFs ranked by H loading — the program's per-TF coefficient in the "
    "NMF decomposition. Higher = the TF is more characteristic of this "
    "program. Showing top 30."
)
HELP_POS_DENSITY = (
    "Histogram of where each module of this program is centered "
    "(transcription-direction-oriented). A program is 'core' if centered "
    "near 0, 'upstream' if negative, 'downstream' if positive."
)
HELP_TOP_GO = (
    "MSigDB c5.go.bp BP enrichment for the gene set associated with this "
    "program (genes whose canonical TSS has ≥1 module dominantly "
    "assigned). Hypergeometric p, BH-FDR; background = MSigDB universe."
)
HELP_DRIVER_CLASS = (
    "Each module is classified by how many of its TFs co-vary with the "
    "target gene's expression across the 66 GTEx tissues "
    "(|r| ≥ 0.5):  \n"
    "**no-driver** = 0 strong TFs (constitutive / tissue-invariant);  \n"
    "**single-driver** = 1 strong TF (clean single-factor regulation);  \n"
    "**multi-driver** = ≥2 strong TFs (combinatorial or redundant).  \n\n"
    "Programs that read as 'mostly multi-driver' look enhancer-like; "
    "programs that read as 'mostly no-driver' look constitutive."
)
HELP_TF_TISSUE_HEATMAP = (
    "Per-TF expression (GTEx, max-of-transcripts mean TPM) for this "
    "program's top TFs across 66 tissues. Color = log10(TPM+1). Rows are "
    "ordered by H loading (most characteristic TF on top); tissues are "
    "ordered by aggregate program signal so the program's 'hot' tissues "
    "cluster on the left. This is the *direct* TF×tissue view — "
    "complementary to the module-aggregated activity shown elsewhere."
)
HELP_PROG_SUMMARY = (
    "One row per program. `n modules` is the count of modules whose "
    "dominant program is this one (out of 76,999 total). `median bp from "
    "TSS` and `median width` summarize the position panel."
)


def render() -> None:
    ui.intro_card(
        title="Programs and modules — recurring promoter archetypes",
        what="A **module** is a local cluster of TF binding within a "
             "single promoter (±1.5 kb of its TSS). A **program** is one "
             "of 10 archetypal modules — discovered by NMF on the "
             "[77,000 modules × 1,304 TFs] occupancy matrix — each with a "
             "TF signature and a characteristic position relative to the TSS.",
        objective="Find the small number of *reusable* regulatory units "
                   "from which human promoters are built.",
        significance="The 10 programs cover ~93% of canonical TSSs with "
                      "chip-atlas evidence. Each lights up a clean "
                      "biological signal: GABPA/THAP11 → ribosome "
                      "biogenesis (OR=5.4); RUNX/FLI/SPI → immune system; "
                      "CTCF splits into two distinct programs at two "
                      "distinct positions. This is the *parts list* for "
                      "the human promoter.",
    )

    st.subheader("Programs and modules (k=10)", help=HELP_PROGRAMS)

    progs = db.get_programs()

    # ---- Drill into a program (PRIMARY ACTION — moved above summary) ------
    st.markdown("### Drill into a program")
    options = progs["program"].tolist()
    preselected = st.session_state.get("preselected_program")
    default_index = options.index(preselected) if (preselected in options) else 0
    if preselected is not None:
        st.info(f"Preselected from the Per-transcript tab: P{preselected}",
                 icon="🔗")
    sel = st.selectbox(
        "Program",
        options=options,
        format_func=lambda p: (f"P{p} — "
                                f"{progs.loc[progs['program']==p, 'reading'].iloc[0]}"
                                f"  (n="
                                f"{int(progs.loc[progs['program']==p, 'n_modules'].iloc[0]):,}"
                                f")"),
        index=default_index,
        help="Pick a program to see its top TFs, position density, "
             "fingerprint promoters, and GO BP enrichment.",
    )
    reading = progs.loc[progs["program"] == sel, "reading"].iloc[0]

    # ---- Program summary card (reference / context for all 10) -----------
    with st.container(border=True):
        st.markdown(
            "### Program summary",
            help=HELP_PROG_SUMMARY + "  \n\n"
                 "**`tau`** is the Yanai 2005 tissue-specificity index over "
                 "the program's mean module-TF TPM across the 66 GTEx "
                 "tissues. 0 = uniform, 1 = single-tissue. **`top tissue`** "
                 "= where the program's TFs are most expressed on average.",
        )
        progs_disp = progs.copy()
        if db.gtex_available():
            tau_df = db.gtex_program_tissue_specificity()
            if not tau_df.empty:
                progs_disp = progs_disp.merge(
                    tau_df[["program", "tau", "top_tissue",
                             "top_tissue_mean_tpm"]],
                    on="program", how="left",
                )

        cols = ["program", "n_modules", "median_center", "median_width",
                "mean_dom_weight", "top_tfs", "reading",
                "tau", "top_tissue", "top_tissue_mean_tpm"]
        cols = [c for c in cols if c in progs_disp.columns]
        st.dataframe(
            progs_disp[cols],
            column_config={
                "program":         st.column_config.TextColumn(
                    "P",
                    help="Program id (1–10), assigned by NMF rank."),
                "n_modules":       st.column_config.NumberColumn(
                    "# modules",
                    help="Number of modules whose dominant program is this "
                         "one (out of 76,999 total)."),
                "median_center":   st.column_config.NumberColumn(
                    "median bp from TSS",
                    help="Median position of the program's modules relative "
                         "to the TSS, in transcription-oriented bp. "
                         "Negative = upstream, positive = downstream."),
                "median_width":    st.column_config.NumberColumn(
                    "median width (bp)",
                    help="Median module width (bp) — narrower modules tend "
                         "to mark precise TF clusters, wider ones span "
                         "extended regulatory regions."),
                "mean_dom_weight": st.column_config.NumberColumn(
                    "mean dominant weight",
                    format="%.3f",
                    help="Average NMF coefficient (H matrix) of the program "
                         "across modules it's assigned to. Higher = a "
                         "more 'concentrated' program signature."),
                "top_tfs":         st.column_config.TextColumn(
                    "top TFs",
                    help="Top 4–6 TFs by H loading, comma-joined."),
                "reading":         st.column_config.TextColumn(
                    "reading",
                    help="A short human label summarizing the program's "
                         "biology (curated from top TFs + top GO terms)."),
                "tau":             st.column_config.NumberColumn(
                    "tau (tissue specificity)",
                    format="%.3f",
                    help="Yanai 2005 index over the program's mean module-"
                         "TF TPM across GTEx tissues (tissues with ≥40 "
                         "donors). 0 = uniform across tissues, 1 = single-"
                         "tissue. See Methods for MIN_SAMPLES_FOR_TAU."),
                "top_tissue":      st.column_config.TextColumn(
                    "top tissue",
                    help="GTEx tissue where the program's TFs are most "
                         "expressed on average."),
                "top_tissue_mean_tpm": st.column_config.NumberColumn(
                    "top tissue TPM",
                    format="%.1f",
                    help="Mean TPM of the program's TFs in the top tissue. "
                         "Useful sanity check — should be ≫ 1 for any "
                         "biologically active program."),
            },
            hide_index=True, width="stretch",
        )

    # ---- Driver-class distribution (cross-program view, highlights sel) ---
    if db.gtex_available():
        dist = db.gtex_program_driver_class_distribution()
        if not dist.empty:
            with st.container(border=True):
                st.markdown("#### Module driver classes by program",
                            help=HELP_DRIVER_CLASS)
                st.caption(
                    "Each row = one program; segments split its modules into "
                    f"the three driver classes. P{int(sel)} is outlined."
                )
                st.plotly_chart(
                    plotting.fig_program_driver_class_distribution(
                        dist, selected=int(sel),
                    ),
                    width="stretch",
                )

    # ---- Drill-down: top-TFs (left) | position density (right) -----------
    col1, col2 = st.columns(2, gap="medium")
    with col1:
        with st.container(border=True):
            st.markdown("#### Top TFs by H loading", help=HELP_TOP_TFS)
            ttf = db.get_program_top_tfs(int(sel), limit=30)
            st.plotly_chart(plotting.fig_program_tf_top(ttf, int(sel)),
                            width="stretch")
    with col2:
        with st.container(border=True):
            st.markdown("#### Position density", help=HELP_POS_DENSITY)
            centers_df = db.get_program_module_centers(int(sel))
            st.plotly_chart(
                plotting.fig_program_position_density(
                    centers_df["center_offset"].to_numpy(),
                    int(sel), reading,
                ),
                width="stretch",
            )

    # ---- Top TFs enriched table (full width) ------------------------------
    with st.container(border=True):
        st.markdown(
            "#### Top TFs — full table",
            help=(HELP_TOP_TFS + "  \n\n"
                  "**`TF cluster (K=8)`** = which of the precomputed "
                  "filtered-K8 hierarchical clusters this TF falls in. "
                  "**`# atlas TSSs bound`** = how many of the 19,745 "
                  "canonical TSSs this TF binds with at least one core "
                  "score≥500 peak. **`top tissue`** + **`TPM`** = where "
                  "the TF gene is most expressed in GTEx (max-of-"
                  "transcripts max-tissue)."),
        )
        ttf_full = db.get_program_top_tfs_enriched(int(sel), limit=30)
        if ttf_full.empty:
            st.info("—")
        else:
            cols = ["rank", "tf", "loading",
                    "cluster_filtered", "cluster_no_filter",
                    "n_bound_tss_core", "top_tissue", "top_tpm"]
            cols = [c for c in cols if c in ttf_full.columns]
            st.dataframe(
                ttf_full[cols],
                hide_index=True, width="stretch",
                column_config={
                    "rank":             st.column_config.NumberColumn(
                        "rank",
                        help="Rank within this program by H loading "
                             "(1 = most characteristic TF)."),
                    "tf":               st.column_config.TextColumn(
                        "TF",
                        help="Transcription factor gene symbol."),
                    "loading":          st.column_config.NumberColumn(
                        "H loading", format="%.3f",
                        help="NMF H-matrix coefficient — the program's "
                             "per-TF weight. Higher = more characteristic "
                             "of this program."),
                    "cluster_filtered": st.column_config.TextColumn(
                        "K=8 cluster",
                        help="Hierarchical cluster id under the K=8 "
                             "FILTERED clustering of the per-TF aggregate "
                             "profiles (score-filtered peaks). Useful as "
                             "an independent biological grouping."),
                    "cluster_no_filter": st.column_config.TextColumn(
                        "K=12 cluster",
                        help="Hierarchical cluster id under the K=12 "
                             "UNFILTERED clustering (all peaks regardless "
                             "of score). Finer-grained alternative."),
                    "n_bound_tss_core": st.column_config.NumberColumn(
                        "# atlas TSSs bound",
                        help="How many of the 19,745 canonical TSSs this "
                             "TF binds with at least one core "
                             "score≥500 peak."),
                    "top_tissue":       st.column_config.TextColumn(
                        "GTEx top tissue",
                        help="Tissue where this TF gene is most expressed "
                             "(max-of-transcripts max-tissue across "
                             "GTEx v8)."),
                    "top_tpm":          st.column_config.NumberColumn(
                        "TPM", format="%.1f",
                        help="TPM in the top tissue (max-of-transcripts "
                             "max-tissue)."),
                },
            )

    # ---- TF × tissue heatmap (full width) ---------------------------------
    if db.gtex_available():
        with st.container(border=True):
            st.markdown("#### TF expression across GTEx tissues",
                        help=HELP_TF_TISSUE_HEATMAP)
            c1, c2 = st.columns([1, 5])
            with c1:
                n_tfs = st.slider(
                    "Top N TFs", min_value=10, max_value=50, value=20, step=5,
                    help="How many of the program's top TFs (by H loading) "
                         "to show as rows.",
                    key=f"prog_tf_tissue_n_{int(sel)}",
                )
                tissue_order = st.radio(
                    "Tissue order", ["program signal", "alphabetical"],
                    index=0, horizontal=False,
                    help="`program signal` groups the program's hot tissues "
                         "together; `alphabetical` keeps tissues in a fixed "
                         "order for cross-program comparison.",
                    key=f"prog_tf_tissue_order_{int(sel)}",
                )
            M, loadings = db.gtex_program_tf_tissue_matrix(
                int(sel), limit=int(n_tfs),
            )
            with c2:
                if M.empty:
                    st.info("No GTEx expression data for this program's top "
                            "TFs.")
                else:
                    order_key = ("program_signal"
                                 if tissue_order == "program signal"
                                 else "alpha")
                    st.plotly_chart(
                        plotting.fig_program_tf_tissue_heatmap(
                            M, loadings, int(sel), reading,
                            tissue_order=order_key,
                        ),
                        width="stretch",
                    )

    # ---- Top GO BP terms (collapsed; click to expand) ---------------------
    with st.expander("Top GO BP terms (genome bg) — click to expand",
                      expanded=False):
        st.caption(HELP_TOP_GO)
        go_df = db.get_program_top_go(int(sel), limit=15)
        if go_df.empty:
            st.info("No GO terms loaded for this program.")
        else:
            go_df["term"] = (go_df["term"]
                              .str.replace("GOBP_", "")
                              .str.replace("_", " ").str.lower())
            st.dataframe(
                go_df[["rank", "term", "fg_in", "set_size_in_bg",
                       "odds_ratio", "q_value"]],
                hide_index=True, width="stretch",
                column_config={
                    "rank":           st.column_config.NumberColumn(
                        "rank",
                        help="Rank within this program by q-value."),
                    "term":           st.column_config.TextColumn(
                        "term",
                        help="GO Biological Process term, prettified "
                             "(GOBP_ prefix and underscores removed)."),
                    "fg_in":          st.column_config.NumberColumn(
                        "fg",
                        help="Foreground intersection — # genes annotated "
                             "to this GO term that are ALSO in the "
                             "program's gene set."),
                    "set_size_in_bg": st.column_config.NumberColumn(
                        "term size",
                        help="Total # MSigDB genes annotated to this GO "
                             "term (the term's universe size)."),
                    "odds_ratio":     st.column_config.NumberColumn(
                        "OR", format="%.2f",
                        help="Odds ratio of program-gene set vs MSigDB "
                             "background. OR > 1 = enrichment."),
                    "q_value":        st.column_config.NumberColumn(
                        "q", format="%.1e",
                        help="Hypergeometric p-value adjusted by "
                             "Benjamini–Hochberg FDR across all terms."),
                },
            )

    # ---- Promoter fingerprint gallery (collapsed; click to expand) --------
    with st.expander(
            f"Representative P{int(sel)} promoters — fingerprint gallery "
            "(click to expand)",
            expanded=False):
        st.caption(
            "Top modules of this program ranked by dominant NMF weight, "
            "with each module's top-5 driver TFs (by |r| against the "
            f"target across GTEx tissues). Quick answer to 'what does a "
            f"P{int(sel)} promoter actually look like?' Click ↗ to "
            "inspect any gene in the Per-transcript tab."
        )
        n_fp = st.slider(
            "# representative promoters", 3, 12, 6, step=3,
            help="How many top-weighted modules of this program to show "
                 "as fingerprint cards. Cards are sorted by dominant "
                 "NMF weight descending.",
            key=f"prog_fp_n_{int(sel)}",
        )
        reps = db.get_program_representative_modules(int(sel), n=int(n_fp))
        if reps.empty:
            st.info("No modules found for this program.")
        else:
            rep_rows = reps.to_dict("records")
            # 2-column grid
            for i in range(0, len(rep_rows), 2):
                cols = st.columns(2, gap="small")
                for col, row in zip(cols, rep_rows[i:i+2]):
                    with col:
                        with st.container(border=True):
                            st.markdown(
                                f"##### {row['gene_name']}  "
                                f"<span style='color:#888;font-weight:normal;"
                                f"font-size:0.85em'>· P{int(sel)} weight "
                                f"{float(row['dominant_weight']):.2f}</span>",
                                unsafe_allow_html=True,
                            )
                            st.caption(
                                f"module @ {int(row['center_offset']):+,} bp · "
                                f"width {int(row['width'])} bp · "
                                f"`{row['transcript_id']}`"
                            )
                            if row["top_tfs"]:
                                st.markdown(
                                    f"**Top TFs:**  {row['top_tfs']}"
                                )
                            else:
                                st.caption(
                                    "_(no module–TF correlation evidence)_"
                                )
                            if st.button(
                                f"↗ Open {row['gene_name']} →",
                                key=f"fp_open_{int(sel)}_{row['module_id']}",
                                use_container_width=True,
                            ):
                                st.session_state["tx_gene_select"] = \
                                    row["gene_name"]
                                nav.goto("transcript")

    # ---- Strand-asymmetry verification anchor (expander) ------------------
    with st.expander("Strand asymmetry — verification anchor",
                      expanded=False):
        st.caption(
            "Fraction of each program's modules whose parent transcript "
            "is on the (+) strand. Should be near 50% — module discovery "
            "and program assignment are computed in **transcription-"
            "oriented** coordinates, so any large per-program strand "
            "skew would point to a chromatin or pipeline artifact rather "
            "than biology."
        )
        sd = db.get_program_strand_distribution()
        if sd.empty:
            st.info("No strand data — module → transcript join returned 0.")
        else:
            st.plotly_chart(
                plotting.fig_program_strand_asymmetry(sd, selected=int(sel)),
                width="stretch",
            )

    # ---- Co-occurrence demoted to an expander -----------------------------
    with st.expander("Program × program co-occurrence at the gene level",
                      expanded=False):
        st.caption(
            "For every (P_i, P_j), how many genes have at least one module "
            "in P_i AND at least one in P_j (diagonal = same program with "
            "multiple modules in the same gene). Lift = observed / expected "
            "under independence; >1 means these programs cluster at the "
            "same promoters more often than chance, <1 means they tend to "
            "be mutually exclusive."
        )
        col_a, col_b = st.columns([1, 4])
        with col_a:
            mode = st.radio("Display", ["lift", "count"], index=0,
                             horizontal=False,
                             help="`lift` = observed/expected (color-"
                                  "diverging around 1); `count` = raw "
                                  "shared-gene tally.")
        coocc = db.get_program_cooccurrence()
        with col_b:
            st.plotly_chart(
                plotting.fig_program_cooccurrence(coocc, mode=mode),
                width="stretch",
            )

    # ---- Downloads --------------------------------------------------------
    with st.expander("Download data", expanded=False):
        st.caption(
            "TSVs for the tables on this page. The full multi-tab summary "
            "lives at `docs/summary_tables.xlsx` in the repo."
        )
        st.download_button(
            "Program summary (all 10 programs, with tau + top tissue) — TSV",
            data=progs_disp.to_csv(sep="\t", index=False).encode(),
            file_name="program_summary.tsv",
            mime="text/tab-separated-values",
            key="prog_dl_summary",
        )
        if not ttf_full.empty:
            st.download_button(
                f"P{int(sel)} top TFs (enriched) — TSV",
                data=ttf_full.to_csv(sep="\t", index=False).encode(),
                file_name=f"program_P{int(sel)}_top_tfs.tsv",
                mime="text/tab-separated-values",
                key="prog_dl_tfs",
            )
        if not go_df.empty:
            st.download_button(
                f"P{int(sel)} top GO BP terms — TSV",
                data=go_df.to_csv(sep="\t", index=False).encode(),
                file_name=f"program_P{int(sel)}_top_go.tsv",
                mime="text/tab-separated-values",
                key="prog_dl_go",
            )
        if db.gtex_available():
            dist_dl = db.gtex_program_driver_class_distribution()
            if not dist_dl.empty:
                st.download_button(
                    "Module driver-class distribution (all programs) — TSV",
                    data=dist_dl.to_csv(sep="\t", index=False).encode(),
                    file_name="program_driver_class_distribution.tsv",
                    mime="text/tab-separated-values",
                    key="prog_dl_driver",
                )
