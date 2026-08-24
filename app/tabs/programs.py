"""Programs — the 140 genome-wide regulatory programs.

Reads the annotation-free build. The promoter factorization this tab formerly
showed no longer runs: the hybrid architecture uses ONE program vocabulary
across the site, taken from the genome-wide build, because the regression gate
showed both layers see the same promoters (98.2% recovery at 12 bp median
offset) and a second factorization would give users two sets of program
numbers to reconcile.

The previous promoter implementation is in git history.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import db, plotting, ui


def render() -> None:
    ui.intro_card(
        title="Programs — recurring TF co-binding across the genome",
        what="**140 programs** factorized from 467,223 annotation-free "
             "elements × 1,793 TFs. **72 are substantive** (≥100 elements and "
             "seed stability ≥0.90) and **64 replicate across disjoint "
             "experiments**.",
        objective="Answer 'which factors act together, and where' — including "
                   "the 353,550 distal elements a ±1.5 kb promoter window "
                   "cannot see at all.",
        significance="Canonical complexes fall out without being told: PRC2 "
                      "(EZH2/SUZ12/JARID2), CoREST (KDM1A/RCOR1/HMG20B), "
                      "COMPASS (KMT2A/RBBP5/AFF1), PRC1.6 (E2F6/L3MBTL2/MGA), "
                      "cohesin, and Integrator/NELF pausing. No complex "
                      "annotation enters the pipeline.",
    )

    if not db.has_genome_layer():
        st.info("This database predates the genome-wide layer. Rebuild with "
                "`data/build_app_db_genome.py`.")
        return

    progs = db.get_genome_programs()
    fams = db.get_program_families()
    fam_label = ({int(r.family): str(r.label) for _, r in fams.iterrows()}
                 if "label" in fams.columns else {})

    # ---- overview ---------------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "### All programs",
            help="`substantive` combines size with stability: a program pinned "
                 "to a handful of elements reconverges perfectly across seeds "
                 "and would otherwise read as highly reproducible — 22 of the "
                 "140 have fewer than 100 elements and cover 0.1% of the data "
                 "between them. `promFEm`/`distFEm` are complexity-matched "
                 "log2 enrichments; distal elements carry fewer TFs (median 21 "
                 "vs 48), so an unmatched value would make any sparse-loading "
                 "program look distal-specific.",
        )
        only_sub = st.checkbox("substantive only", value=True,
                               key="prog_only_sub")
        view = progs[progs.substantive] if only_sub else progs
        show = view.copy()
        if fam_label:
            show["family"] = show.family.map(
                lambda f: f"{int(f)} — {fam_label.get(int(f), '')}"
                if pd.notna(f) else "")
        cols = [c for c in ("program", "family", "n_elements", "seed_stability",
                            "substantive", "promoter_log2FE_matched",
                            "distal_log2FE_matched", "median_n_tfs", "top_tfs")
                if c in show.columns]
        st.dataframe(show[cols], hide_index=True, use_container_width=True,
                     column_config=ui.PROGRAM_COLUMNS)
        st.caption(f"{len(view):,} of {len(progs):,} programs shown.")

    # ---- one program ------------------------------------------------------
    sel = st.selectbox(
        "Program", options=list(progs.program),
        key="prog_pick",
        format_func=lambda p: (
            f"{int(p)} — {str(progs.loc[progs.program == p, 'top_tfs'].iloc[0])[:44]}"))
    row = progs[progs.program == sel].iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("elements", f"{int(row.n_elements):,}")
    c2.metric("seed stability", f"{row.seed_stability:.3f}")
    c3.metric("substantive", "yes" if row.substantive else "no")
    if pd.notna(row.get("family")):
        c4.metric("family", f"{int(row.family)}",
                  help=fam_label.get(int(row.family), ""))

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(
            plotting.fig_program_distance(
                db.get_program_distance_hist(int(sel)), int(sel)),
            width="stretch", theme=None)
    with right:
        st.markdown("**Where its elements sit**",
                    help="Distances are medians and p90 rather than means: the "
                         "distribution is heavily skewed, so a mean would "
                         "describe no actual element.")
        st.dataframe(
            db.get_program_element_stats(int(sel)),
            hide_index=True, use_container_width=True,
            column_config={
                "stratum": st.column_config.TextColumn(
                    "stratum",
                    help="Where the element sits relative to the nearest "
                         "canonical TSS: promoter (within ±1.5 kb), proximal, "
                         "or distal."),
                "n": st.column_config.NumberColumn(
                    "elements", format="%d",
                    help="This program's elements in that stratum."),
                "median_dist": st.column_config.NumberColumn(
                    "median bp", format="%d",
                    help="Median absolute distance to the nearest TSS. Median "
                         "rather than mean: the distribution spans three "
                         "orders of magnitude and a mean would describe no "
                         "actual element."),
                "p90_dist": st.column_config.NumberColumn(
                    "p90 bp", format="%d",
                    help="90th percentile distance — the tail the median "
                         "hides. A program can have a 500 bp median and a "
                         "1.7 Mb p90."),
                "median_tfs": st.column_config.NumberColumn(
                    "median TFs", format="%d",
                    help="Median assigned TFs per element in this stratum. "
                         "Promoter elements carry roughly twice as many as "
                         "distal ones, which is why the enrichments are "
                         "complexity-matched."),
                "median_width": st.column_config.NumberColumn(
                    "median width", format="%d",
                    help="Median element width in bp."),
            })

    st.markdown("**Top transcription factors**")
    st.dataframe(
        db.get_genome_program_tfs(int(sel), limit=30),
        hide_index=True, use_container_width=True,
        column_config=ui.PROGRAM_TF_COLUMNS)
