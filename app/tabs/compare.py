"""Compare tab: side-by-side view of two transcripts' promoter architecture,
program presence, GTEx expression, and DepMap essentiality."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import db, plotting, ui


# Curated example pairs to seed the comparison with informative biology.
EXAMPLE_PAIRS = [
    ("GAPDH", "IL6",
     "housekeeper vs immune cytokine"),
    ("TBP",   "MYC",
     "general PIC vs proliferation amplifier"),
    ("CDK4",  "RB1",
     "G1/S cell-cycle pair (kinase vs its substrate)"),
]


def _gene_to_canonical_tx(gene: str) -> str | None:
    """Resolve a gene symbol to its canonical transcript_id, or None."""
    if not gene:
        return None
    df = db.get_transcripts_for_gene(gene)
    if df is None or df.empty:
        return None
    return df["transcript_id"].iloc[0]


def _set_pair(gene_a: str, gene_b: str) -> None:
    """Stash the two gene symbols in session state so the selectors render
    them on the next rerun.

    Writes the *_select widget keys directly — once a keyed selectbox has
    rendered, Streamlit's widget state wins over the index= parameter, so
    writing only to cmp_gene_a / cmp_gene_b has no visible effect.
    """
    st.session_state["cmp_gene_a"] = gene_a
    st.session_state["cmp_gene_b"] = gene_b
    st.session_state["cmp_gene_a_select"] = gene_a
    st.session_state["cmp_gene_b_select"] = gene_b


def render() -> None:
    ui.intro_card(
        title="Compare two transcripts side-by-side",
        what="Two canonical transcripts on aligned coordinates: promoter "
             "maps, shared/distinct programs, paired GTEx expression, and "
             "paired DepMap essentiality — all in one view.",
        objective="Answer *'how is gene A regulated compared to gene B'* "
                   "without flipping between two Per-transcript pages.",
        significance="Most biological questions are comparative — "
                      "housekeeper vs tissue-specific, oncogene vs tumor "
                      "suppressor, paralog vs paralog. The atlas is "
                      "sharpest when it makes those comparisons direct.",
    )

    # ---- Curated quick-fill pairs -----------------------------------------
    st.markdown("### Quick-fill pairs",
                 help="One-click prefill of two gene symbols into the "
                      "selectors below. Pick a pair to see a worked example.")
    cols = st.columns(len(EXAMPLE_PAIRS))
    for i, (a, b, blurb) in enumerate(EXAMPLE_PAIRS):
        with cols[i]:
            st.button(
                f"**{a}** vs **{b}**  \n*{blurb}*",
                use_container_width=True,
                on_click=_set_pair, args=(a, b),
                key=f"cmp_pair_btn_{i}",
            )

    # ---- Gene selectors ---------------------------------------------------
    gene_list = [""] + db.list_genes()

    def _idx(gene: str) -> int:
        try:
            return gene_list.index(gene)
        except ValueError:
            return 0

    col_a, col_swap, col_b = st.columns([5, 1, 5])
    with col_a:
        gene_a = st.selectbox(
            "Gene A",
            options=gene_list,
            index=_idx(st.session_state.get("cmp_gene_a", "")),
            placeholder="Type to search…",
            key="cmp_gene_a_select",
            help="Left-hand transcript in the comparison.",
        )
    with col_swap:
        st.write("")
        st.write("")
        if st.button("↔", help="Swap A and B",
                      use_container_width=True, key="cmp_swap"):
            cur_a = st.session_state.get("cmp_gene_a_select", "")
            cur_b = st.session_state.get("cmp_gene_b_select", "")
            _set_pair(cur_b, cur_a)
            st.rerun()
    with col_b:
        gene_b = st.selectbox(
            "Gene B",
            options=gene_list,
            index=_idx(st.session_state.get("cmp_gene_b", "")),
            placeholder="Type to search…",
            key="cmp_gene_b_select",
            help="Right-hand transcript in the comparison.",
        )

    tx_a = _gene_to_canonical_tx(gene_a)
    tx_b = _gene_to_canonical_tx(gene_b)

    if not tx_a or not tx_b:
        st.info("Pick two genes (or click a quick-fill pair above) to load "
                "the comparison.")
        with st.container(border=True):
            st.markdown("##### What you'll see once a pair is loaded")
            p1, p2, p3, p4 = st.columns(4)
            with p1:
                st.markdown("**Promoter maps**")
                st.caption("Side-by-side KDE + TF rugs + module ribbons on "
                            "aligned coordinates (±1.5 kb of TSS).")
            with p2:
                st.markdown("**Program presence diff**")
                st.caption("Shared vs A-only vs B-only programs — which "
                            "modules are common, which are distinct.")
            with p3:
                st.markdown("**Paired GTEx expression**")
                st.caption("Per-tissue TPM side-by-side, with a correlation "
                            "summary across tissues.")
            with p4:
                st.markdown("**Paired DepMap essentiality**")
                st.caption("Median Chronos per lineage for each gene, with "
                            "the essentiality threshold marked.")
        return
    if tx_a == tx_b:
        st.warning("Both selectors resolved to the same transcript — pick "
                   "two different genes to compare.")
        return

    meta_a = db.get_tss_meta(tx_a)
    meta_b = db.get_tss_meta(tx_b)
    cfg_a  = db.get_gene_config(tx_a)
    cfg_b  = db.get_gene_config(tx_b)
    arch_a = db.get_gene_archetype(tx_a)
    arch_b = db.get_gene_archetype(tx_b)

    # ---- Header chips -----------------------------------------------------
    with st.container(border=True):
        c1, c2 = st.columns(2)
        for col, gene, meta, cfg, arch in [
            (c1, gene_a, meta_a, cfg_a, arch_a),
            (c2, gene_b, meta_b, cfg_b, arch_b),
        ]:
            with col:
                if meta is None:
                    st.error(f"No TSS data for {gene}.")
                    continue
                st.markdown(
                    f"**{meta['gene_name']}** · `{meta['transcript_id']}` · "
                    f"chr{meta['chrom']}:{int(meta['tss']):,} "
                    f"({meta['strand']})"
                )
                if arch:
                    a = int(arch["dominant_archetype"])
                    # Thin colored archetype bar (cue, not a sentence).
                    st.markdown(
                        f"<div style='height:6px;background:"
                        f"{plotting.PROGRAM_COLORS[(a - 1) % 10]};"
                        f"border-radius:3px;margin:6px 0 8px 0'></div>",
                        unsafe_allow_html=True,
                    )
                # Four metric cards per side: Archetype · weight · Modules ·
                # Program path. Folds the old inline sentence in.
                ma, mw, mm, mp = st.columns(4)
                if arch:
                    a = int(arch["dominant_archetype"])
                    ma.metric("Archetype", f"A{a}",
                              help="Dominant gene-level archetype.")
                    mw.metric("Weight",
                              f"{arch['dominant_weight']:.2f}",
                              help="How cleanly this gene fits its "
                                   "dominant archetype (1.0 = pure).")
                if cfg:
                    mm.metric("Modules", int(cfg["n_modules"]),
                              help="# regulatory modules at this TSS.")
                    mp.metric("Program path", cfg["program_path"],
                              help="Ordered dominant programs across the "
                                   "gene's modules, upstream → downstream.")

    # ---- Shared controls --------------------------------------------------
    # Slider lives inside the promoter-maps card below so it sits with the
    # plots it actually controls.

    # ---- Aligned promoter maps -------------------------------------------
    with st.container(border=True):
        st.markdown(
            "**Promoter maps — stacked, shared X-axis**",
            help="Compact view: KDE density + TF rug (colored by score) + "
                 "module ribbon for each transcript. Per-program rows are "
                 "omitted to keep the comparison vertically tight; visit "
                 "the Per-transcript tab for the full breakdown.",
        )
        # Score-range slider lives with the plots it controls.
        sc_col, _ = st.columns([2, 5])
        with sc_col:
            score_range = st.slider(
                "peak score range",
                0, 1000, (500, 1000), step=50,
                key="cmp_score_range",
                help="Same range applied to both promoter maps for fair "
                     "comparison. Each rug tick is colored by its "
                     "ChIP-atlas score (Viridis 0–1000).",
            )
        for label, tx, meta in [("A", tx_a, meta_a), ("B", tx_b, meta_b)]:
            st.markdown(f"##### {label} — {meta['gene_name']}")
            modules_df = db.get_modules_for_transcript(tx)
            peaks_df   = db.get_peaks_for_tss(int(meta["tss_id"]), min_score=0)
            if modules_df.empty and peaks_df.empty:
                st.info(f"No modules/peaks for {meta['gene_name']}.")
                continue
            fig = plotting.fig_transcript_view(
                peaks_df, modules_df, meta,
                score_range=score_range, compact=True,
            )
            st.plotly_chart(fig, width="stretch", theme=None)

    # ---- Program presence diff -------------------------------------------
    mod_a = db.get_modules_for_transcript(tx_a)
    mod_b = db.get_modules_for_transcript(tx_b)
    progs_a = (set(int(p) for p in mod_a["dominant_program"])
               if not mod_a.empty else set())
    progs_b = (set(int(p) for p in mod_b["dominant_program"])
               if not mod_b.empty else set())
    counts_a = (mod_a.groupby("dominant_program").size().to_dict()
                if not mod_a.empty else {})
    counts_b = (mod_b.groupby("dominant_program").size().to_dict()
                if not mod_b.empty else {})

    only_a = sorted(progs_a - progs_b)
    only_b = sorted(progs_b - progs_a)
    both   = sorted(progs_a & progs_b)

    with st.container(border=True):
        st.markdown(
            "**Program presence diff**",
            help="Which k=10 programs operate at each promoter, and which "
                 "are shared. Counts in parentheses = number of modules in "
                 "that gene assigned to that program.",
        )
        c_only_a, c_both, c_only_b = st.columns(3)
        with c_only_a:
            st.markdown(f"**Only in A ({gene_a})** — {len(only_a)}")
            for p in only_a:
                st.markdown(_program_chip(p, counts_a.get(p, 0)),
                             unsafe_allow_html=True)
            if not only_a:
                st.caption("_none_")
        with c_both:
            st.markdown(f"**In both** — {len(both)}")
            for p in both:
                st.markdown(_program_chip(p, counts_a.get(p, 0),
                                             counts_b.get(p, 0)),
                             unsafe_allow_html=True)
            if not both:
                st.caption("_none_")
        with c_only_b:
            st.markdown(f"**Only in B ({gene_b})** — {len(only_b)}")
            for p in only_b:
                st.markdown(_program_chip(p, counts_b.get(p, 0)),
                             unsafe_allow_html=True)
            if not only_b:
                st.caption("_none_")

    # ---- GTEx expression overlay -----------------------------------------
    if db.gtex_available():
        with st.container(border=True):
            st.markdown(
                "**GTEx tissue expression — paired**",
                help="Mean TPM across GTEx V11 tissues for each transcript, "
                     "shown as grouped bars. Tissues sorted by combined "
                     "mean (most expressed at left).",
            )
            stats_a = db.gtex_transcript_stats(tx_a)
            stats_b = db.gtex_transcript_stats(tx_b)
            if stats_a.empty and stats_b.empty:
                st.info("No GTEx coverage for either transcript.")
            else:
                fig = plotting.fig_gtex_compare(
                    stats_a, stats_b, gene_a, gene_b)
                st.plotly_chart(fig, width="stretch")

    # ---- DepMap essentiality side-by-side --------------------------------
    if db.depmap_available():
        with st.container(border=True):
            st.markdown(
                "**DepMap CRISPR essentiality — per lineage**",
                help="Median Chronos score per cancer lineage. More-negative "
                     "= more essential. Dashed line at −1 = standard "
                     "essentiality threshold.",
            )
            d_a = db.depmap_gene_lineage(gene_a)
            d_b = db.depmap_gene_lineage(gene_b)
            c1, c2 = st.columns(2)
            with c1:
                if d_a.empty:
                    st.info(f"{gene_a} not in DepMap.")
                else:
                    st.plotly_chart(
                        plotting.fig_depmap_lineage_bar(d_a, gene_a),
                        width="stretch",
                    )
            with c2:
                if d_b.empty:
                    st.info(f"{gene_b} not in DepMap.")
                else:
                    st.plotly_chart(
                        plotting.fig_depmap_lineage_bar(d_b, gene_b),
                        width="stretch",
                    )

    # ---- Downloads --------------------------------------------------------
    with st.expander("Download this comparison's data", expanded=False):
        for label, tx, gene in [("A", tx_a, gene_a), ("B", tx_b, gene_b)]:
            mods = db.get_modules_for_transcript(tx)
            if not mods.empty:
                st.download_button(
                    f"{label} ({gene}) modules — TSV",
                    data=mods.to_csv(sep="\t", index=False).encode(),
                    file_name=f"{gene}_{tx}_modules.tsv",
                    mime="text/tab-separated-values",
                    key=f"cmp_dl_mod_{label}",
                )


def _program_chip(program: int, count_a: int, count_b: int | None = None) -> str:
    """Color-keyed chip line: 'P3 — 2 modules (A) · 1 module (B)'."""
    color = plotting.PROGRAM_COLORS[(program - 1) % 10]
    if count_b is None:
        body = f"{count_a} module{'s' if count_a != 1 else ''}"
    else:
        body = (f"{count_a} (A) · {count_b} (B)"
                if count_a != count_b
                else f"{count_a} each")
    return (
        f"<div style='border-left:5px solid {color};"
        f"padding:3px 8px;margin:3px 0;background:#f6f8fb;"
        f"border-radius:3px;font-size:0.9em'>"
        f"<b>P{program}</b> — {body}</div>"
    )
