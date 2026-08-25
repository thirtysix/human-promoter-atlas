"""Per-transcript explorer: search by gene/transcript, see modules + peaks +
   the gene's program_path."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from app.lib import db, plotting, ui, nav


HELP_GENE = (
    "Search by gene symbol. Each gene has one canonical-tagged transcript "
    "(Ensembl_canonical) on a standard chromosome (1–22, X, Y, MT)."
)
HELP_SCORE_RANGE = (
    "Filter for the TF rug panel only — peaks outside this score range are "
    "hidden from the visualization but kept for the per-TSS density. "
    "ChIP-atlas peak score scales 0–1000 by how many experiments support the "
    "peak. **It opens at 500, above the score this build assigned at, so the "
    "default view is the better-supported peaks — which means the rug shows "
    "fewer TFs than the module tables below count. Drag the lower handle down "
    "to the assignment score to see every peak the modules were built from.** "
    "Each TF gets its own row and its own color; hover any tick for its score."
)
HELP_MODULES_TABLE = (
    "Each row is one detected regulatory module — a local concentration of "
    "TF binding within ±1.5 kb of the TSS. `weight` is the dominant program's "
    "share of the module's NMF mixture (0–1). `n_TFs (≥assigned score)` is the number "
    "of distinct TFs with at least one high-confidence peak in this module."
)
HELP_PROG_CARDS = (
    "One card per program present at this promoter. Click a card to see "
    "the program's top TFs and top GO BP terms inline. The 'Open in "
    "Programs tab' button preselects this program over there."
)


def _with_genome_annotation(modules_df, transcript_id: str):
    """Add `genome_program` / `family` to the promoter modules.

    Modules carry no program of their own on this build; they inherit one from
    the genome element at the same locus. A module below the genome support
    floor of 11 assigned TFs has no matching element, so the columns stay NA
    and the band falls back to neutral -- which is the honest rendering of
    "no program evidence here".
    """
    if modules_df is None or modules_df.empty:
        return modules_df
    ann = db.get_module_annotation(transcript_id)
    if ann is None or ann.empty or "module_id" not in ann.columns:
        return modules_df
    keep = ann[["module_id", "program", "family", "family_label"]].rename(
        columns={"program": "genome_program"})
    return modules_df.merge(keep, on="module_id", how="left")


def render() -> None:
    ui.intro_card(
        title="Per-transcript view — what's happening at this promoter",
        what="The full module decomposition of a single canonical promoter: "
             "smoothed density of TF binding, every individual TF binding "
             "within ±1.5 kb at the assignment score, and the modules detected from "
             "that density — each colored by the program family it inherits from the genome layer.",
        objective="Answer 'what TFs control THIS gene, in what positions, "
                   "and which of the 10 archetypal programs do they "
                   "implement?' for any canonical protein-coding transcript.",
        significance="Most canonical promoters carry **4.2 modules on "
                      "average** — multiple programs in parallel. The "
                      "per-program rows make e.g. *'this promoter is "
                      "regulated by P5 cohesin AND P7 PIC AND P1 chromatin'* "
                      "visible at a glance.",
    )

    st.subheader("Per-transcript view",
                  help="Search a gene to see its module decomposition, the TFs "
                       "binding each module, and which k=10 programs operate "
                       "at this promoter.")

    col_g, col_t = st.columns(2)
    with col_g:
        gene = st.selectbox(
            "Gene",
            options=[""] + db.list_genes(),
            index=0,
            placeholder="Type to search…",
            key="tx_gene_select",
            help=HELP_GENE,
        )
    transcripts = db.get_transcripts_for_gene(gene) if gene else None

    with col_t:
        if transcripts is not None and len(transcripts):
            tx_id = st.selectbox(
                "Transcript (canonical)",
                options=transcripts["transcript_id"].tolist(),
                index=0,
                key="tx_id_select",
                help="If a gene has more than one canonical TSS in Ensembl, "
                     "pick which one.",
            )
        else:
            tx_id = st.selectbox(
                "Transcript",
                options=[""] + db.list_transcripts(),
                index=0,
                placeholder="Or type any transcript_id",
                key="tx_id_freetext",
                help="Free-text transcript ID lookup.",
            )

    if not tx_id:
        st.info("Pick a gene or transcript above to load the per-promoter view.")
        return

    tss_meta = db.get_tss_meta(tx_id)
    if tss_meta is None:
        st.error(f"Transcript {tx_id} not found in the canonical TSS table.")
        return

    cfg = db.get_gene_config(tx_id)
    arch = db.get_gene_archetype(tx_id)
    arch_summary = db.get_archetypes() if arch else None

    # ----- Header card ------------------------------------------------------
    with st.container(border=True):
        head_md = (f"**{tss_meta['gene_name']}** · "
                   f"`{tss_meta['transcript_id']}` · "
                   f"chr{tss_meta['chrom']}:{tss_meta['tss']:,} "
                   f"({tss_meta['strand']})")
        st.markdown(head_md)

        # Top-line metrics — archetype + module/program counts in one row.
        a = None
        top_progs = ""
        n_arch_genes = 0
        if arch and arch_summary is not None:
            a = int(arch["dominant_archetype"])
            row = arch_summary[arch_summary["archetype"] == a]
            top_progs = row["top_programs"].iloc[0] if len(row) else ""
            n_arch_genes = int(row["n_genes"].iloc[0]) if len(row) else 0
            # Thin colored bar above the metrics to keep the archetype
            # color cue without the wordy sentence.
            st.markdown(
                f"<div style='height:6px;background:"
                f"{plotting.PROGRAM_COLORS[(a - 1) % 10]};"
                f"border-radius:3px;margin:6px 0 8px 0'></div>",
                unsafe_allow_html=True,
            )

        if cfg or a is not None:
            cols = st.columns(5)
            if a is not None:
                cols[0].metric(
                    "Archetype", f"A{a}",
                    help=f"Top programs in this archetype: {top_progs}. "
                         f"Dominant-archetype weight {arch['dominant_weight']:.2f} "
                         f"(how cleanly this gene fits A{a}; 1.0 = pure).")
                cols[1].metric(
                    "Genes in archetype", f"{n_arch_genes:,}",
                    help="How many canonical-promoter genes share this "
                         "archetype — a measure of how 'typical' this "
                         "gene's program loading is.")
            if cfg:
                cols[2].metric(
                    "Modules", int(cfg["n_modules"]),
                    help="Number of regulatory modules detected at this TSS.")
                cols[3].metric(
                    "Distinct programs",
                    len(set(cfg["program_path"].split(","))),
                    help="Unique k=10 programs operating at this promoter "
                         "— several programs = multiple co-existing TF "
                         "configurations.")
                cols[4].metric(
                    "Program path", cfg["program_path"],
                    help="Ordered list of dominant programs across the "
                         "gene's modules, upstream → downstream.")

    modules_df = db.get_modules_for_transcript(tx_id)
    peaks_df   = db.get_peaks_for_tss(int(tss_meta["tss_id"]), min_score=0)

    if modules_df.empty and peaks_df.empty:
        st.warning("No modules or peaks found for this TSS.")
        return

    # ----- Main figure card -------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "**Promoter map** — KDE density of TF binding, individual TF "
            "rugs, and modules colored by their program family.",
            help="Top: smoothed density of distinct-TF binding (each TF "
                 "contributes mass = 1 per TSS). Middle: one row per TF "
                 "with peak midpoints as vertical ticks (upstream TFs at "
                 "the top, downstream at the bottom). Bottom: each module "
                 "shown both in the overview ribbon and on a dedicated row "
                 "for the program it belongs to.",
        )

        # Score-range slider lives next to the plot it controls.
        c_score, c_filter = st.columns([1, 2])
        with c_score:
            score_range = st.slider(
                # Display filter, NOT the build threshold -- that is baked
                # into the tables and no slider can move it. Starts at 500 so
                # the default view is the better-supported peaks; max() keeps
                # it from ever sitting below the score the build assigned at.
                # While it sits ABOVE that score the rug deliberately shows
                # fewer TFs than the module tables count, which is why the
                # caption below says so rather than leaving it to be found.
                "peak score range", 0, 1000,
                (max(500, db.min_score_assign()), 1000), step=50,
                help=HELP_SCORE_RANGE,
                key=f"tx_score_range_{tx_id}",
            )

        # TFs available within the chosen score range, sorted by mean
        # position; rug panel renders upstream-at-top (descending).
        smin, smax = score_range
        avail = peaks_df[(peaks_df["score"] >= smin)
                         & (peaks_df["score"] <= smax)]
        if not avail.empty:
            tf_options = (avail.groupby("tf")["local_offset"].mean()
                                .sort_values(ascending=False).index.tolist())
        else:
            tf_options = []

        with c_filter:
            tf_filter = st.multiselect(
                f"Focus on specific TFs (showing {len(tf_options)} total — "
                "leave empty to show all)",
                options=tf_options,
                default=[],
                key=f"tx_tf_filter_{tx_id}",
                help="Type to search TF names. Picking one or more "
                     "restricts the rug panel to those TFs only — the KDE "
                     "density and module rows are unaffected.",
            )

        # The figure colours module bands by their genome-layer family, so
        # the annotation has to travel with the modules. Without it every
        # band fell back to neutral grey while three captions promised
        # colour-coding.
        modules_df = _with_genome_annotation(modules_df, tx_id)
        # The rug is 18 px per TF and alone decides page height, so it is
        # capped by default and opened on request.
        n_avail = int(avail["tf"].nunique()) if not avail.empty else 0
        show_all = False
        if n_avail > plotting.MAX_TF_ROWS:
            show_all = st.checkbox(
                f"Show all {n_avail} TF rows "
                f"(default: the {plotting.MAX_TF_ROWS} with the most peaks)",
                value=False, key=f"tx_show_all_tfs_{tx_id}",
                help=f"The rug adds ~18 px per TF, so all {n_avail} rows make "
                     f"a figure several screens tall. The hidden rows are the "
                     f"ones with fewest peaks here; the density curve, the "
                     f"structure track and the module rows use every TF "
                     f"either way.")
        gs = db.get_gene_structure(str(tss_meta["chrom"]), int(tss_meta["tss"]),
                                   str(tss_meta["strand"]),
                                   gene_name=str(tss_meta.get("gene_name") or ""))
        fig = plotting.fig_transcript_view(
            peaks_df, modules_df, tss_meta,
            score_range=score_range,
            tf_filter=tf_filter or None,
            gene_structure=gs,
            max_tf_rows=None if (show_all or tf_filter)
                        else plotting.MAX_TF_ROWS)
        st.plotly_chart(fig, width="stretch", theme=None)

    # ----- Beyond the promoter window --------------------------------------
    # The view above is fixed to +/-1.5 kb and cannot show a distal element:
    # SOX2's run from 11 kb to 535 kb. A single axis wide enough for both would
    # compress the promoter region to a few pixels, so the wider neighbourhood
    # gets its own zoom hierarchy, with the promoter box marked at every level
    # so the relationship stays legible.
    modules_df.attrs["transcript_id"] = tx_id
    _render_module_composition(modules_df, peaks_df)
    _render_neighbourhood(tss_meta)

    # ----- Programs present (clickable cards) -------------------------------
    if not modules_df.empty:
        st.markdown("### Programs operating at this promoter",
                     help=HELP_PROG_CARDS)
        prog_counts = (modules_df.groupby("dominant_program")
                                   .agg(n=("module_id", "size"),
                                        reading=("program_reading", "first"))
                                   .reset_index()
                                   .sort_values("n", ascending=False))
        # prog_counts is EMPTY when modules carry no promoter-program
        # assignment -- the build takes programs from the genome layer. The
        # enclosing guard checks modules_df, which is not empty, so this is
        # reached with zero rows and st.columns(0) raises
        # StreamlitInvalidColumnSpecError. The genome programs for this gene
        # are shown by the neighbourhood section below.
        if prog_counts.empty:
            st.caption(
                "Promoter modules here carry no program assignment of their "
                "own — this build takes programs from the genome-wide layer. "
                "See the element view below for the programs at this gene."
            )
            prog_counts = prog_counts.iloc[0:0]
        cols = st.columns(min(len(prog_counts), 5)) if len(prog_counts) else []
        for i, (_, row) in enumerate(prog_counts.iterrows()):
            p = int(row["dominant_program"])
            n = int(row["n"])
            with cols[i % len(cols)]:
                _render_program_card(p, row["reading"], n)

    # ----- GTEx expression + module-tissue activity (combined fig) ---------
    if db.gtex_available():
        with st.container(border=True):
            st.markdown(
                "### GTEx tissue expression & module activity",
                help="**Top panel:** per-tissue mean TPM (with IQR error "
                     "bars) for this transcript across GTEx V11. **Bottom "
                     "panel:** for each module at this promoter, the mean "
                     "TPM of its assigned TFs (at the build's assignment score, "
                     "inside the module) "
                     "per tissue. Both panels share the same tissue order — "
                     "hot cells under tall bars = a tissue where both the "
                     "transcript itself and a module's TFs are highly "
                     "expressed. Module-row labels are colored by the "
                     "module's program family, matching the "
                     "promoter map above.",
            )
            stats    = db.gtex_transcript_stats(tx_id)
            activity = db.gtex_module_activity_for_transcript(tx_id)
            if stats.empty:
                st.info("No GTEx coverage for this transcript "
                        "(3 of 19,745 atlas TSSs are missing from V11).")
            else:
                st.plotly_chart(
                    plotting.fig_gtex_expression_with_modules(
                        stats, activity, modules_df,
                        title_prefix=f"{tss_meta['gene_name']} — "),
                    width="stretch",
                )
                with st.expander("Per-tissue stats table", expanded=False):
                    st.dataframe(stats, hide_index=True,
                                  width="stretch",
                                  column_config={
                                      "tissue":    st.column_config.TextColumn(
                                          "tissue",
                                          help="GTEx v8 tissue name."),
                                      "n_samples": st.column_config.NumberColumn(
                                          "n",
                                          help="# donor samples in this "
                                               "tissue with quantified "
                                               "expression for this "
                                               "transcript."),
                                      "mean":   st.column_config.NumberColumn(
                                          "mean", format="%.2f",
                                          help="Mean TPM across donors."),
                                      "median": st.column_config.NumberColumn(
                                          "median", format="%.2f",
                                          help="Median TPM across donors. "
                                               "Robust to outliers."),
                                      "q1":     st.column_config.NumberColumn(
                                          "Q1", format="%.2f",
                                          help="25th percentile TPM."),
                                      "q3":     st.column_config.NumberColumn(
                                          "Q3", format="%.2f",
                                          help="75th percentile TPM."),
                                      "std":    st.column_config.NumberColumn(
                                          "std", format="%.2f",
                                          help="Standard deviation of TPM "
                                               "across donors."),
                                  })

        with st.container(border=True):
            st.markdown(
                "### TF–target expression correlation (across tissues)",
                help="Pearson r between each TF's tissue-expression profile "
                     "and *this transcript's* tissue expression, computed "
                     "across the 66 GTEx tissues. Strong positive r = TF "
                     "and target co-expressed across tissues (consistent "
                     "with regulation); strong negative = anti-correlated. "
                     "Restricted to |r| ≥ 0.3.",
            )
            min_r = st.slider("min |r|", 0.30, 0.95, 0.30, step=0.05,
                               help="Minimum absolute Pearson r to "
                                    "include. Raise to focus on the "
                                    "strongest TF–target correlations; "
                                    "0.30 is a permissive default.",
                               key="tx_corr_min_r")
            corrs = db.gtex_tf_target_correlations(tx_id, min_abs_r=min_r)
            if corrs.empty:
                st.info("No TF reaches the |r| threshold for this transcript.")
            else:
                # Annotate which TFs actually bind this gene
                bound_tfs = set()
                if not peaks_df.empty:
                    bound_tfs = set(peaks_df[peaks_df["score"] >= db.min_score_assign()]
                                     ["tf"].dropna().unique())
                corrs["binds_promoter"] = corrs["tf"].isin(bound_tfs)
                st.dataframe(
                    corrs[["tf", "r", "binds_promoter"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "tf": st.column_config.TextColumn(
                            "TF",
                            help="Transcription factor gene symbol."),
                        "r": st.column_config.NumberColumn(
                            "r", format="%.2f",
                            help="Pearson correlation between this TF's "
                                 "and the target transcript's TPM across "
                                 "the 66 GTEx tissues. + = co-expressed, "
                                 "− = anti-correlated."),
                        "binds_promoter": st.column_config.CheckboxColumn(
                            "binds promoter (assigned)",
                            help="True if this TF has at least one peak "
                                 "in the focal TSS's ±1.5 kb window with "
                                 "the assignment score. A binding-AND-correlation "
                                 "co-occurrence is the strongest "
                                 "tissue-level evidence."),
                    },
                )

    # ----- Modules table card -----------------------------------------------
    with st.container(border=True):
        st.markdown("### Modules in this promoter",
                     help=HELP_MODULES_TABLE + "  \n\n"
                          "**`r (module ↔ target)`**: Pearson r between the "
                          "module's mean TF TPM and the target's TPM across "
                          "the 66 GTEx tissues. High r = the bound TFs and "
                          "the gene co-vary across tissues — strongest "
                          "tissue-level evidence the module is a real "
                          "regulator.  \n"
                          "**`top driver TF`**: TF in this module with the "
                          "highest |r| against the target — likely the "
                          "individual factor doing the work.  \n"
                          "**`# TFs |r|≥0.5`**: how many of the module's "
                          "TFs co-vary strongly with the target.  \n"
                          "**`# supp. tissues`**: tissues where both the "
                          "module's TFs and the target sit in their "
                          "respective top quartile (Q3).")
        if modules_df.empty:
            st.info("No modules detected at this TSS (no chip-atlas "
                    "evidence in ±1.5 kb).")
        else:
            evidence = (db.gtex_module_target_evidence(tx_id)
                          if db.gtex_available() else pd.DataFrame())
            if not evidence.empty:
                merged = modules_df.merge(
                    evidence[["module_id", "r_module_target",
                              "top_driver_tf", "top_driver_r",
                              "n_tfs_high_r", "n_supporting_tissues",
                              "top_supporting_tissue"]],
                    on="module_id", how="left",
                )
            else:
                merged = modules_df.copy()
                for c in ("r_module_target", "top_driver_tf",
                          "top_driver_r", "n_tfs_high_r",
                          "n_supporting_tissues", "top_supporting_tissue"):
                    merged[c] = None

            # Derive driver-class label (no- / single- / multi-driver) from
            # n_tfs_high_r — scannable categorical next to the raw count.
            if "n_tfs_high_r" in merged.columns:
                merged["driver_class"] = db.driver_class_series(
                    merged["n_tfs_high_r"]
                )

            # Add DepMap essentiality of the assigned TFs per module.
            if db.depmap_available() and not peaks_df.empty:
                # For each module, compute median + best-TF Chronos across
                # the TFs the build actually assigned, within [lo, hi].
                hp = peaks_df[peaks_df["score"] >= db.min_score_assign()]
                # Index peaks once
                peaks_local = hp[["local_offset", "tf"]].dropna()
                tf_chronos_rows = []
                for _, m in merged.iterrows():
                    in_mod = peaks_local[
                        (peaks_local["local_offset"] >= int(m["lo_offset"])) &
                        (peaks_local["local_offset"] <= int(m["hi_offset"]))]
                    tfs = tuple(sorted(set(in_mod["tf"].astype(str))))
                    if not tfs:
                        tf_chronos_rows.append({"module_id": int(m["module_id"]),
                                                  "tf_median_chronos":  np.nan,
                                                  "tf_best_essential":  None,
                                                  "tf_best_chronos":    np.nan})
                        continue
                    s = db.depmap_gene_summary(tfs)
                    if s.empty:
                        tf_chronos_rows.append({"module_id": int(m["module_id"]),
                                                  "tf_median_chronos":  np.nan,
                                                  "tf_best_essential":  None,
                                                  "tf_best_chronos":    np.nan})
                        continue
                    med = float(np.median(s["median_chronos"]))
                    best_idx = int(s["median_chronos"].idxmin())
                    tf_chronos_rows.append({
                        "module_id": int(m["module_id"]),
                        "tf_median_chronos": round(med, 2),
                        "tf_best_essential": s.loc[best_idx, "gene"],
                        "tf_best_chronos":   round(float(s.loc[best_idx, "median_chronos"]), 2),
                    })
                tf_chronos_df = pd.DataFrame(tf_chronos_rows)
                merged = merged.merge(tf_chronos_df, on="module_id", how="left")
            else:
                for c in ("tf_median_chronos", "tf_best_essential",
                          "tf_best_chronos"):
                    merged[c] = None

            cols = ["module_id", "module_local_idx", "lo_offset",
                    "hi_offset", "center_offset", "width",
                    "n_tfs_supporting", "n_tfs_assigned",
                    "dominant_program", "dominant_weight",
                    "program_reading",
                    "r_module_target", "top_driver_tf", "top_driver_r",
                    "n_tfs_high_r", "driver_class",
                    "n_supporting_tissues",
                    "top_supporting_tissue",
                    "tf_median_chronos", "tf_best_essential",
                    "tf_best_chronos"]
            cols = [c for c in cols if c in merged.columns]
            st.dataframe(
                merged[cols],
                hide_index=True, width="stretch",
                column_config={
                    "module_id": st.column_config.NumberColumn(
                        "global_id", width="small",
                        help="Atlas-wide unique module id (0–76,998)."),
                    "module_local_idx": st.column_config.NumberColumn(
                        "local_idx", width="small",
                        help="Module's index within this transcript "
                             "(0-based, in genomic order)."),
                    "lo_offset": st.column_config.NumberColumn(
                        "lo (bp)", width="small",
                        help="Module's start position relative to the TSS "
                             "(txn-oriented bp; negative = upstream)."),
                    "hi_offset": st.column_config.NumberColumn(
                        "hi (bp)", width="small",
                        help="Module's end position relative to the TSS."),
                    "center_offset": st.column_config.NumberColumn(
                        "center (bp)", width="small",
                        help="Module's KDE-peak center, txn-oriented bp "
                             "from TSS."),
                    "width": st.column_config.NumberColumn(
                        "width (bp)", width="small",
                        help="Module width hi − lo."),
                    "n_tfs_supporting": st.column_config.NumberColumn(
                        "TFs (any)", width="small",
                        help="# distinct TFs with at least one peak "
                             "(any chip-atlas score) in this module."),
                    "n_tfs_assigned":   st.column_config.NumberColumn(
                        f"TFs (≥{db.min_score_assign()})", width="small",
                        help="# distinct TFs with at least one core "
                             "assigned-score peak in this module — the strict "
                             "TF assignment used downstream."),
                    "dominant_program": st.column_config.NumberColumn(
                        "program", width="small",
                        help="Program id (1–10) with the highest NMF "
                             "weight at this module."),
                    "dominant_weight":  st.column_config.NumberColumn(
                        "weight", format="%.3f", width="small",
                        help="NMF coefficient of the dominant program at "
                             "this module. 1.0 = pure single-program."),
                    "program_reading":  st.column_config.TextColumn(
                        "reading", width="medium",
                        help="Human label of the dominant program (see "
                             "Programs tab summary)."),
                    "r_module_target":  st.column_config.NumberColumn(
                        "r (mod↔target)", format="%.2f", width="small",
                        help="Pearson r between the module's mean TF TPM "
                             "and the target transcript's TPM across the "
                             "66 GTEx tissues. High r = TFs and gene "
                             "co-vary; strongest tissue-level evidence."),
                    "top_driver_tf":    st.column_config.TextColumn(
                        "top driver", width="small",
                        help="TF in this module with the highest |r| "
                             "against the target — the most likely "
                             "individual factor driving regulation."),
                    "top_driver_r":     st.column_config.NumberColumn(
                        "driver r", format="%.2f", width="small",
                        help="That top driver TF's r against the target."),
                    "n_tfs_high_r":     st.column_config.NumberColumn(
                        "# |r|≥0.5", width="small",
                        help="How many of the module's TFs co-vary with "
                             "the target above the |r|≥0.5 threshold."),
                    "driver_class":     st.column_config.TextColumn(
                        "driver class", width="small",
                        help="Categorical from `# |r|≥0.5`: "
                             "0 → no-driver (no TF co-varies with target); "
                             "1 → single-driver (one TF drives the signal); "
                             "≥2 → multi-driver (combinatorial / redundant)."),
                    "n_supporting_tissues": st.column_config.NumberColumn(
                        "# supp.", width="small",
                        help="Tissues where both the module's mean TF TPM "
                             "and the target sit in their respective top "
                             "quartile (Q3)."),
                    "top_supporting_tissue": st.column_config.TextColumn(
                        "top supp. tissue", width="medium",
                        help="The supporting tissue with the highest "
                             "joint Q3 evidence."),
                    "tf_median_chronos":  st.column_config.NumberColumn(
                        "med. Chronos", format="%.2f", width="small",
                        help="Median DepMap Chronos across the module's "
                             "assigned-score TFs. < −1 = essential."),
                    "tf_best_essential":  st.column_config.TextColumn(
                        "best TF", width="small",
                        help="TF in this module with the most-negative "
                             "median Chronos — most likely to be a "
                             "selectable target."),
                    "tf_best_chronos":    st.column_config.NumberColumn(
                        "best Chronos", format="%.2f", width="small",
                        help="That TF's median Chronos across all DepMap "
                             "cell lines."),
                },
            )

            # Per-TF-in-module driver detail
            if not evidence.empty:
                with st.expander("Per-TF correlation drivers within each "
                                  "module", expanded=False):
                    st.caption(
                        "For each module, the per-TF Pearson r against the "
                        "target's tissue-expression profile. Sorted by |r|. "
                        "Strong-positive TFs are the most likely drivers; "
                        "near-zero TFs are 'passengers' that bind but don't "
                        "co-vary across tissues."
                    )
                    pick_mid = st.selectbox(
                        "Module",
                        options=merged["module_id"].tolist(),
                        format_func=lambda mid: _module_label(merged, mid),
                        help="Pick a module to inspect the per-TF "
                             "correlation breakdown. Labels show: "
                             "M<id> (center bp · dominant program · "
                             "module-target r).",
                    )
                    drivers = db.gtex_module_tf_evidence(int(pick_mid))
                    if drivers.empty:
                        st.info("No driver TFs found.")
                    else:
                        st.dataframe(
                            drivers, hide_index=True,
                            width="stretch",
                            column_config={
                                "tf": st.column_config.TextColumn(
                                    "TF",
                                    help="TF in this module."),
                                "r": st.column_config.NumberColumn(
                                    "r (TF ↔ target)", format="%.2f",
                                    help="Pearson r between this TF's TPM "
                                         "and the focal target's TPM across "
                                         "the 66 GTEx tissues. Strong + r "
                                         "names the most plausible "
                                         "individual driver."),
                            },
                        )

    # ----- TF-target essentiality coupling (DepMap, across cell lines) -----
    if db.depmap_tf_target_corr_available() and not peaks_df.empty:
        with st.container(border=True):
            st.markdown(
                "### TF–target essentiality coupling (DepMap)",
                help=(
                    "Pearson correlation between each TF's CRISPR Chronos "
                    "essentiality score and this gene's expression "
                    "(log10(TPM+1)), computed across the ~1,000 DepMap "
                    "cell lines that have both measurements.  \n\n"
                    "**Strong negative r** = cells that highly express "
                    "the target also depend on the TF for survival — "
                    "the cleanest mechanistic signal that the TF is "
                    "regulating the target. **Strong positive r** = the "
                    "TF is more dispensable in cells with high target "
                    "expression (often saturation / paralog redundancy). "
                    "Complementary to the GTEx tissue-level correlation: "
                    "DepMap adds perturbation-validated evidence across "
                    "cell-line genetic backgrounds."
                ),
            )
            target_gene = tss_meta["gene_name"]
            bound_tfs = tuple(sorted(set(
                peaks_df.loc[peaks_df["score"] >= db.min_score_assign(), "tf"]
                        .dropna().astype(str)
            )))
            if not bound_tfs:
                st.info("No assigned-score TFs in this TSS's window.")
            else:
                with st.spinner("Computing TF×target correlations…"):
                    corr_df = db.depmap_tf_target_correlation(
                        target_gene, bound_tfs,
                    )
                if corr_df.empty:
                    st.info(
                        f"No correlations available — `{target_gene}` may be "
                        "absent from the DepMap expression matrix, or fewer "
                        "than 50 cell lines have both measurements."
                    )
                else:
                    n_cols = st.columns([1, 2])
                    with n_cols[0]:
                        top_n = st.slider(
                            "Top N TFs", min_value=10,
                            max_value=min(40, len(corr_df)),
                            value=min(20, len(corr_df)),
                            step=5,
                            help="Sorted by |r| descending.",
                            key=f"depmap_tt_top_{tx_id}",
                        )
                        st.dataframe(
                            corr_df.head(top_n),
                            hide_index=True, width="stretch",
                            column_config={
                                "tf":           st.column_config.TextColumn(
                                    "TF",
                                    help="TF that binds this TSS at "
                                         "the assignment score. Only TFs in DepMap's "
                                         "Chronos panel are shown."),
                                "r":            st.column_config.NumberColumn(
                                    "r", format="%.2f",
                                    help="Pearson r between this TF's "
                                         "Chronos essentiality and the "
                                         "target's log10(TPM+1) across the "
                                         "common DepMap cell lines. Strong "
                                         "− r = cells expressing the "
                                         "target also depend on the TF "
                                         "(cleanest mechanistic signal)."),
                                "n_cell_lines": st.column_config.NumberColumn(
                                    "n cell lines",
                                    help="# DepMap cell lines with paired "
                                         "non-NaN Chronos + expression for "
                                         "this (TF, target) pair."),
                            },
                            height=min(380, 35 * top_n + 60),
                        )
                    with n_cols[1]:
                        st.plotly_chart(
                            plotting.fig_depmap_tf_target_corr_bar(
                                corr_df, target_gene, top_n=top_n,
                            ),
                            width="stretch",
                        )

                    # Drill into one TF — per-cell-line scatter requires
                    # the raw CSVs to be present (precomputed shards only
                    # carry summary r + n, not per-cell-line values).
                    if db.depmap_raw_available():
                        tf_options = corr_df.head(top_n)["tf"].tolist()
                        pick_tf = st.selectbox(
                            "Inspect one TF — per-cell-line scatter",
                            options=tf_options, index=0,
                            help="Each point is one DepMap cell line. Look "
                                 "for lineage-driven clusters in addition "
                                 "to the global trend.",
                            key=f"depmap_tt_scatter_{tx_id}",
                        )
                        pair_df = db.depmap_tf_target_pair_values(
                            target_gene, pick_tf,
                        )
                        r_pick = float(
                            corr_df.loc[corr_df["tf"] == pick_tf, "r"].iloc[0]
                        )
                        st.plotly_chart(
                            plotting.fig_depmap_tf_target_scatter(
                                pair_df, pick_tf, target_gene, r=r_pick,
                            ),
                            width="stretch",
                        )
                    else:
                        st.caption(
                            "Per-cell-line scatter not available — raw "
                            "DepMap CSVs not present on this host."
                        )

    # ----- TF list expander --------------------------------------------------
    with st.expander("All TFs binding this TSS (filtered table)",
                      expanded=False):
        st.caption(
            f"Distinct TFs with peaks at score ∈ [{smin}, {smax}] in this "
            "TSS's ±1.5 kb window. `n_peaks` is the number of recentered "
            "25-nt peak blocks for that TF here."
        )
        if peaks_df.empty:
            st.info("No peaks in window.")
        else:
            tf_summary = (peaks_df[(peaks_df["score"] >= smin)
                                    & (peaks_df["score"] <= smax)]
                            .groupby("tf")
                            .agg(n_peaks=("score", "size"),
                                 max_score=("score", "max"),
                                 min_offset=("local_offset", "min"),
                                 max_offset=("local_offset", "max"))
                            .reset_index()
                            .sort_values("n_peaks", ascending=False))
            st.dataframe(
                tf_summary, hide_index=True, width="stretch",
                column_config={
                    "tf": st.column_config.TextColumn(
                        "TF",
                        help="Transcription factor, by ChIP-Atlas antigen "
                             "name."),
                    "n_peaks": st.column_config.NumberColumn(
                        "# peaks", format="%d",
                        help="Recentered 25-nt peak blocks for this TF in "
                             "this window, within the score range above. One "
                             "TF can have several peaks at one promoter."),
                    "max_score": st.column_config.NumberColumn(
                        "best score", format="%d",
                        help="Highest ChIP-Atlas score among them. The score "
                             "IS the q-value in another unit, so a higher "
                             "number is stronger evidence of binding, not "
                             "more binding."),
                    "min_offset": st.column_config.NumberColumn(
                        "earliest bp", format="%d",
                        help="Most upstream peak position, in "
                             "transcription-oriented bp from the TSS. "
                             "Negative is 5′ of the TSS."),
                    "max_offset": st.column_config.NumberColumn(
                        "latest bp", format="%d",
                        help="Most downstream peak position, same "
                             "convention — positive is 3′ of the TSS."),
                })

    # ----- Downloads --------------------------------------------------------
    with st.expander("Download this transcript's data", expanded=False):
        if not modules_df.empty:
            st.download_button(
                f"Modules for {tx_id} (TSV)",
                data=modules_df.to_csv(sep="\t", index=False).encode(),
                file_name=f"{tx_id}_modules.tsv",
                mime="text/tab-separated-values",
            )
        if not peaks_df.empty:
            st.download_button(
                f"Peaks for {tx_id} (TSV, all scores)",
                data=peaks_df.to_csv(sep="\t", index=False).encode(),
                file_name=f"{tx_id}_peaks.tsv",
                mime="text/tab-separated-values",
            )


def _render_program_card(program: int, reading: str, n_modules: int) -> None:
    """A clickable popover card per program present at the focal TSS,
    with inline top TFs + top GO + a 'open in Programs tab' button that
    preselects via session_state."""
    color = plotting.PROGRAM_COLORS[(program - 1) % 10]
    label = f"P{program} — {reading}  ·  {n_modules} module{'s' if n_modules != 1 else ''}"
    with st.popover(label, width="stretch"):
        st.markdown(
            f"<div style='border-left:6px solid {color};padding-left:10px;'>"
            f"<b>P{program}</b> — {reading}"
            f"</div>",
            unsafe_allow_html=True,
        )

        # Top TFs
        ttf = db.get_program_top_tfs(program, limit=10)
        st.markdown("**Top TFs (H loading)**")
        if ttf.empty:
            st.caption("—")
        else:
            ttf_disp = ttf.head(10).copy()
            ttf_disp["loading"] = ttf_disp["loading"].round(3)
            st.dataframe(ttf_disp[["rank", "tf", "loading"]],
                          hide_index=True, width="stretch",
                          column_config=ui.PROGRAM_TF_COLUMNS)

        # Top GO terms
        go = db.get_program_top_go(program, limit=5)
        st.markdown("**Top GO BP terms (genome bg, q < 0.05)**")
        if go.empty:
            st.caption("—")
        else:
            go["term"] = (go["term"]
                            .str.replace("GOBP_", "")
                            .str.replace("_", " ").str.lower())
            go_disp = go[["term", "fg_in", "set_size_in_bg",
                           "odds_ratio", "q_value"]].copy()
            go_disp["odds_ratio"] = go_disp["odds_ratio"].round(2)
            go_disp["q_value"]    = go_disp["q_value"].apply(
                lambda x: f"{x:.1e}")
            st.dataframe(go_disp, hide_index=True,
                          width="stretch",
                          column_config={
                              "term": st.column_config.TextColumn(
                                  "term", width="medium",
                                  help="GO biological-process term, GOBP_ "
                                       "prefix stripped."),
                              "fg_in": st.column_config.NumberColumn(
                                  "fg", format="%d",
                                  help="Genes of this program's promoters "
                                       "that are in the term."),
                              "set_size_in_bg": st.column_config.NumberColumn(
                                  "term size", format="%d",
                                  help="Genes in the term overall, within the "
                                       "background. Read with `fg` — 4/12 and "
                                       "4/900 are not the same evidence."),
                              "odds_ratio": st.column_config.NumberColumn(
                                  "odds", format="%.2f",
                                  help="Odds ratio of the overlap against a "
                                       "genome background."),
                              "q_value": st.column_config.TextColumn(
                                  "q",
                                  help="Benjamini-Hochberg FDR. Only terms "
                                       "under 0.05 are listed."),
                          })

        if st.button(f"Open P{program} in the Programs page",
                      key=f"open_p{program}",
                      help="Switches to the Programs & Modules page with "
                           "this program preselected."):
            st.session_state["prog_pick"] = program
            nav.goto("programs")


def _render_neighbourhood(tss_meta) -> None:
    """Genome-wide elements around this gene, at quantile zoom levels."""
    if not db.has_genome_layer():
        return
    gene = (tss_meta or {}).get("gene_name")
    if not gene:
        return
    el = db.get_elements_for_gene(gene)
    if el.empty:
        return

    n_far = int((el.stratum != "promoter").sum())
    if n_far == 0:
        # 789 genes (4.2%) have only promoter-stratum elements. Drawing a
        # neighbourhood plot for them is worse than drawing nothing: the zoom
        # collapses to a span narrower than the promoter box, so the shading
        # covers the whole canvas and one or two markers sit against it. Say
        # the fact in a line instead.
        n_el = len(el)
        with st.container(border=True):
            st.markdown("### Beyond the promoter window")
            st.caption(
                f"No proximal or distal elements for this gene — all "
                f"{n_el} of its genome-wide element{'s' if n_el != 1 else ''} "
                f"fall inside the promoter window. This is the case for 789 "
                f"of 18,984 genes (4.2%)."
            )
        return
    with st.container(border=True):
        st.markdown(
            f"### Beyond the promoter window — {n_far:,} further elements",
            help="The profile above covers +/-1.5 kb around the canonical TSS. "
                 "These are elements found genome-wide without reference to "
                 "genes, then labelled by distance. Zoom levels are quantiles "
                 "of THIS gene's element distances rather than fixed widths, "
                 "so they adapt to genes that sprawl and genes that do not; "
                 "the promoter region stays boxed at every level.",
        )
        levels = plotting.gene_zoom_levels(el.dist_to_tss)
        if len(levels) < 2:
            st.caption("Only promoter-proximal elements here.")
            return
        names = [L["label"] for L in levels]
        pick = st.radio("zoom", names, index=len(names) - 1, horizontal=True,
                        key=f"nbhd_zoom_{gene}",
                        help="Percentages are the share of this gene's "
                             "elements brought into view, not a fixed span.")
        L = levels[names.index(pick)]
        st.plotly_chart(
            plotting.fig_gene_neighbourhood(
                el, L["lo"], L["hi"], gene=gene,
                family_labels=db.get_family_labels()),
            width="stretch", theme=None)

        amb = int((el.n_tss_comparably_close >= 2).sum())
        distal = el[el.stratum == "distal"]
        if len(distal):
            amb_d = float((distal.n_tss_comparably_close >= 2).mean() * 100)
            st.caption(
                f"Elements are attributed to their NEAREST canonical TSS, "
                f"which is a locator and not a regulatory assignment. "
                f"{amb_d:.0f}% of this gene's {len(distal):,} distal elements "
                f"have another TSS within twice the distance ({amb:,} of "
                f"{len(el):,} elements overall). Hollow markers are programs "
                f"that are not substantive — fewer than 100 elements or seed "
                f"stability below 0.90."
            )


def _module_label(merged, mid) -> str:
    """Label for the module picker, tolerant of a NULL program.

    dominant_program is NULL on a build whose programs come from the genome
    layer, and int(NA) raises -- which would break the selectbox rather than
    just omitting a letter.
    """
    row = merged.loc[merged["module_id"] == mid]
    if row.empty:
        return f"M{int(mid)}"
    r = row.iloc[0]
    parts = [f"M{int(mid)}", f"{int(r['center_offset']):+d} bp"]
    if pd.notna(r.get("dominant_program")):
        parts.append(f"P{int(r['dominant_program'])}")
    if pd.notna(r.get("r_module_target")):
        parts.append(f"r={r['r_module_target']:.2f}")
    return f"{parts[0]}  ({' · '.join(parts[1:])})"


def _render_module_composition(modules_df, peaks_df) -> None:
    """Which TFs make up each module, straight from the peaks.

    The modules table alone is sparse -- widths, counts and a scatter of NULL
    program columns. The detail that answers "what IS this module" used to sit
    only inside the GTEx driver expander, which is gated on
    module_target_correlation.parquet; that file is not built for every tier,
    so on a build without it the whole thing disappears and the module becomes
    an unexplained row.

    Composition needs no GTEx at all. A module is the TFs with peaks inside its
    [lo_offset, hi_offset] window, and peaks are already loaded for the profile
    above. Correlation drivers are a separate, richer question that does need
    the parquet -- this is the part that should always be available.
    """
    if modules_df.empty or peaks_df.empty:
        return
    thr = db.min_score_assign()

    # What each module maps to in the genome layer. A module on its own is a
    # span and some counts; through the element at the same locus it inherits a
    # program, a family and the family's enrichment label, which is what says
    # what it is.
    ann = db.get_module_annotation(str(modules_df.attrs.get("transcript_id", "")) )
    with st.container(border=True):
        st.markdown(
            "### What's in each module",
            help=f"TFs with a peak inside the module's window. `assigned` "
                 f"marks those at score ≥ {thr}, the threshold this build used "
                 f"to assign a TF to a module -- the rest bind but did not "
                 f"clear it. Positions are bp from the TSS.",
        )
        if ann is not None and not ann.empty:
            show = ann[["module_id", "center_offset", "n_tfs_assigned",
                        "program", "family_label", "program_tfs",
                        "substantive", "match_bp"]].copy()
            n_link = int(show.program.notna().sum())
            st.caption(
                f"{n_link} of {len(show)} modules map to a genome-wide element "
                f"and inherit its program and family. The rest fall below the "
                f"genome support floor of 11 assigned TFs — the promoter "
                f"pipeline keeps modules that floor rejects, so a blank here "
                f"is a statement about evidence, not a missing lookup."
            )
            st.dataframe(
                show.rename(columns={
                    "module_id": "module", "center_offset": "center (bp)",
                    "n_tfs_assigned": "TFs", "program": "program",
                    "family_label": "family", "program_tfs": "program TFs",
                    "substantive": "substantive", "match_bp": "offset (bp)"}),
                hide_index=True, use_container_width=True,
                column_config={
                    "module": st.column_config.NumberColumn(
                        "module", format="%d",
                        help="Module id in this build. Ids are not stable "
                             "across builds."),
                    "center (bp)": st.column_config.NumberColumn(
                        "center (bp)", format="%d",
                        help="Module midpoint in transcription-oriented bp "
                             "from the TSS — negative is 5′, positive 3′."),
                    "TFs": st.column_config.NumberColumn(
                        "TFs", format="%d",
                        help=f"Distinct TFs assigned to the module, i.e. with "
                             f"a peak at score ≥ {thr} inside it."),
                    "program": st.column_config.NumberColumn(
                        "program", format="%d",
                        help="Genome-wide program inherited from the element "
                             "at the same locus. Blank means the module has "
                             "no matching element — it sits below the genome "
                             "support floor of 11 assigned TFs, which the "
                             "promoter pipeline does not apply. A blank is a "
                             "statement about evidence, not a failed lookup."),
                    "family": st.column_config.TextColumn(
                        "family", width="medium",
                        help="The program's family label, from MSigDB "
                             "enrichment with an FDR behind it."),
                    "program TFs": st.column_config.TextColumn(
                        "program TFs", width="large",
                        help="Top-loading TFs of that program. These are the "
                             "program's signature genome-wide, NOT the TFs "
                             "found in this particular module — compare them "
                             "against the per-module list below."),
                    "substantive": st.column_config.CheckboxColumn(
                        "substantive",
                        help="Whether the inherited program has ≥100 elements "
                             "and seed stability ≥0.90. An unticked box means "
                             "the label rests on a program pinned to very few "
                             "elements."),
                    "offset (bp)": st.column_config.NumberColumn(
                        "offset (bp)", format="%d",
                        help="Distance between the module's center and the "
                             "matched element's center. The regression gate "
                             "recovered 98.2% of comparable modules at a "
                             "12 bp median offset, so large values here are "
                             "worth a second look."),
                })

        labels = {
            int(r.module_id): (f"M{int(r.module_id)}  ({int(r.center_offset):+d} bp"
                               f" · {int(r.n_tfs_assigned)} assigned"
                               f" · {int(r.width)} bp wide)")
            for _, r in modules_df.iterrows()}
        mid = st.selectbox("Module", options=list(labels),
                           format_func=lambda m: labels[m],
                           key="mod_comp_pick")
        m = modules_df[modules_df.module_id == mid].iloc[0]
        inside = peaks_df[(peaks_df.local_offset >= int(m.lo_offset))
                          & (peaks_df.local_offset <= int(m.hi_offset))]
        if inside.empty:
            st.caption("No peaks inside this module's window.")
            return
        per_tf = (inside.groupby("tf")
                  .agg(peaks=("score", "size"), best_score=("score", "max"),
                       median_offset=("local_offset", "median"))
                  .reset_index())
        per_tf["assigned"] = per_tf.best_score >= thr
        per_tf = per_tf.sort_values(["assigned", "best_score"],
                                    ascending=[False, False])
        n_asg = int(per_tf.assigned.sum())
        st.caption(
            f"M{int(mid)} spans {int(m.lo_offset):+d} to {int(m.hi_offset):+d} bp "
            f"and contains {len(inside):,} peaks from {len(per_tf)} TFs — "
            f"**{n_asg} assigned** at score ≥ {thr}, {len(per_tf) - n_asg} below it."
        )
        st.dataframe(
            per_tf.rename(columns={"tf": "TF", "peaks": "# peaks",
                                   "best_score": "best score",
                                   "median_offset": "median bp"}),
            hide_index=True, use_container_width=True,
            column_config={
                "TF": st.column_config.TextColumn(
                    "TF", help="Transcription factor with at least one peak "
                               "inside this module's window."),
                "# peaks": st.column_config.NumberColumn(
                    "# peaks", format="%d",
                    help="Recentered 25-nt peak blocks for this TF inside the "
                         "module."),
                "best score": st.column_config.NumberColumn(
                    "best score", format="%d",
                    help=f"Highest ChIP-Atlas score among them — this is what "
                         f"the ≥ {thr} assignment test is applied to. The "
                         f"score is the q-value in another unit, so it "
                         f"measures confidence that the TF binds, not how "
                         f"much of it binds."),
                "median bp": st.column_config.NumberColumn(
                    "median bp", format="%d",
                    help="Median peak position in transcription-oriented bp "
                         "from the TSS."),
                "assigned": st.column_config.CheckboxColumn(
                    "assigned",
                    help=f"Whether this TF cleared score {thr} and therefore "
                         f"counts toward the module's TF total. Unticked TFs "
                         f"do bind here — they just did not clear the "
                         f"threshold this build assigns at."),
            })
