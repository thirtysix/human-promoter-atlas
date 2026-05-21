"""GO-term reverse search: enter a biological process, see the programs,
archetypes, and genes that enrich for it."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import db, ui, nav


HELP_INTRO = (
    "All foreground/background tests used the MSigDB c5.go.bp universe "
    "(~18,000 genes) as the background, hypergeometric test + BH-FDR. "
    "Only terms with at least one significant program-level or "
    "archetype-level enrichment are searchable here."
)

HELP_PROG_TBL = (
    "Programs whose dominantly-assigned gene set is enriched for this GO BP "
    "term. `OR` = odds ratio over the genome background; `fg/bg` is the "
    "overlap size and the term's size in the background; `q` is BH-FDR "
    "adjusted p."
)

HELP_ARCH_TBL = (
    "Archetypes whose member gene set is enriched for this GO BP term. "
    "Same statistic as the programs table."
)

HELP_GENES = (
    "Union of `genes_in_overlap` across all programs/archetypes that hit "
    "this term. Frequency = number of program+archetype hits the gene "
    "contributed to. Click any gene to open its Per-transcript page."
)


def _pretty(term: str) -> str:
    """GOBP_RIBOSOME_BIOGENESIS -> 'ribosome biogenesis'."""
    return term.replace("GOBP_", "").replace("_", " ").lower()


def render() -> None:
    ui.intro_card(
        title="Reverse search by GO term",
        what="Type a biological process (or pick from autocomplete) and "
             "see every k=10 program + archetype enriched for it, plus "
             "the genes that drove each hit.",
        objective="Answer *'which regulatory programs implement this "
                   "biology?'* — the inverse of the per-program / "
                   "per-archetype views.",
        significance="Natural entry point for visitors who know the "
                      "biology but not the regulator: arrive with "
                      "*'ribosome biogenesis'* or *'immune system'*, "
                      "leave with P10 / A1 / a gene list to drill into.",
    )

    terms_df = db.list_go_terms()
    if terms_df.empty:
        st.warning("No GO terms are loaded — the program/archetype "
                    "enrichment tables appear empty.")
        return

    # Pretty-name mapping (raw <-> display)
    raw_to_pretty = dict(zip(terms_df["term"], terms_df["term"].map(_pretty)))
    pretty_to_raw = {v: k for k, v in raw_to_pretty.items()}

    # ---- Search widget ---------------------------------------------------
    with st.container(border=True):
        st.markdown("### Pick a GO BP term", help=HELP_INTRO)
        # Seed from URL ?go= or session state
        preset = st.session_state.get("go_search_preset", "")
        options = [""] + sorted(pretty_to_raw)
        default_idx = options.index(preset) if preset in pretty_to_raw else 0
        choice = st.selectbox(
            "Type to search (147 terms enriched somewhere in the atlas)",
            options=options,
            index=default_idx,
            placeholder="e.g. ribosome biogenesis, cell cycle, immune…",
            key="go_search_select",
            help="Start typing — Streamlit filters in real-time.",
        )

        # Top-of-table teaser when nothing selected
        if not choice:
            st.markdown("**Most-shared terms** (high cross-program "
                         "overlap = broadly used biology):")
            teaser = terms_df.head(10).copy()
            teaser["term"] = teaser["term"].map(_pretty)
            st.dataframe(
                teaser[["term", "go_id", "n_programs", "n_archetypes",
                         "min_q"]],
                hide_index=True, width="stretch",
                column_config={
                    "term": st.column_config.TextColumn(
                        "GO BP term",
                        help="GO Biological Process term (prettified — "
                             "GOBP_ prefix and underscores removed)."),
                    "go_id": st.column_config.TextColumn(
                        "GO ID",
                        help="Gene Ontology canonical ID."),
                    "n_programs":   st.column_config.NumberColumn(
                        "# progs",
                        help="# of the 10 programs that enrich for this "
                             "GO term in their top-15 hits."),
                    "n_archetypes": st.column_config.NumberColumn(
                        "# arch",
                        help="# archetypes that enrich for this GO term "
                             "in their top-15 hits."),
                    "min_q": st.column_config.NumberColumn(
                        "best q-value", format="%.1e",
                        help="Minimum BH-FDR q-value across all "
                             "programs and archetypes that hit this term."),
                },
            )
            st.info("Pick a term above to see which programs / archetypes / "
                    "genes drive that enrichment.")
            with st.container(border=True):
                st.markdown("##### What you'll see once a term is selected")
                p1, p2, p3 = st.columns(3)
                with p1:
                    st.markdown("**Programs that enrich**")
                    st.caption("Which of the 10 k=10 programs hit this GO "
                                "term in their top-15 enrichments — with "
                                "OR, q-value, and a deep-link into Programs.")
                with p2:
                    st.markdown("**Archetypes that enrich**")
                    st.caption("Which of the 8 archetypes are GO-hit by "
                                "this term — same OR + q + deep-link, but "
                                "at gene-level granularity.")
                with p3:
                    st.markdown("**Driver genes**")
                    st.caption("The overlap genes that drove the hits — "
                                "click any to jump to its Per-transcript "
                                "view.")
            return

    term = pretty_to_raw[choice]
    st.session_state["go_search_preset"] = choice

    # ---- Programs that enrich for this term ------------------------------
    progs = db.programs_for_go_term(term)
    archs = db.archetypes_for_go_term(term)

    head_meta = terms_df.loc[terms_df["term"] == term].iloc[0]
    st.markdown(
        f"#### {_pretty(term)}  "
        f"<span style='color:#888;font-weight:normal;font-size:0.85em'>"
        f"· <code>{head_meta['go_id']}</code></span>",
        unsafe_allow_html=True,
    )
    # Thin colored bar + three metric cards — matches the pattern used
    # across the rest of the site (Per-transcript, Archetypes, Compare).
    st.markdown(
        "<div style='height:6px;background:#0b6e4f;"
        "border-radius:3px;margin:6px 0 8px 0'></div>",
        unsafe_allow_html=True,
    )
    m1, m2, m3 = st.columns(3)
    m1.metric("Programs hit", f"{len(progs)}",
               help="# of the 10 k=10 programs whose top-15 GO BP "
                    "enrichment includes this term.")
    m2.metric("Archetypes hit", f"{len(archs)}",
               help="# gene-level archetypes whose top-15 GO BP "
                    "enrichment includes this term.")
    m3.metric("Best q-value", f"{head_meta['min_q']:.1e}",
               help="Minimum BH-FDR q-value across every program × "
                    "term and archetype × term hit for this term.")

    col_p, col_a = st.columns(2)

    # ---- Programs panel --------------------------------------------------
    with col_p:
        with st.container(border=True):
            st.markdown("### Programs", help=HELP_PROG_TBL)
            if progs.empty:
                st.info("No program-level enrichment for this term.")
            else:
                disp = progs.copy()
                disp["bg"] = (disp["fg_in"].astype(str) + " / "
                                + disp["set_size_in_bg"].astype(str))
                disp = disp.rename(columns={"program": "P"})
                st.dataframe(
                    disp[["P", "rank", "odds_ratio", "q_value", "bg"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "P": st.column_config.NumberColumn(
                            "program",
                            help="Program id (1–10) that enriches for "
                                 "this GO term."),
                        "rank": st.column_config.NumberColumn(
                            "rank",
                            help="Rank of this term within the program's "
                                 "top-15 GO hits."),
                        "odds_ratio": st.column_config.NumberColumn(
                            "OR", format="%.2f",
                            help="Odds ratio of program-gene set "
                                 "intersection with this GO term."),
                        "q_value": st.column_config.NumberColumn(
                            "q", format="%.1e",
                            help="BH-FDR adjusted hypergeometric p-value."),
                        "bg": st.column_config.TextColumn(
                            "fg / bg",
                            help="Foreground intersection / GO term size. "
                                 "fg = # program genes annotated to term; "
                                 "bg = total # MSigDB genes annotated."),
                    },
                )
                _open_buttons("program", progs["program"].astype(int).tolist(),
                              prefix="P", goto_page="programs")

    # ---- Archetypes panel ------------------------------------------------
    with col_a:
        with st.container(border=True):
            st.markdown("### Archetypes", help=HELP_ARCH_TBL)
            if archs.empty:
                st.info("No archetype-level enrichment for this term.")
            else:
                disp = archs.copy()
                disp["bg"] = (disp["fg_in"].astype(str) + " / "
                                + disp["set_size_in_bg"].astype(str))
                disp = disp.rename(columns={"archetype": "A"})
                st.dataframe(
                    disp[["A", "rank", "odds_ratio", "q_value", "bg"]],
                    hide_index=True, width="stretch",
                    column_config={
                        "A": st.column_config.NumberColumn(
                            "archetype",
                            help="Archetype id that enriches for this "
                                 "GO term."),
                        "rank": st.column_config.NumberColumn(
                            "rank",
                            help="Rank of this term within the "
                                 "archetype's top-15 GO hits."),
                        "odds_ratio": st.column_config.NumberColumn(
                            "OR", format="%.2f",
                            help="Odds ratio of archetype-gene set "
                                 "intersection with this GO term."),
                        "q_value": st.column_config.NumberColumn(
                            "q", format="%.1e",
                            help="BH-FDR adjusted hypergeometric p-value."),
                        "bg": st.column_config.TextColumn(
                            "fg / bg",
                            help="Foreground intersection / GO term size."),
                    },
                )
                _open_buttons("archetype",
                              archs["archetype"].astype(int).tolist(),
                              prefix="A", goto_page="archetypes")

    # ---- Genes panel -----------------------------------------------------
    with st.container(border=True):
        st.markdown("### Genes that drove these hits", help=HELP_GENES)
        gene_freq: dict[str, int] = {}
        for src in (progs, archs):
            if src.empty:
                continue
            for s in src["genes_in_overlap"].fillna(""):
                for g in (g.strip() for g in s.split(",")):
                    if g:
                        gene_freq[g] = gene_freq.get(g, 0) + 1
        if not gene_freq:
            st.info("No overlap genes recorded for this term.")
            return
        # Restrict to genes that exist in our canonical-transcript table
        all_genes = set(db.list_genes())
        rows = sorted(
            ((g, n) for g, n in gene_freq.items() if g in all_genes),
            key=lambda x: (-x[1], x[0]),
        )
        df = pd.DataFrame(rows, columns=["gene", "n_hits"])
        st.caption(
            f"{len(df)} unique gene{'s' if len(df) != 1 else ''} drove the "
            f"{len(progs) + len(archs)} program/archetype hit"
            f"{'s' if (len(progs)+len(archs)) != 1 else ''}. "
            "Pick one to open its Per-transcript page."
        )
        # Show top N with hit-counts
        top = df.head(40)
        st.dataframe(
            top, hide_index=True, width="stretch",
            column_config={
                "gene": st.column_config.TextColumn(
                    "gene symbol",
                    help="Gene that contributed to one or more "
                         "program/archetype × GO term overlaps."),
                "n_hits": st.column_config.NumberColumn(
                    "# (prog+arch) hits",
                    help="In how many program-or-archetype hits this "
                         "gene appears (as part of the overlap with "
                         "the GO term). Higher = more central to the "
                         "term's signal."),
            },
        )
        # Quick-open: gene selectbox + button
        pick = st.selectbox(
            "Open a gene:", options=[""] + df["gene"].tolist(),
            index=0, key=f"go_gene_open_{term}",
            placeholder="Type to search…",
            help="Jumps to the Per-transcript page for the selected "
                 "gene with the gene preselected.",
        )
        if pick:
            st.session_state["tx_gene_select"] = pick
            nav.goto("transcript")

    # ---- Downloads --------------------------------------------------------
    with st.expander("Download data", expanded=False):
        if not progs.empty:
            st.download_button(
                f"Programs hit by `{term}` — TSV",
                data=progs.to_csv(sep="\t", index=False).encode(),
                file_name=f"go_{term}_programs.tsv",
                mime="text/tab-separated-values",
                key=f"go_dl_progs_{term}",
            )
        if not archs.empty:
            st.download_button(
                f"Archetypes hit by `{term}` — TSV",
                data=archs.to_csv(sep="\t", index=False).encode(),
                file_name=f"go_{term}_archetypes.tsv",
                mime="text/tab-separated-values",
                key=f"go_dl_archs_{term}",
            )
        if not df.empty:
            st.download_button(
                f"Genes driving `{term}` hits — TSV",
                data=df.to_csv(sep="\t", index=False).encode(),
                file_name=f"go_{term}_genes.tsv",
                mime="text/tab-separated-values",
                key=f"go_dl_genes_{term}",
            )


def _open_buttons(kind: str, ids: list[int], prefix: str,
                   goto_page: str) -> None:
    """Render a row of small buttons that jump to the target page with the
    given program/archetype preselected."""
    if not ids:
        return
    cols = st.columns(min(len(ids), 5))
    for i, pid in enumerate(ids):
        with cols[i % len(cols)]:
            if st.button(f"Open {prefix}{pid} →",
                          key=f"go_open_{kind}_{pid}",
                          use_container_width=True):
                if kind == "program":
                    st.session_state["preselected_program"] = int(pid)
                # archetype selection is local to its tab; no global key
                nav.goto(goto_page)
