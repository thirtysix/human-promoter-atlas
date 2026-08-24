"""Program families — the browsable vocabulary over the genome-wide programs.

Replaces the gene-level archetype view. That view assumed genes have
concentrated program compositions, which is true at k=10 and measurably false
at k=140: diag_gene_coherence.py found gene identity carries no information
about an element's program once genomic distance is controlled (same-gene
element pairs score 0.2006 mean cosine against 0.2243 for different-gene pairs
at matched separation), and per-gene composition is near-flat (median
normalised spread 0.851, top program holding a median 5.4% of a gene's mass).

Families are the layer that IS supported: 140 programs grouped into 28 by
co-occurrence across elements, named by enrichment against MSigDB with an FDR
on every label. The previous gene-archetype implementation is in git history.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import db, ui


def render() -> None:
    ui.intro_card(
        title="Program families — grouping the 140 genome-wide programs",
        what="The k=140 programs are grouped into **28 families** by how they "
             "**co-occur across elements** — do two programs mark the same "
             "places — and each family is named by enrichment against MSigDB "
             "GO, with an FDR on every label.",
        objective="Give the 140 programs a vocabulary. Browse by family, then "
                   "drill to the programs and the evidence behind each name.",
        significance="Grouping by shared TFs does **not** work here: NMF drives "
                      "components onto disjoint factor sets, and the complexes "
                      "worth grouping share almost no subunits — PRC2 and "
                      "PRC1.1 have a TF-loading cosine of 0.009. What they "
                      "share is location, and co-occurrence finds it "
                      "(r = +0.256). It also reunites the AP-1 programs that "
                      "k=140 split apart (r = +0.664).",
    )

    if not db.has_genome_layer():
        st.info(
            "This database predates the genome-wide layer. Rebuild with "
            "`data/build_app_db_genome.py` to browse program families."
        )
        return

    fams = db.get_program_families()
    named = int(fams["named"].sum()) if "named" in fams.columns else 0

    st.caption(
        f"{len(fams)} families over 140 programs · {named} named by enrichment "
        f"(FDR ≤ 0.05) · unnamed families are shown by their top TFs, which is "
        f"a fact about the data rather than a gap — the KRAB-ZNF family shares "
        f"neither a complex nor a process."
    )

    # ---- family summary ---------------------------------------------------
    with st.container(border=True):
        st.markdown(
            "### Families",
            help="`promFEm` / `distFEm` are log2 fold enrichment for promoter "
                 "and distal elements, COMPLEXITY-MATCHED: distal elements "
                 "carry fewer assigned TFs (median 21 vs 48 at promoters), so "
                 "an unmatched enrichment would make any program that loads on "
                 "sparse elements look distal-specific. `substantive` counts "
                 "programs with ≥100 elements and seed stability ≥0.90 — a "
                 "program pinned to three elements reconverges perfectly and "
                 "would otherwise score as highly reproducible.",
        )
        show = fams.copy()
        cols = {"family": "family", "label": "label", "n_programs": "programs",
                "n_substantive": "substantive", "n_elements": "elements",
                "median_stability": "stability",
                "promoter_log2FE_matched": "promFEm",
                "distal_log2FE_matched": "distFEm", "top_tfs": "top TFs"}
        show = show[[c for c in cols if c in show.columns]].rename(columns=cols)
        st.dataframe(
            show, hide_index=True, use_container_width=True,
            column_config={
                "family": st.column_config.NumberColumn(
                    "family", format="%d",
                    help="Family number. Arbitrary — families are the "
                         "clusters of the 140 programs, not a ranking."),
                "label": st.column_config.TextColumn(
                    "label", width="medium",
                    help="Name from MSigDB enrichment over the family's TFs, "
                         "chosen complex-first then by odds ratio, with an "
                         "FDR behind it. Blank means no term cleared FDR — a "
                         "fact about the family, not a gap: the KRAB-ZNF "
                         "family shares neither a complex nor a process."),
                "programs": st.column_config.NumberColumn(
                    "programs", format="%d",
                    help="How many of the 140 programs are in this family."),
                "substantive": st.column_config.NumberColumn(
                    "substantive", format="%d",
                    help="Of those, how many have ≥100 elements AND seed "
                         "stability ≥0.90. A family can be large and mostly "
                         "insubstantial."),
                "elements": st.column_config.NumberColumn(
                    "elements", format="%d",
                    help="Total elements across the family's programs."),
                "stability": st.column_config.NumberColumn(
                    "stability", format="%.3f",
                    help="Median seed stability of the member programs."),
                "promFEm": st.column_config.NumberColumn(
                    "promFEm", format="%.2f",
                    help="Complexity-matched log2 fold enrichment at "
                         "promoters. Matched because distal elements carry "
                         "fewer assigned TFs (median 21 vs 48), so a raw "
                         "value would make any sparse-loading family look "
                         "distal-specific."),
                "distFEm": st.column_config.NumberColumn(
                    "distFEm", format="%.2f",
                    help="The same for distal elements — beyond ±1.5 kb, "
                         "which is 353,550 of the 467,223 elements."),
                "top TFs": st.column_config.TextColumn(
                    "top TFs", width="large",
                    help="Most-loaded TFs across the family's programs. "
                         "Members are grouped by WHERE they act, not by "
                         "shared subunits — PRC2 and PRC1.1 have a TF-loading "
                         "cosine of 0.009 — so this list can look "
                         "heterogeneous and still be one family."),
            })

    # ---- family detail ----------------------------------------------------
    labels = {int(r.family): (f"{int(r.family)} — {r.label}"
                              if "label" in fams.columns else f"{int(r.family)}")
              for _, r in fams.iterrows()}
    pick = st.selectbox("Family", options=list(labels), key="fam_pick",
                        format_func=lambda f: labels[f])

    row = fams[fams.family == pick].iloc[0]
    c1, c2, c3 = st.columns(3)
    c1.metric("elements", f"{int(row.n_elements):,}")
    c2.metric("programs", f"{int(row.n_programs)} ({int(row.n_substantive)} substantive)")
    if "q" in fams.columns and pd.notna(row.get("q")):
        c3.metric("label FDR", f"{row.q:.1e}",
                  help=f"{int(row.overlap)}/{int(row.set_size)} of the term's "
                       f"TFs are in this family")

    tab_ev, tab_prog, tab_genes = st.tabs(
        ["Enrichments", "Programs", "Genes"])

    with tab_ev:
        st.markdown(
            "Every term clearing FDR, best evidence first.",
            help="The headline label is chosen complex-first then by odds "
                 "ratio, so it is usually NOT the top row here (median rank 4) "
                 "— this list ranks by q. The row it came from is flagged. "
                 "`overlap TFs` names which family members are in the term: "
                 "that is the evidence, and it lets you judge a 3/7 hit "
                 "instead of taking the name on trust.")
        lib = st.radio("library", ["all", "CC (complexes)", "BP (processes)"],
                       horizontal=True, key="fam_lib")
        terms = db.get_family_terms(
            pick, limit=40,
            lib={"CC (complexes)": "CC", "BP (processes)": "BP"}.get(lib))
        if terms.empty:
            st.info("No term cleared FDR for this family.")
        else:
            t = terms.rename(columns={
                "is_label": "headline", "label": "term", "overlap": "hits",
                "set_size": "term size", "overlap_tfs": "overlap TFs"})
            st.dataframe(
                t.drop(columns=["url"], errors="ignore"),
                hide_index=True, use_container_width=True,
                column_config={
                    "rank": st.column_config.NumberColumn(
                        "rank", format="%d",
                        help="Position by q among this family's enriched "
                             "terms."),
                    "headline": st.column_config.CheckboxColumn(
                        "headline",
                        help="The row the family's displayed label came from. "
                             "It is usually NOT rank 1 (median rank 4): the "
                             "label is chosen complex-first then by odds "
                             "ratio, while this list ranks by q."),
                    "term": st.column_config.TextColumn(
                        "term", width="medium",
                        help="MSigDB term name."),
                    "lib": st.column_config.TextColumn(
                        "lib",
                        help="CC = cellular component (complexes), "
                             "BP = biological process."),
                    "go_id": st.column_config.TextColumn(
                        "GO id", help="Gene Ontology accession."),
                    "hits": st.column_config.NumberColumn(
                        "hits", format="%d",
                        help="Family TFs that are in the term."),
                    "term size": st.column_config.NumberColumn(
                        "term size", format="%d",
                        help="TFs in the term overall. Read with `hits` — "
                             "3/5 and 3/400 are very different evidence."),
                    "odds": st.column_config.NumberColumn(
                        "odds", format="%.2f",
                        help="Odds ratio of the overlap."),
                    "q": st.column_config.NumberColumn(
                        "q", format="%.1e",
                        help="Benjamini-Hochberg FDR. Every term shown "
                             "cleared 0.05."),
                    "overlap TFs": st.column_config.TextColumn(
                        "overlap TFs", width="large",
                        help="Which family members are in the term. This is "
                             "the evidence — it lets you judge a 3/7 hit "
                             "rather than take the name on trust."),
                })

    with tab_prog:
        progs = db.get_genome_programs(family=pick)
        st.dataframe(
            progs[[c for c in ("program", "n_elements", "seed_stability",
                               "substantive", "promoter_log2FE_matched",
                               "distal_log2FE_matched", "top_tfs")
                   if c in progs.columns]],
            hide_index=True, use_container_width=True,
            column_config=ui.PROGRAM_COLUMNS)

    with tab_genes:
        st.markdown(
            "Genes with the most **promoter-stratum** elements in this family.",
            help="Promoter stratum only, ranked by count then total loading. "
                 "Ranking on all strata returns segmental duplications and "
                 "repeat clusters, which accumulate spurious peaks — for the "
                 "PRC2 family that gave SRGAP2C with 217 elements, 99% of them "
                 "distal. This is a lookup, not an assignment: gene identity "
                 "does not predict an element's program.")
        st.dataframe(
            db.get_genes_in_family(pick, limit=100),
            hide_index=True, use_container_width=True,
            column_config={
                "gene_name": st.column_config.TextColumn(
                    "gene", help="Gene symbol. This is a LOOKUP, not an "
                                 "assignment — gene identity does not predict "
                                 "an element's program once distance is "
                                 "controlled."),
                "n_promoter": st.column_config.NumberColumn(
                    "promoter", format="%d",
                    help="Elements of this family within ±1.5 kb of the "
                         "gene's TSS. The ranking uses this column."),
                "n_distal": st.column_config.NumberColumn(
                    "distal", format="%d",
                    help="Elements of this family beyond the proximal window. "
                         "Ranking on distal counts returns segmental "
                         "duplications and repeat clusters, which accumulate "
                         "spurious peaks — SRGAP2C tops the PRC2 family with "
                         "217 elements, 99% of them distal."),
                "n_elements": st.column_config.NumberColumn(
                    "total", format="%d",
                    help="All of this family's elements nearest this gene."),
                "promoter_weight": st.column_config.NumberColumn(
                    "prom. weight", format="%.3f",
                    help="Summed NMF loading of the promoter-stratum "
                         "elements — the tie-break after the promoter count."),
            })
