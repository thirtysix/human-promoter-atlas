"""Reverse search: enter a biological process, see the program families that
enrich for it and the TFs that carried the hit.

Previously searched the k=10 promoter programs and the gene archetypes. Both
are gone on this build -- `list_go_terms()` returned nothing and the page was
a dead nav entry showing one warning. `family_terms` is the live equivalent:
981 terms across 27 of the 28 families, every row FDR-backed.

The unit changed with the layer, and honestly so. Program enrichment was over
the GENES whose promoters a program dominated; family enrichment is over the
FACTORS a family is built from. So the drill-down is "which TFs in this family
are in the term", not "which genes drove it" -- and gene-level drill-down is
not recoverable here, because gene identity carries no information about an
element's program once distance is controlled.
"""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import db, ui, nav


HELP_INTRO = (
    "Enrichment is over each family's TF set against the MSigDB universe, "
    "hypergeometric with BH-FDR. Only terms clearing FDR ≤ 0.05 in at least "
    "one family are searchable, so every row here already cleared it — the "
    "q column tells you by how much."
)

HELP_FAM_TBL = (
    "Families whose TF set is enriched for this term. `hits/size` is the "
    "overlap against the term's own size — 3/7 and 3/400 are very different "
    "evidence for the same odds ratio. `headline` marks a family whose "
    "displayed name came from this very term, rather than a term it merely "
    "also hits."
)


def render() -> None:
    ui.intro_card(
        title="Reverse search by biological process",
        what="Type a process and see every <b>program family</b> enriched "
             "for it, with an FDR on each hit and the family TFs that "
             "carried it.",
        objective="Answer <i>which regulators implement this biology?</i> — "
                   "the inverse of browsing families and reading their labels.",
        significance="The entry point for arriving with the biology rather "
                      "than the regulator. <i>Immune</i> returns 10 families; "
                      "<i>response to growth factor</i> returns 13, led by "
                      "MLL3/4 at odds 6.7.",
    )

    terms = db.list_family_go_terms()
    if terms.empty:
        st.info(
            "No family enrichments are loaded. This page reads "
            "`family_terms`, written by `pipeline/genome_family_labels.py`."
        )
        return

    # ---- Search widget ---------------------------------------------------
    with st.container(border=True):
        st.markdown("### Pick a process", help=HELP_INTRO)
        preset = st.session_state.get("go_search_preset", "")
        labels = sorted(terms["label"])
        lookup = {l.lower(): l for l in labels}
        # ?go= arrives lowercased and underscore-stripped; match forgivingly
        # rather than dropping a link on the floor.
        seed = lookup.get(str(preset).strip().lower(), "")
        options = [""] + labels
        choice = st.selectbox(
            f"Type to search ({len(labels):,} terms enriched in at least "
            f"one family)",
            options=options,
            index=options.index(seed) if seed in options else 0,
            placeholder="e.g. immune, chromatin, cell fate commitment…",
            key="go_search_select",
            help="Start typing — Streamlit filters as you type.",
        )
        # A term that is absent from this list was not tested and found wanting
        # -- it is simply not in the vocabulary, and saying "not enriched for
        # any program" would claim a negative result we do not have.
        st.caption(
            f"This searches the **{len(labels):,} terms that clear FDR ≤ 0.05 "
            "in at least one family** — not all of GO. The families are built "
            "from TF sets, so the vocabulary leans toward complexes and "
            "regulatory processes (2,895 BP rows against 71 CC). Plenty of "
            "real biology is legitimately absent: *ribosome biogenesis*, for "
            "one, returns nothing here. A term missing from this list has not "
            "been tested and rejected — it is out of scope for this layer."
        )

        if not choice:
            st.markdown(
                "**Most broadly shared** — terms that enrich across the most "
                "families, i.e. biology many regulatory contexts touch:")
            teaser = terms.head(12)
            st.dataframe(
                teaser[["label", "lib", "go_id", "n_families", "min_q",
                        "max_odds"]],
                hide_index=True, width="stretch",
                column_config=_TERM_COLUMNS,
            )
            st.info("Pick a term above to see which families enrich for it.")
            return

    _render_term(choice, terms)


_TERM_COLUMNS = {
    "label": st.column_config.TextColumn(
        "term", width="large",
        help="MSigDB term, in its readable form."),
    "lib": st.column_config.TextColumn(
        "lib",
        help="CC = cellular component (complexes), BP = biological process. "
             "The families skew heavily BP; only 71 of 2,966 rows are CC."),
    "go_id": st.column_config.TextColumn(
        "GO ID", help="Gene Ontology accession."),
    "n_families": st.column_config.NumberColumn(
        "# families", format="%d",
        help="How many of the 28 families enrich for this term. High = broad "
             "biology touched by many regulatory contexts; 1 = specific."),
    "min_q": st.column_config.NumberColumn(
        "best q", format="%.1e",
        help="Smallest BH-FDR across the families that hit this term."),
    "max_odds": st.column_config.NumberColumn(
        "best odds", format="%.2f",
        help="Largest odds ratio across those families."),
}


def _render_term(label: str, terms: pd.DataFrame) -> None:
    fams = db.families_for_go_term(label)
    row = terms[terms["label"] == label]
    go_id = str(row["go_id"].iloc[0]) if len(row) else ""
    raw = str(row["term"].iloc[0]) if len(row) else ""

    if fams.empty:
        st.warning(f"No family clears FDR for “{label}”.")
        return

    c1, c2, c3 = st.columns(3)
    c1.metric("families enriched", f"{len(fams)}",
              help="Of 28. A term hitting one family is specific to that "
                   "regulatory context; a term hitting a dozen is biology "
                   "many contexts touch.")
    c2.metric("best FDR", f"{fams['q'].min():.1e}",
              help="Smallest BH-FDR among the families below.")
    c3.metric("term size", f"{int(fams['set_size'].iloc[0]):,} TFs",
              help="How many TFs are in this MSigDB term overall. Read the "
                   "per-family overlap against it.")

    st.caption(f"`{raw}` · {go_id}")

    with st.container(border=True):
        st.markdown("### Families enriched for this term", help=HELP_FAM_TBL)
        show = fams.rename(columns={
            "family_label": "family name", "is_label": "headline",
            "overlap": "hits", "set_size": "term size",
            "overlap_tfs": "overlap TFs", "n_programs": "programs"})
        st.dataframe(
            show[["family", "family name", "headline", "hits", "term size",
                  "odds", "q", "programs", "overlap TFs"]],
            hide_index=True, width="stretch",
            column_config={
                "family": st.column_config.NumberColumn(
                    "family", format="%d",
                    help="Family number. Arbitrary — families are clusters, "
                         "not a ranking."),
                "family name": st.column_config.TextColumn(
                    "family name", width="medium",
                    help="The family's own headline label, which may be a "
                         "different term from the one you searched."),
                "headline": st.column_config.CheckboxColumn(
                    "headline",
                    help="Ticked when the family's displayed name came from "
                         "THIS term. Unticked means the family hits the term "
                         "but is named after something else."),
                "hits": st.column_config.NumberColumn(
                    "hits", format="%d",
                    help="Family TFs that are in the term."),
                "term size": st.column_config.NumberColumn(
                    "term size", format="%d",
                    help="TFs in the term overall. Read with `hits` — 3/7 and "
                         "3/400 are not the same evidence."),
                "odds": st.column_config.NumberColumn(
                    "odds", format="%.2f",
                    help="Odds ratio of the overlap."),
                "q": st.column_config.NumberColumn(
                    "q", format="%.1e",
                    help="BH-FDR. Everything listed cleared 0.05."),
                "programs": st.column_config.NumberColumn(
                    "programs", format="%d",
                    help="How many of the 140 programs are in this family."),
                "overlap TFs": st.column_config.TextColumn(
                    "overlap TFs", width="large",
                    help="Which family members are in the term — the "
                         "evidence behind the row, so a 3/7 hit can be "
                         "judged rather than taken on trust."),
            },
        )

        st.markdown("**Open a family**")
        ids = [int(f) for f in fams["family"].head(8)]
        cols = st.columns(min(len(ids), 4))
        for i, fid in enumerate(ids):
            name = str(fams.loc[fams.family == fid, "family_label"].iloc[0])
            with cols[i % len(cols)]:
                if st.button(f"Family {fid} — {name[:28]}",
                              key=f"go_open_fam_{fid}",
                              width="stretch"):
                    st.session_state["fam_pick"] = fid
                    nav.goto("archetypes")

    with st.expander("Download", expanded=False):
        st.download_button(
            f"Families enriched for “{label}” — TSV",
            data=fams.to_csv(sep="\t", index=False).encode(),
            file_name=f"families_{go_id or label.replace(' ', '_')}.tsv",
            mime="text/tab-separated-values",
            key=f"go_dl_fams_{label}",
        )
