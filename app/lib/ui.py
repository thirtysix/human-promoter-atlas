"""Small UI helpers shared across tabs."""
from __future__ import annotations

import re

import streamlit as st


# `**bold**`, `*italic*` and `` `code` `` written inside an intro card used to
# reach the reader as literal asterisks. The card wraps its three strings in a
# raw <div>, and CommonMark does not process markdown inside a block-level HTML
# element -- so `st.markdown(..., unsafe_allow_html=True)` passed them straight
# through. 14 spans across seven tabs read as `**140 programs**` on the live
# site. Converting here rather than rewriting every string keeps markdown
# working the way whoever writes the next card will expect.
_MD_CODE   = re.compile(r"`([^`]+)`")
_MD_BOLD   = re.compile(r"\*\*([^*]+)\*\*")
_MD_ITALIC = re.compile(r"(?<!\*)\*([^*]+)\*(?!\*)")


def md_inline(text: str) -> str:
    """Inline markdown -> HTML, for strings destined for a raw HTML block.

    Bold before italic, or `**x**` is eaten by the italic rule as an empty
    emphasis wrapping `*x*`.
    """
    out = _MD_CODE.sub(r"<code>\1</code>", str(text))
    out = _MD_BOLD.sub(r"<b>\1</b>", out)
    out = _MD_ITALIC.sub(r"<i>\1</i>", out)
    return out


def intro_card(title: str, what: str,
               objective: str, significance: str,
               icon: str = "") -> None:
    """Renders an 'About this view' card at the top of a tab.

    Three short paragraphs: what's shown, what it's for, why it matters.
    Kept narrow on purpose so a first-time visitor isn't lectured.
    """
    with st.container(border=True):
        heading = f"{icon}  {title}".strip() if icon else title
        st.markdown(f"#### {heading}")
        st.markdown(
            f"<div style='line-height:1.55'>"
            f"<b>What you're seeing.</b> {md_inline(what)}<br>"
            f"<b>Objective.</b> {md_inline(objective)}<br>"
            f"<b>Why it matters.</b> {md_inline(significance)}"
            f"</div>",
            unsafe_allow_html=True,
        )


# ---------------------------------------------------------------------------
# Shared column help
# ---------------------------------------------------------------------------
# The Programs tab and the Programs sub-tab of Program families render the
# same columns from the same query. Two copies of this dict would be two
# copies of a definition, which is how `min_score_assign` ended up stated in
# sixteen places and disagreeing with the build. Streamlit ignores keys that
# are not in the frame, so one dict safely covers both the full table and the
# subset.
PROGRAM_COLUMNS = {
    "program": st.column_config.NumberColumn(
        "program", help="Program number — an NMF component over the "
                        "[elements × TFs] occupancy matrix. Numbering is "
                        "arbitrary and specific to this k=140 build; it does "
                        "not correspond to the k=10 promoter programs."),
    "family": st.column_config.TextColumn(
        "family", help="Which of the 28 families this program was grouped "
                       "into, by CO-OCCURRENCE across elements rather than "
                       "shared TFs. Programs in a family mark the same "
                       "places; they need not share subunits."),
    "n_elements": st.column_config.NumberColumn(
        "elements", format="%d",
        help="Genome-wide elements whose dominant loading is this program."),
    "seed_stability": st.column_config.NumberColumn(
        "stability", format="%.3f",
        help="Mean cosine of this component's TF loading across NMF restarts "
             "from different seeds. High on its own is not enough — a "
             "program pinned to a handful of elements reconverges perfectly. "
             "Read it with the element count, or use `substantive`."),
    "substantive": st.column_config.CheckboxColumn(
        "substantive",
        help="≥100 elements AND seed stability ≥0.90. 22 of the 140 programs "
             "hold fewer than 100 elements and cover 0.1% of the data "
             "between them; they are kept but should not carry the same "
             "weight as PRC2."),
    "promoter_log2FE_matched": st.column_config.NumberColumn(
        "promFEm", format="%.2f",
        help="log2 fold enrichment for promoter-stratum elements, "
             "COMPLEXITY-MATCHED. Distal elements carry fewer assigned TFs "
             "(median 21 vs 48), so an unmatched value would make any "
             "sparse-loading program look distal-specific. Positive = this "
             "program is over-represented at promoters."),
    "distal_log2FE_matched": st.column_config.NumberColumn(
        "distFEm", format="%.2f",
        help="Same, for distal elements — those beyond the ±1.5 kb promoter "
             "window, which is 353,550 of the 467,223 elements and the part "
             "a promoter-only analysis cannot see at all."),
    "median_n_tfs": st.column_config.NumberColumn(
        "median TFs", format="%d",
        help="Median number of assigned TFs on this program's elements. This "
             "is the complexity that promFEm/distFEm control for."),
    "top_tfs": st.column_config.TextColumn(
        "top TFs", width="large",
        help="Highest-loading TFs in the component, in order. No complex "
             "annotation enters the pipeline — where these read as a known "
             "complex, the factorization recovered it de novo."),
}


# rank/tf/loading is the shape of every "top TFs in a component" table on the
# site — the genome programs, and the legacy promoter-program popover.
PROGRAM_TF_COLUMNS = {
    "rank": st.column_config.NumberColumn(
        "rank", format="%d",
        help="Position by loading within this component. Rank alone says "
             "nothing about absolute strength — read it with the loading."),
    "tf": st.column_config.TextColumn(
        "TF", help="Transcription factor, by ChIP-Atlas antigen name."),
    "loading": st.column_config.NumberColumn(
        "loading", format="%.3f",
        help="Weight of this TF in the component's H row. NMF drives "
             "components onto largely disjoint TF sets, so a high loading "
             "here usually means a low one everywhere else — which is why "
             "two programs marking the same complex can have a TF-loading "
             "cosine near zero and still belong in one family."),
}
