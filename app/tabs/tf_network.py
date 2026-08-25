"""TF network tab — sortable atlas-wide TF × TF co-occurrence table.

Source: the precomputed `data/tf_pair_cooccurrence.parquet` produced by
`data/build_tf_pair_table.py`. ~330k unordered TF pairs (n_shared ≥ 5),
each with Jaccard symmetric overlap and lift over independence.
"""
from __future__ import annotations

import streamlit as st

from app.lib import db, ui


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def _full_table_gzipped() -> bytes:
    """Gzipped TSV of the entire 327k-pair table — cached so the
    serialization + compression only runs once per process lifetime."""
    import gzip
    full = db.tf_pair_query(
        search="", min_shared=0, min_jaccard=0.0,
        sort_by="n_shared", ascending=False, limit=10_000_000,
    )
    return gzip.compress(full.to_csv(sep="\t", index=False).encode(),
                          compresslevel=6)


def render() -> None:
    ui.intro_card(
        title="TF network — atlas-wide co-binding pairs",
        what="One row per unordered (TF A, TF B) pair counting how many "
             "of the 117,006 modules they share. Jaccard = symmetric "
             "overlap; lift = co-occurrence over independence.",
        objective="Surface TF cliques and obligate partnerships across "
                   "the whole atlas — complement to the focal-TF view "
                   "in the Per-TF tab.",
        significance="Lift names specific biology that raw counts hide. "
                      "MAX × MYC, an obligate heterodimer, sits high on "
                      "lift even though neither lands at the very top "
                      "of the raw-shared-modules ranking.",
    )

    if not db.tf_pair_table_available():
        st.warning(
            "TF pair table not built yet. Run "
            "`python data/build_tf_pair_table.py` from the repo root."
        )
        return

    stats = db.tf_pair_table_stats()
    if stats:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Pairs in table", f"{stats['n_pairs']:,}",
                   help="Unordered (TF A, TF B) pairs with at least "
                        "5 shared modules (the build threshold).")
        c2.metric("Unique TFs", f"{stats['n_tfs']:,}",
                   help="TFs appearing in at least one qualifying pair.")
        c3.metric("Max shared modules", f"{stats['max_share']:,}",
                   help="Highest n_shared value in the table (the "
                        "tightest binding co-pair across the atlas).")
        c4.metric(
            "Max lift",
            f"{stats['max_lift']:.1f}×",
            help=(
                "Highest observed / expected ratio under independence "
                "among pairs with at least "
                f"{stats.get('lift_min_shared', 1000):,} shared modules "
                "(matches the table's default filter — keeps the metric "
                "consistent with what you see when browsing). Lift is "
                "noisy for pairs with very small marginals, so unfiltered "
                "max-lift is dominated by rare-rare statistical artifacts."
            ),
        )

    # ---- Filter / sort controls -------------------------------------------
    with st.container(border=True):
        st.markdown("### Browse pairs")
        f1, f2, f3, f4, f5 = st.columns([2, 1, 1, 1, 1])
        with f1:
            search = st.text_input(
                "Search TF (substring)",
                value="",
                placeholder="e.g. CTCF, MYC, RAD…",
                help="Case-insensitive substring; matches either TF in "
                     "the pair. Empty = no filter.",
                key="tfnet_search",
            )
        with f2:
            min_shared = st.number_input(
                "Min n_shared", min_value=5, max_value=20000,
                value=1000, step=50,
                help="Lower bound on the number of co-occurring modules. "
                     "1,000 is a generous default — try 5,000 to focus on "
                     "the tightest pairs only.",
                key="tfnet_min_shared",
            )
        with f3:
            min_jaccard = st.slider(
                "Min Jaccard", 0.0, 1.0, 0.0, step=0.05,
                help="Lower bound on symmetric overlap "
                     "n_shared / (n_a + n_b − n_shared).",
                key="tfnet_min_jaccard",
            )
        with f4:
            sort_by = st.selectbox(
                "Sort by (full table)", ["n_shared", "jaccard", "lift"],
                index=0,
                help="Re-queries the entire 327 K-pair table by this "
                     "column descending, then keeps the top N. "
                     "`n_shared` ranks raw co-occurrence; `jaccard` "
                     "controls for marginal TF abundance; `lift` "
                     "captures specificity vs independence.",
                key="tfnet_sort",
            )
        with f5:
            limit = st.number_input(
                "Show top N", min_value=50, max_value=5000,
                value=2000, step=50,
                help="How many top-ranked rows to render. Larger N lets "
                     "you re-sort the visible rows by clicking a column "
                     "header in the table and still see meaningful "
                     "variation (the in-table sort doesn't re-query the "
                     "full pool — use the dropdown for that).",
                key="tfnet_limit",
            )

    pairs = db.tf_pair_query(
        search=search,
        min_shared=int(min_shared),
        min_jaccard=float(min_jaccard),
        sort_by=sort_by, ascending=False,
        limit=int(limit),
    )

    with st.container(border=True):
        st.markdown(
            f"### Top {len(pairs):,} pairs",
            help="Each row is one unordered TF pair. Pair order is "
                 "alphabetical (A < B) so each pair appears once.",
        )
        st.caption(
            "**How sorting works:** the **Sort by** dropdown above "
            "re-queries the *full* 327 K-pair table by your chosen "
            "column descending and keeps the top N. **Clicking a "
            "column header in the table below** only re-sorts the "
            "rows already shown — so values further down the full "
            "table won't appear unless you bump *Show top N* or "
            "change the Sort-by column."
        )
        if pairs.empty:
            st.info("No pairs match the current filters — relax min_shared "
                    "or clear the search.")
        else:
            st.dataframe(
                pairs, hide_index=True, width="stretch",
                column_config={
                    "tf_a": st.column_config.TextColumn(
                        "TF A", width="small",
                        help="First TF (alphabetical)."),
                    "tf_b": st.column_config.TextColumn(
                        "TF B", width="small",
                        help="Second TF (alphabetical)."),
                    "n_shared": st.column_config.NumberColumn(
                        "# shared", width="small",
                        help="# modules containing BOTH TFs."),
                    "n_a": st.column_config.NumberColumn(
                        "# TF A", width="small",
                        help="Total atlas-wide modules containing TF A."),
                    "n_b": st.column_config.NumberColumn(
                        "# TF B", width="small",
                        help="Total atlas-wide modules containing TF B."),
                    "jaccard": st.column_config.NumberColumn(
                        "Jaccard", format="%.3f", width="small",
                        help="n_shared / (n_a + n_b − n_shared). "
                             "Symmetric overlap; 1 = identical, 0 = disjoint."),
                    "lift": st.column_config.NumberColumn(
                        "lift", format="%.2f", width="small",
                        help="Observed / expected under independence. "
                             "Lift > 1 = pair co-occurs more than chance, "
                             "< 1 = mutually avoided."),
                },
            )
            d1, d2 = st.columns(2)
            with d1:
                st.download_button(
                    f"Filtered slice ({len(pairs):,} rows) — TSV",
                    data=pairs.to_csv(sep="\t", index=False).encode(),
                    file_name=("tf_pair_cooccurrence_"
                               f"{sort_by}_top{len(pairs)}.tsv"),
                    mime="text/tab-separated-values",
                    help="The rows currently visible — respects search, "
                         "min n_shared, min Jaccard, sort, and 'Show top N'.",
                    key="tfnet_dl_filtered",
                    width="stretch",
                )
            with d2:
                # Full atlas — cached gzip bytes; serialization +
                # compression only happens once per process lifetime.
                gz_bytes = _full_table_gzipped()
                n_rows = db.tf_pair_table_stats().get("n_pairs", 0)
                st.download_button(
                    f"Full atlas table ({n_rows:,} rows) — TSV.gz",
                    data=gz_bytes,
                    file_name="tf_pair_cooccurrence.tsv.gz",
                    mime="application/gzip",
                    help="Every TF pair with n_shared ≥ 5 across the "
                         "atlas. Gzipped to ~4 MB; unpack with "
                         "`gunzip` or open directly in pandas "
                         "(`pd.read_csv('...tsv.gz', sep='\\t')`).",
                    key="tfnet_dl_full",
                    width="stretch",
                )
