"""Aggregate tab: TF×position heatmap + global metaplot."""
from __future__ import annotations

import streamlit as st

from app.lib import db, plotting, ui, nav


HIGHLIGHT_DEFAULT = ["CTCF", "YY1", "MYC", "SP1", "NRF1", "REST", "TBP", "EP300"]

HELP_AGG = (
    "Mean per-bp binding profile averaged across all 19,745 canonical "
    "protein-coding TSSs in the human genome (Ensembl GRCh38.114). For "
    "each TF, every chip-atlas peak's midpoint is recentered to a 25-nt "
    "block; the value at offset x is the fraction of TSSs with that TF's "
    "block at position x relative to the TSS, in transcription direction."
)
HELP_FLAVOR = (
    "binary = TF × TSS coverage probability after merging overlapping "
    "peaks within a single TF (true [0, 1] occupancy across the dataset). "
    "score = sum of chip-atlas peak scores at each bp (peaks NOT merged; "
    "reflects total binding evidence density)."
)
HELP_HIGHLIGHT = (
    "Each highlighted TF is drawn as a colored line on top of the green "
    "mean-of-means band. Useful for comparing where individual factors sit "
    "relative to the population average."
)
HELP_HEATMAP = (
    "Top-N TFs by total signal across the window. Each row is one TF, "
    "each column is one bp. Hover for exact values."
)


def render() -> None:
    ui.intro_card(
        title="Aggregate TF binding — the genome-wide baseline",
        what="Mean binding profile of each TF across all 19,745 canonical "
             "protein-coding promoters. Position is transcription-oriented; "
             "0 = TSS.",
        objective="Establish the population-level pattern *before* "
                   "drilling into any one gene — spot universal anchors "
                   "(TBP at ~−30 bp = the TATA box) and compare individual "
                   "TFs to the crowd.",
        significance="The reference everything else is interpreted against. "
                      "If TBP doesn't peak just upstream of TSS, every "
                      "downstream conclusion is suspect.",
    )

    # Quick-start cards — useful for first-time visitors
    with st.container(border=True):
        st.markdown(
            "#### Quick starts",
            help="Hand-curated jumping-off points. Each link sets the "
                 "search state for one of the explorer tabs — click, then "
                 "switch to that tab to see the result.",
        )
        c1, c2, c3, c4 = st.columns(4)
        # No hard-coded counts on these labels. "GAPDH (10 modules, 8
        # programs)" survived a rebuild that left the gene with 9 modules and
        # 6 programs, and a wrong number on the first thing a visitor clicks
        # is worse than no number.
        if c1.button("GAPDH — a multi-module housekeeping promoter",
                      width="stretch",
                      help="The textbook case: several distinct modules across "
                           "one promoter, each drawing a different set of "
                           "factors."):
            st.session_state["tx_gene_select"] = "GAPDH"
            nav.goto("transcript")
        if c2.button("ESR1 (hormone receptor)",
                      width="stretch",
                      help="Estrogen receptor — promoter showcases hormone-"
                           "responsive program composition."):
            st.session_state["tx_gene_select"] = "ESR1"
            nav.goto("transcript")
        if c3.button("CTCF — cohesin at the TSS",
                      width="stretch",
                      help="CTCF's loading is concentrated almost entirely on "
                           "one genome program, alongside RAD21 and STAG1 — "
                           "the mitotic cohesin family, recovered without "
                           "any complex annotation."):
            st.session_state["tf_select"] = "CTCF"
            nav.goto("tf")
        # Was "A6 (cohesin → adhesion)": an archetype that no longer exists,
        # whose help described k=10 P5, and which actually landed on program
        # 5 -- a histone deacetylase program in the current numbering.
        if c4.button("PRC2 — a complex found, not told",
                      width="stretch",
                      help="Program 12: EZH2, JARID2 and SUZ12 load together "
                           "across 11,573 elements at seed stability 0.978. "
                           "No complex annotation enters the pipeline — the "
                           "factorization put them together on its own."):
            st.session_state["prog_pick"] = 12
            nav.goto("programs")

    st.subheader("Aggregate TF binding around canonical TSSs",
                  help=HELP_AGG)
    st.caption(
        "Mean per-bp binding profile across 19,745 canonical protein-coding "
        "TSSs (Ensembl GRCh38.114). chip-atlas peaks recentered to a 25-nt "
        "block around their midpoint; ±1,000 bp window, txn-oriented."
    )

    with st.container(border=True):
        col_f, col_h = st.columns([1, 3])
        with col_f:
            flavor_label = st.radio(
                "Signal flavor",
                ["binary occupancy", "summed score"],
                horizontal=False, index=0,
                help=HELP_FLAVOR,
            )
            flavor = "binary" if flavor_label.startswith("binary") else "score"

        matrix = db.load_aggregate_matrix(flavor)
        if matrix.empty:
            st.error("Aggregate matrix not found. "
                     "Run `python data/build_app_db.py`.")
            return

        with col_h:
            tf_options = sorted(matrix.index.tolist())
            highlight = st.multiselect(
                "Highlight TFs in metaplot",
                options=tf_options,
                default=[t for t in HIGHLIGHT_DEFAULT if t in tf_options],
                help=HELP_HIGHLIGHT,
            )

    ylabel = ("mean per-bp coverage probability"
              if flavor == "binary" else "mean per-bp summed score")
    title  = f"Mean across {len(matrix)} TFs — {flavor_label}"
    with st.container(border=True):
        st.plotly_chart(
            plotting.fig_aggregate_metaplot(matrix, highlight, ylabel, title),
            width="stretch",
        )

    with st.expander("TF × position heatmap (top N by total signal)",
                      expanded=False):
        st.caption(HELP_HEATMAP)
        col_n, col_o = st.columns([1, 2])
        with col_n:
            n_show = st.slider("Number of TFs",
                               50, min(500, len(matrix)),
                               value=200, step=25,
                               help="More TFs = taller heatmap; fewer = "
                                    "easier to scan the strongest binders.")
        with col_o:
            order_label = st.radio(
                "Row ordering",
                ["total signal (default)",
                 "argmax position (upstream → downstream)",
                 "hierarchical (Ward, on shape)",
                 "filtered K=8 cluster"],
                horizontal=True, index=0,
                help="`total signal` shows the strongest binders first; "
                     "`argmax position` lays rows out by where each TF "
                     "peaks (upstream at top, downstream at bottom); "
                     "`hierarchical` clusters TFs by *shape* (peak-"
                     "normalized) so TFs with similar profiles sit next "
                     "to each other; `K=8 cluster` groups by the precomputed "
                     "TF-clustering blocks."
            )
        order_by = {
            "total signal (default)":                 "total_signal",
            "argmax position (upstream → downstream)":"argmax_position",
            "hierarchical (Ward, on shape)":          "hierarchical",
            "filtered K=8 cluster":                   "cluster",
        }[order_label]

        # Provide cluster lookup only when needed
        tf_to_cluster = None
        if order_by == "cluster":
            from app.lib import db as _db
            con = _db.get_con()
            tfc = con.execute(
                "SELECT tf, cluster_filtered FROM tf "
                "WHERE cluster_filtered IS NOT NULL"
            ).df()
            tf_to_cluster = dict(zip(tfc["tf"], tfc["cluster_filtered"].astype(int)))

        cmap = "Viridis" if flavor == "binary" else "Magma"
        st.plotly_chart(
            plotting.fig_aggregate_heatmap(
                matrix, ylabel, cmap, n_show=n_show,
                order_by=order_by, tf_to_cluster=tf_to_cluster,
            ),
            width="stretch",
        )

    with st.expander("Download data", expanded=False):
        st.markdown(
            "Per-TF aggregate matrices in TSV (gzipped) and parquet are at "
            "`analyses/canonical_promoter/matrices/` in the analysis repo. "
            "The slice rendered above lives at "
            f"`data/aggregate/tf_x_position.{flavor}.parquet`."
        )
        sample_csv = matrix.head(50).to_csv(index=True).encode()
        st.download_button(
            f"Top-50-TF rows of {flavor} matrix (CSV)",
            data=sample_csv,
            file_name=f"tf_x_position.{flavor}.top50.csv",
            mime="text/csv",
        )
