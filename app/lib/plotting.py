"""Plotly figure builders for the Human Promoter Atlas."""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.colors as pc
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.ndimage import gaussian_filter1d


OUTER_HALF = 1500
KDE_BW     = 25
LEN_GRID   = 2 * OUTER_HALF + 1


# Brand palette — keep in sync with .streamlit/config.toml primaryColor.
PRIMARY      = "#0F766E"          # teal-700: primary data trace / hero line
PRIMARY_FILL = "rgba(15,118,110,0.18)"   # PRIMARY at 18% alpha
REFERENCE    = "#64748b"          # slate-500: population mean / reference
ACCENT       = "#C2410C"          # orange-700: comparison highlight
WARM         = "#DC2626"          # red-600: threshold markers
# Single-hue sequential scale for "more is more" continuous metrics.
BRAND_SCALE  = "Teal"


# 10 distinct, color-blind-friendly hues for k=10 program ribbon
PROGRAM_COLORS = pc.qualitative.Plotly[:10]


# ---------------------------------------------------------------------------
# Aggregate tab
# ---------------------------------------------------------------------------
def fig_aggregate_metaplot(matrix: pd.DataFrame, highlight_tfs: list[str],
                            ylabel: str, title: str) -> go.Figure:
    """Mean-of-means + per-TF light traces for a few highlighted TFs."""
    x = matrix.columns.values
    mean = matrix.values.mean(axis=0)
    sem  = matrix.values.std(axis=0, ddof=1) / np.sqrt(len(matrix))

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=np.concatenate([x, x[::-1]]),
        y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
        fill="toself", fillcolor=PRIMARY_FILL, line=dict(color="rgba(0,0,0,0)"),
        name="±1 SEM (across TFs)", hoverinfo="skip", showlegend=True,
    ))
    fig.add_trace(go.Scatter(
        x=x, y=mean, mode="lines", name=f"mean across {len(matrix)} TFs",
        line=dict(color=PRIMARY, width=2.5),
    ))
    palette = pc.qualitative.D3
    for i, tf in enumerate(highlight_tfs):
        if tf in matrix.index:
            fig.add_trace(go.Scatter(
                x=x, y=matrix.loc[tf].values,
                mode="lines", name=tf,
                line=dict(color=palette[i % len(palette)], width=1.5),
            ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1, opacity=0.6)
    fig.update_layout(
        title=title, xaxis_title="bp from TSS (txn-oriented)",
        yaxis_title=ylabel,
        hovermode="x unified",
        legend=dict(orientation="v", x=1.02, y=1, font=dict(size=10)),
        margin=dict(l=60, r=160, t=60, b=50), height=440,
    )
    return fig


def fig_aggregate_heatmap(matrix: pd.DataFrame, label: str,
                          colorscale: str = "Viridis",
                          n_show: int = 200,
                          order_by: str = "total_signal",
                          tf_to_cluster: dict | None = None) -> go.Figure:
    """Top n_show TFs by total signal, but row order configurable.
    order_by ∈ {'total_signal', 'argmax_position', 'hierarchical', 'cluster'}.
    Plotly heatmap chokes >>500 rows; capped n_show is mandatory."""
    # First: subset to top-N by total signal (the "strongest binders" filter).
    by_signal = matrix.sum(axis=1).sort_values(ascending=False).index
    M = matrix.loc[by_signal].head(n_show)

    # Then re-order the rows according to order_by.
    if order_by == "argmax_position":
        peak_pos = M.values.argmax(axis=1)
        # bp at the argmax = M.columns[peak_pos]; sort upstream→downstream
        col_pos = np.array([M.columns[p] for p in peak_pos])
        new_order = M.index.values[np.argsort(col_pos)]
        M = M.loc[new_order]
    elif order_by == "hierarchical":
        from scipy.cluster.hierarchy import linkage, leaves_list
        # Peak-normalize each row before clustering — we want to cluster on
        # SHAPE, not amplitude (matches cluster_tfs.001.py)
        row_max = M.values.max(axis=1, keepdims=True)
        norm = M.values / np.where(row_max > 0, row_max, 1.0)
        Z = linkage(norm, method="ward", metric="euclidean")
        order_idx = leaves_list(Z)
        M = M.iloc[order_idx]
    elif order_by == "cluster" and tf_to_cluster:
        # Sort by precomputed K=8 cluster id, then by total signal within cluster
        clu = M.index.map(lambda t: tf_to_cluster.get(t, 999))
        sig = M.sum(axis=1).values
        # primary: cluster (asc), secondary: total signal (desc)
        new_order = M.index.values[np.lexsort((-sig, np.asarray(clu)))]
        M = M.loc[new_order]
    # else: total_signal (already done)

    vmax = float(np.quantile(M.values, 0.99))
    fig = go.Figure(data=go.Heatmap(
        z=M.values, x=M.columns.values, y=M.index.tolist(),
        colorscale=colorscale, zmin=0, zmax=vmax,
        colorbar=dict(title=label, thickness=12),
        hovertemplate="TF: %{y}<br>bp: %{x}<br>signal: %{z:.4f}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="white", line_width=1, opacity=0.7)
    fig.update_layout(
        title=f"TF × position heatmap (top {n_show} by total signal)",
        xaxis_title="bp from TSS", yaxis_title="TF",
        margin=dict(l=80, r=40, t=60, b=50),
        height=max(400, 14 * len(M.index)),
    )
    fig.update_yaxes(autorange="reversed", tickmode="linear",
                      tickfont=dict(size=9), automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Programs tab
# ---------------------------------------------------------------------------
def fig_program_cooccurrence(coocc: pd.DataFrame, mode: str = "lift") -> go.Figure:
    """Heatmap of P_i × P_j gene-level co-occurrence.
    mode='count' shows raw n_both; mode='lift' shows observed/expected.
    Diagonal = same program present multiple times in the same gene."""
    progs = sorted(set(coocc["p_i"]) | set(coocc["p_j"]))
    M = np.zeros((len(progs), len(progs)), dtype=np.float64)
    for _, r in coocc.iterrows():
        i, j = progs.index(int(r["p_i"])), progs.index(int(r["p_j"]))
        M[i, j] = float(r["n_both"]) if mode == "count" else float(r["lift"] or 1.0)

    if mode == "lift":
        # diverging around 1.0 (independence)
        cmax = max(2.0, float(np.nanmax(M)))
        cmin = max(0.0, 2.0 - cmax)
        cscale = "RdBu_r"
        zmid = 1.0
        label = "Lift = observed / expected (1 = independent)"
    else:
        cmax = float(np.nanmax(M))
        cmin = 0.0
        cscale = "Viridis"
        zmid = None
        label = "Genes with both programs"

    fig = go.Figure(go.Heatmap(
        z=M, x=[f"P{p}" for p in progs], y=[f"P{p}" for p in progs],
        colorscale=cscale, zmin=cmin, zmax=cmax, zmid=zmid,
        colorbar=dict(title=label, thickness=12),
        hovertemplate=(
            "row: %{y}<br>col: %{x}<br>"
            f"{'lift' if mode=='lift' else 'genes'}: %{{z:.2f}}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=("Program co-occurrence at the gene level "
               f"({'lift over independence' if mode == 'lift' else 'shared gene count'})"),
        margin=dict(l=60, r=60, t=70, b=60),
        height=520, width=560,
        xaxis=dict(side="bottom"),
        yaxis=dict(autorange="reversed"),
    )
    return fig


def fig_program_position_density(centers: np.ndarray, program: int,
                                  reading: str) -> go.Figure:
    """Histogram of module-center positions for one program."""
    bins = np.linspace(-OUTER_HALF, OUTER_HALF, 121)
    h, edges = np.histogram(centers, bins=bins)
    mids = 0.5 * (edges[:-1] + edges[1:])
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=mids, y=h, width=np.diff(edges), name=f"P{program}",
        marker_color=PROGRAM_COLORS[(program - 1) % 10],
        marker_line_width=0,
        hovertemplate="bp: %{x:.0f}<br>n_modules: %{y}<extra></extra>",
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1, opacity=0.6)
    fig.update_layout(
        title=f"P{program} ({reading}) — module center positions",
        xaxis_title="module center, bp from TSS (txn-oriented)",
        yaxis_title=f"# modules  (n={len(centers):,})",
        margin=dict(l=60, r=20, t=60, b=50), height=740,
        showlegend=False,
    )
    return fig


def fig_program_tf_top(top_df: pd.DataFrame, program: int) -> go.Figure:
    """Horizontal bar of top TFs for one program. Force every label visible."""
    df = top_df.head(30).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=df["loading"], y=df["tf"], orientation="h",
        marker_color=PROGRAM_COLORS[(program - 1) % 10],
        hovertemplate="TF: %{y}<br>H loading: %{x:.3f}<extra></extra>",
    ))
    fig.update_layout(
        title=f"P{program} — top TFs by H loading",
        xaxis_title="H loading", yaxis_title="",
        margin=dict(l=80, r=20, t=60, b=40),
        height=max(480, 22 * len(df) + 80),
    )
    fig.update_yaxes(tickmode="linear", tickfont=dict(size=10),
                      automargin=True)
    return fig


DRIVER_CLASS_COLORS = {
    "no-driver":     "#b0b0b0",
    "single-driver": "#4c9be8",
    "multi-driver":  "#e67c4a",
}


def fig_program_driver_class_distribution(
    dist: pd.DataFrame,
    selected: int | None = None,
    class_order: list[str] | None = None,
) -> go.Figure:
    """Stacked horizontal bar — one row per program, segments are the share
    of modules in each driver class (no-/single-/multi-driver). Hover
    shows raw module count and percentage. The selected program (if given)
    is outlined in black.

    dist: long DataFrame [program, driver_class, n_modules].
    """
    if class_order is None:
        class_order = ["multi-driver", "single-driver", "no-driver"]

    # Pivot: rows = program, cols = driver_class, values = n_modules.
    M = (dist.pivot(index="program", columns="driver_class",
                    values="n_modules")
              .fillna(0).astype(int))
    for c in class_order:
        if c not in M.columns:
            M[c] = 0
    M = M[class_order].sort_index()
    totals = M.sum(axis=1).replace(0, 1)
    pct = M.div(totals, axis=0) * 100.0

    progs = M.index.tolist()
    y_labels = [f"P{p}" for p in progs]
    fig = go.Figure()
    for cls in class_order:
        outline = ["black" if (selected is not None and p == selected)
                   else "rgba(0,0,0,0)" for p in progs]
        line_w  = [2.0 if (selected is not None and p == selected) else 0.0
                   for p in progs]
        fig.add_trace(go.Bar(
            y=y_labels, x=pct[cls].values, name=cls, orientation="h",
            marker=dict(color=DRIVER_CLASS_COLORS[cls],
                        line=dict(color=outline, width=line_w)),
            customdata=np.column_stack([M[cls].values, totals.values]),
            hovertemplate=(
                "%{y} — " + cls + "<br>"
                "%{customdata[0]:,} of %{customdata[1]:,} modules"
                " (%{x:.1f}%)<extra></extra>"
            ),
        ))
    fig.update_layout(
        barmode="stack",
        title="Module driver-class distribution by program",
        xaxis=dict(title="share of modules (%)", range=[0, 100],
                   ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=20, t=60, b=50),
        height=max(280, 28 * len(progs) + 120),
        legend=dict(orientation="h", yanchor="bottom", y=1.02,
                    xanchor="right", x=1),
    )
    return fig


def fig_tf_cobinding_partners(df: pd.DataFrame, focal_tf: str) -> go.Figure:
    """Horizontal bar — top partner TFs by # shared modules with focal_tf.
    Bar color encodes 'pct_of_partner_modules' (how dependent partner is
    on co-occurring with focal). Hover shows both directional %s plus
    Jaccard."""
    if df.empty:
        return go.Figure()
    d = df.iloc[::-1]  # top at top
    fig = go.Figure(go.Bar(
        x=d["n_shared"], y=d["partner"], orientation="h",
        marker=dict(color=d["pct_of_partner_modules"],
                     colorscale=BRAND_SCALE, cmin=0, cmax=100,
                     colorbar=dict(title="% of partner's<br>modules",
                                    thickness=10),
                     line=dict(color="rgba(0,0,0,0.4)", width=0.3)),
        customdata=d[["partner_total", "focal_total",
                      "pct_of_partner_modules",
                      "pct_of_focal_modules", "jaccard"]].values,
        hovertemplate=(
            "partner: %{y}<br>"
            "shared modules: %{x:,}<br>"
            "%{customdata[0]:,} total %{y} modules · "
            "%{customdata[1]:,} total " + focal_tf + " modules<br>"
            "%{customdata[2]:.1f}% of %{y}'s modules contain " + focal_tf + "<br>"
            "%{customdata[3]:.1f}% of " + focal_tf + "'s modules contain %{y}<br>"
            "Jaccard: %{customdata[4]:.3f}<extra></extra>"
        ),
    ))
    fig.update_layout(
        title=f"{focal_tf} — top co-binding partner TFs",
        xaxis_title="# shared modules", yaxis_title="",
        margin=dict(l=80, r=20, t=60, b=50),
        height=max(380, 22 * len(d) + 100),
    )
    fig.update_yaxes(tickmode="linear", tickfont=dict(size=10),
                      automargin=True)
    return fig


def fig_program_strand_asymmetry(
    dist: pd.DataFrame, selected: int | None = None,
) -> go.Figure:
    """Horizontal bar — fraction of each program's modules whose parent
    transcript is on the (+) strand. 50% reference line marks no bias.
    Hovers show raw + and − counts.

    dist: DataFrame [program, n_plus, n_minus, total, frac_plus].
    """
    if dist.empty:
        return go.Figure()
    d = dist.sort_values("program").copy()
    y = [f"P{int(p)}" for p in d["program"]]
    deviation = (d["frac_plus"] - 0.5) * 100  # signed %

    outline = ["black" if (selected is not None and int(p) == int(selected))
               else "rgba(0,0,0,0)" for p in d["program"]]
    line_w  = [2.0 if (selected is not None and int(p) == int(selected))
               else 0.0 for p in d["program"]]

    fig = go.Figure(go.Bar(
        y=y, x=d["frac_plus"] * 100.0, orientation="h",
        marker=dict(color=deviation, colorscale="RdBu_r",
                     cmin=-5, cmax=5,
                     line=dict(color=outline, width=line_w)),
        customdata=d[["n_plus", "n_minus", "total"]].values,
        hovertemplate=(
            "%{y} — %{x:.1f}% (+)<br>"
            "+ strand: %{customdata[0]:,}<br>"
            "− strand: %{customdata[1]:,}<br>"
            "total: %{customdata[2]:,}<extra></extra>"
        ),
    ))
    fig.add_vline(x=50, line_dash="dash", line_color="black",
                   line_width=1, opacity=0.4,
                   annotation_text="50% (balanced)",
                   annotation_position="top")
    fig.update_layout(
        title="Strand of parent transcript, by program",
        xaxis=dict(title="% modules on (+) strand", range=[40, 60],
                   ticksuffix="%"),
        yaxis=dict(autorange="reversed"),
        margin=dict(l=60, r=20, t=60, b=50),
        height=max(280, 28 * len(d) + 100),
        showlegend=False,
    )
    return fig


def fig_program_tf_tissue_heatmap(
    M: pd.DataFrame, loadings: pd.Series,
    program: int, reading: str = "",
    tissue_order: str = "program_signal",
) -> go.Figure:
    """Heatmap of (top program TFs × 66 GTEx tissues), TPM color on log scale.

    M:        wide [TF × tissue] TPM, rows in H-loading-desc order.
    loadings: H-loading per TF, aligned to M.index.
    tissue_order: 'program_signal' → tissues sorted by sum log-TPM across these
                  TFs (puts the program's hot tissues together); 'alpha' →
                  alphabetical.
    """
    if tissue_order == "program_signal":
        col_score = np.log1p(M.values).sum(axis=0)
        order = np.argsort(-col_score)
        M = M.iloc[:, order]

    z = np.log10(M.values + 1.0)
    tfs = M.index.tolist()
    tissues = M.columns.tolist()

    # Per-cell hover with raw TPM and the row's H loading.
    raw = M.values
    load_col = loadings.reindex(tfs).to_numpy()
    custom = np.dstack([
        raw,
        np.broadcast_to(load_col[:, None], raw.shape),
    ])

    fig = go.Figure(go.Heatmap(
        z=z, x=tissues, y=tfs,
        customdata=custom,
        colorscale="Viridis",
        zmin=0.0,
        zmax=float(np.nanmax(z)) if np.isfinite(np.nanmax(z)) else 1.0,
        colorbar=dict(title="log10(TPM+1)", thickness=12),
        hovertemplate=(
            "TF: %{y}<br>tissue: %{x}<br>"
            "TPM: %{customdata[0]:.2f}<br>"
            "H loading: %{customdata[1]:.3f}<extra></extra>"
        ),
    ))
    title = f"P{program}"
    if reading:
        title += f" ({reading})"
    title += " — TF expression across GTEx tissues"
    fig.update_layout(
        title=title,
        xaxis_title="", yaxis_title="",
        margin=dict(l=120, r=40, t=60, b=140),
        height=max(360, 22 * len(tfs) + 200),
    )
    fig.update_xaxes(tickangle=-55, tickfont=dict(size=10), automargin=True)
    fig.update_yaxes(autorange="reversed", tickmode="linear",
                      tickfont=dict(size=10), automargin=True)
    return fig


# ---------------------------------------------------------------------------
# Per-transcript explorer  — the load-bearing custom view
# ---------------------------------------------------------------------------
def _add_gene_structure(fig, gs: pd.DataFrame, row: int, focal_tx: str = "") -> None:
    """Exon/CDS models as a genome-browser style track.

    Drawn directly under the density curve rather than at the bottom: the rug
    grows with TF count (90+ rows for a dense promoter), so a bottom-placed
    track ends up hundreds of pixels from the signal it provides context for.

    UTR and CDS are drawn at different heights, the usual thin/thick
    convention, so coding extent is readable at a glance. Neighbouring genes
    are kept and drawn muted -- a promoter sitting inside another gene is the
    case a reader most needs to see, not one to filter away.
    """
    if gs is None or gs.empty:
        return
    # One lane per TRANSCRIPT. A merged gene model hides alternative first
    # exons, which is the structure a promoter view exists to show.
    key = gs["transcript_id"].fillna("")
    order = [t for t in key.unique() if t]
    # focal transcript first, so it reads at the top of the track
    if focal_tx in order:
        order = [focal_tx] + [t for t in order if t != focal_tx]
    # TP53 has 17 coding transcripts in a 3 kb window; drawing them all makes
    # the track a third of the figure and none of them legible. The focal one
    # is always kept, and the overflow is stated rather than silently dropped.
    n_total = len(order)
    order = order[:MAX_STRUCTURE_LANES]
    lanes = {t: i for i, t in enumerate(order)}
    seen = [(t + ("  ←" if t == focal_tx else "")) for t in order]
    if n_total > len(order):
        seen[-1] += f"  (+{n_total - len(order)} more)"
    for label, g in gs.groupby(key):
        if label not in lanes:
            continue
        lane = lanes[label]
        focal = label == focal_tx
        colour = PRIMARY if focal else REFERENCE
        # thin backbone spanning the gene's extent in view
        fig.add_trace(go.Scatter(
            x=[g.local_start.min(), g.local_end.max()], y=[lane, lane],
            mode="lines", line=dict(color=colour, width=1),
            hoverinfo="skip", showlegend=False), row=row, col=1)
        for _, f in g.iterrows():
            # NMD transcripts pass the protein_coding GENE filter but are not
            # translated -- drawn hollow so they are not read as coding.
            nmd = str(f.get("transcript_biotype", "")) == "nonsense_mediated_decay"
            thick = 0.34 if f.feature == "CDS" else 0.16
            fig.add_trace(go.Scatter(
                x=[f.local_start, f.local_end, f.local_end, f.local_start,
                   f.local_start],
                y=[lane - thick, lane - thick, lane + thick, lane + thick,
                   lane - thick],
                mode="lines",
                fill="none" if nmd else "toself",
                fillcolor=None if nmd else colour,
                line=dict(width=1.2 if nmd else 0, color=colour),
                opacity=0.95 if focal else 0.45,
                hovertemplate=(f"<b>{label}</b> {f.feature}"
                               + (" · NMD" if nmd else "")
                               + (f" {int(f.exon_number)}" if f.exon_number else "")
                               + "<br>%{x:+,.0f} bp<extra></extra>"),
                showlegend=False), row=row, col=1)
    fig.update_yaxes(row=row, col=1, tickmode="array",
                     tickvals=list(range(len(seen))), ticktext=seen,
                     range=[-0.6, len(seen) - 0.4], showgrid=False)


def fig_transcript_view(peaks_df: pd.DataFrame, modules_df: pd.DataFrame,
                        tss_meta: dict,
                        score_range: tuple[int, int] = (500, 1000),
                        tf_filter: list[str] | None = None,
                        compact: bool = False,
                        gene_structure: pd.DataFrame | None = None) -> go.Figure:
    """
    Aligned view for a single TSS:
       (1) per-TSS smoothed KDE density
       (2) one row per bound TF in score_range, peak midpoints as ticks
           colored by score (Viridis 0..1000)
       (3) ALL modules as colored blocks (overview ribbon)
       (4..) one row per UNIQUE PROGRAM present at this promoter, showing
             only that program's modules — makes co-existing programs explicit

    `score_range` is a (min, max) tuple; only peaks with score in [min, max]
    appear in the rug panel. The KDE density is computed from the full peak
    set (unfiltered) so the per-TSS density never depends on the slider.

    `tf_filter` restricts the rug panel to a subset of TFs (by name). The
    KDE density and module rows are unaffected — only which TF rows render.

    `compact=True` omits the per-program rows (panels 4..) — useful for
    side-by-side comparison where vertical space is at a premium.
    """
    smin, smax = int(score_range[0]), int(score_range[1])
    use = peaks_df[(peaks_df["score"] >= smin)
                    & (peaks_df["score"] <= smax)].copy()
    n_tfs_total = use["tf"].nunique()
    if tf_filter:
        use = use[use["tf"].isin(tf_filter)]
    n_tfs = use["tf"].nunique()

    # Per-TSS density (mass=1 per TF, distributed across its peaks at this TSS)
    grid = np.zeros(LEN_GRID, dtype=np.float64)
    if not peaks_df.empty:
        tf_counts = peaks_df["tf_idx"].value_counts()
        weights = peaks_df["tf_idx"].map(1.0 / tf_counts).to_numpy()
        idx = peaks_df["local_offset"].to_numpy().astype(np.int32) + OUTER_HALF
        valid = (idx >= 0) & (idx < LEN_GRID)
        np.add.at(grid, idx[valid], weights[valid])
    smoothed = gaussian_filter1d(grid, sigma=KDE_BW)
    x = np.arange(-OUTER_HALF, OUTER_HALF + 1)

    # Per-program row layout
    if not modules_df.empty and not compact:
        prog_counts = (modules_df.groupby("dominant_program")
                                   .size()
                                   .sort_index())
        program_rows = prog_counts.index.tolist()  # ascending by program id
        # Lookup: program -> reading
        prog_reading = (modules_df.groupby("dominant_program")["program_reading"]
                                   .first().to_dict())
    else:
        program_rows = []
        prog_reading = {}

    n_prog = len(program_rows)
    n_rug_rows = max(n_tfs, 1)
    rug_h = max(0.6, min(8.0, 0.18 * n_rug_rows))

    # Subplot row heights: density 1, rugs rug_h, all-modules 0.5,
    # then n_prog rows of 0.32 each
    # Structure track sits between density and rug -- see _add_gene_structure.
    # Lanes are TRANSCRIPTS now, not genes. Counting genes here while the
    # track drew transcripts would size the row for one lane and render 17.
    n_lanes = (0 if gene_structure is None or gene_structure.empty
               else min(gene_structure.transcript_id.nunique(),
                        MAX_STRUCTURE_LANES))
    has_gs = n_lanes > 0
    gs_h = max(0.3, 0.22 * n_lanes) if has_gs else 0.0
    row_heights = ([1.0] + ([gs_h] if has_gs else []) + [rug_h, 0.5]
                   + [0.32] * n_prog)
    n_rows = 3 + n_prog + (1 if has_gs else 0)
    # Row indices derived from the same flag as row_heights. They were literals
    # (row=2 rug, row=3 modules), which silently shifts by one the moment a
    # track is inserted above them.
    row_gs = 2 if has_gs else None
    row_rug = 3 if has_gs else 2
    row_mod = row_rug + 1

    rug_title = (f"TFs binding at score ∈ [{smin}, {smax}] (n={n_tfs})"
                  if smin > 0 or smax < 1000 else
                  f"TFs binding (n={n_tfs})")
    if tf_filter and n_tfs_total != n_tfs:
        rug_title += f" — filtered from {n_tfs_total}"
    titles = ["KDE density (per-TF mass=1)"]
    if has_gs:
        titles.append("protein-coding structure (thick = CDS, thin = UTR)")
    titles += [rug_title,
              "all modules — colored by k=10 dominant program"]
    titles += [f"P{p} only — {prog_reading.get(p, '')}" for p in program_rows]

    fig = make_subplots(
        rows=n_rows, cols=1, shared_xaxes=True,
        row_heights=row_heights,
        vertical_spacing=0.025,
        subplot_titles=titles,
    )

    # (1) Density curve
    fig.add_trace(go.Scatter(
        x=x, y=smoothed, mode="lines", name="density",
        line=dict(color=PRIMARY, width=2),
        fill="tozeroy", fillcolor="rgba(15,118,110,0.20)",
        hovertemplate="bp: %{x}<br>density: %{y:.4f}<extra></extra>",
        showlegend=False,
    ), row=1, col=1)
    for _, m in modules_df.iterrows():
        # dominant_program is NULL when programs come from the genome layer
        # rather than a promoter factorization. int(NA) raises and would take
        # the whole profile down, so an unassigned module shades neutrally.
        raw = m.get("dominant_program")
        fillcolor = (PROGRAM_COLORS[(int(raw) - 1) % len(PROGRAM_COLORS)]
                     if pd.notna(raw) else REFERENCE)
        fig.add_vrect(
            x0=m["lo_offset"], x1=m["hi_offset"],
            fillcolor=fillcolor,
            opacity=0.18, line_width=0, layer="below", row=1, col=1,
        )

    if has_gs:
        _add_gene_structure(fig, gene_structure, row_gs,
                            focal_tx=str(tss_meta.get("transcript_id", "")))

    # (2) TF rug rows — TF identity = color, one solid hue per row.
    # We attempted score-as-opacity and score-as-marker-size, but both
    # encodings cause visual ambiguity: alpha-blending shifts the
    # perceived hue, and per-size anti-aliasing changes apparent
    # saturation. Both made "different scores" read as "different
    # colors" within a single TF row. The honest answer is to keep the
    # visual unambiguous (color = TF identity, full stop) and surface
    # score via the slider (filtering) + the hover (exact value).
    if n_tfs:
        # Sort by mean position DESCENDING so that — combined with plotly's
        # y=0-at-bottom default — the most-upstream TFs render at the top
        # of the rug panel and the most-downstream at the bottom.
        tf_order = (use.groupby("tf")["local_offset"].mean()
                       .sort_values(ascending=False).index.tolist())
        tf_y = {tf: i for i, tf in enumerate(tf_order)}
        use_sorted = use.assign(_y=use["tf"].map(tf_y))

        # Per-TF color: cycle through a high-contrast qualitative palette.
        # Same color may repeat across distant rows (no good palette covers
        # ~300 categories) but adjacent rows always differ.
        from plotly.colors import qualitative
        palette = (qualitative.Light24 + qualitative.Dark24
                    + qualitative.Alphabet)
        tf_color = {tf: palette[i % len(palette)]
                    for i, tf in enumerate(tf_order)}

        # One trace per TF, scalar color/size. Slightly smaller markers
        # for many-TF rugs so they don't crowd adjacent rows.
        rug_size = 10 if n_tfs <= 30 else 8
        for tf in tf_order:
            sub = use_sorted[use_sorted["tf"] == tf]
            if sub.empty:
                continue
            fig.add_trace(go.Scatter(
                x=sub["local_offset"], y=sub["_y"],
                mode="markers", marker_symbol="line-ns-open",
                marker=dict(
                    size=rug_size,
                    line=dict(width=1.8, color=tf_color[tf]),
                ),
                text=sub["tf"],
                customdata=sub["score"],
                hovertemplate="<b>%{text}</b><br>bp: %{x}<br>"
                              "score: %{customdata}<extra></extra>",
                showlegend=False,
            ), row=row_rug, col=1)
        rug_label_size = 12 if n_tfs <= 12 else (10 if n_tfs <= 40 else 8)
        fig.update_yaxes(
            tickvals=list(range(len(tf_order))),
            ticktext=tf_order,
            range=[-0.6, len(tf_order) - 0.4],
            row=row_rug, col=1, automargin=True,
            tickmode="array", tickfont=dict(size=rug_label_size),
            showgrid=True, gridcolor="rgba(0,0,0,0.14)", gridwidth=1,
            zeroline=False,
        )
        # Alternating-row background bands so adjacent TF rows are
        # unambiguously separable even when ticks cluster at the same x.
        for i in range(len(tf_order)):
            if i % 2 == 1:
                fig.add_hrect(y0=i - 0.5, y1=i + 0.5,
                               fillcolor="rgba(0,0,0,0.035)",
                               line_width=0, layer="below",
                               row=row_rug, col=1)

    # (3) All modules ribbon (overview)
    _add_module_blocks(fig, modules_df, row=row_mod)

    # (4..) Per-program rows
    for i, p in enumerate(program_rows):
        sub = modules_df[modules_df["dominant_program"] == p]
        _add_module_blocks(fig, sub, row=4 + i)
        # Y-axis label as P{n} on the left
        fig.update_yaxes(
            title_text=f"<b>P{p}</b>", row=4 + i, col=1,
            visible=True, range=[-1, 1],
            showticklabels=False, showgrid=False,
            title_standoff=4,
        )

    fig.update_xaxes(range=[-OUTER_HALF, OUTER_HALF])
    # Vertical zoom is meaningless here and actively harmful: every row is a
    # fixed lane -- one per TF, one per gene, one per module -- so Plotly's
    # zoom-out button, which scales BOTH axes, squeezed the TF rug into a
    # fraction of its height and made the rows unreadable. Position along the
    # x-axis is the only dimension worth zooming.
    fig.update_yaxes(fixedrange=True)
    fig.update_xaxes(title_text="bp from TSS (txn-oriented)",
                      row=n_rows, col=1)
    fig.update_yaxes(title_text="density", row=1, col=1, showgrid=False)
    fig.update_yaxes(title_text="all", row=row_mod, col=1,
                      visible=True, range=[-1, 1],
                      showticklabels=False, showgrid=False, title_standoff=4)
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1,
                   opacity=0.7, row="all")

    title = (f"<b>{tss_meta.get('gene_name','?')}</b>  "
             f"({tss_meta.get('transcript_id','?')})  •  "
             f"chr{tss_meta.get('chrom','?')}:{tss_meta.get('tss','?')} "
             f"({tss_meta.get('strand','?')})  •  "
             f"{n_prog} program{'s' if n_prog != 1 else ''} present "
             f"({len(modules_df)} module{'s' if len(modules_df) != 1 else ''})")

    height = int(60 + 220 + max(120, 18 * n_tfs) + 60 + n_prog * 55 + 60)
    fig.update_layout(
        title=title, height=height,
        margin=dict(l=100, r=20, t=70, b=50),
        showlegend=False, barmode="overlay", bargap=0,
    )
    # Tighten subplot title sizes
    for ann in fig.layout.annotations:
        ann.font.size = 11
    return fig


def _add_module_blocks(fig: go.Figure, df: pd.DataFrame, row: int) -> None:
    """Add module rectangles (one bar per module) to a given subplot row.

    dominant_program may be NULL: modules exist independently of any
    factorization, and on a build whose programs come from the genome layer
    they carry no promoter-program assignment. int(NA) raises, so the module
    would take the whole profile down -- the blocks are the main content of
    that panel, so they render unassigned rather than not at all.
    """
    for _, m in df.iterrows():
        raw = m.get("dominant_program")
        has_prog = pd.notna(raw)
        p = int(raw) if has_prog else None
        colour = (PROGRAM_COLORS[(p - 1) % len(PROGRAM_COLORS)] if has_prog
                  else REFERENCE)
        label = f"P{p}" if has_prog else "module"
        wt = m.get("dominant_weight")
        wt_txt = f"<br>weight: {wt:.3f}" if pd.notna(wt) else ""
        reading = m.get("program_reading", "")
        reading_txt = f" ({reading})" if has_prog and pd.notna(reading) else ""
        fig.add_trace(go.Bar(
            x=[m["hi_offset"] - m["lo_offset"]],
            y=[0],
            base=m["lo_offset"],
            orientation="h",
            marker=dict(color=colour, line=dict(width=1, color="black")),
            name=label,
            hovertemplate=(
                f"module {int(m['module_id'])}<br>"
                f"{label}{reading_txt}<br>"
                f"bp: {int(m['lo_offset'])}–{int(m['hi_offset'])} "
                f"(width {int(m['width'])})"
                f"{wt_txt}<extra></extra>"
            ),
            showlegend=False,
        ), row=row, col=1)


# ---------------------------------------------------------------------------
# Per-TF explorer
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# GTEx
# ---------------------------------------------------------------------------
def fig_gtex_expression_bar(stats: pd.DataFrame, title_prefix: str = "") -> go.Figure:
    """Bar plot of per-tissue mean TPM with IQR error bars, sorted desc."""
    if stats.empty:
        return go.Figure()
    df = stats.sort_values("mean", ascending=False).copy()
    err_minus = (df["mean"] - df["q1"]).clip(lower=0).values
    err_plus  = (df["q3"] - df["mean"]).clip(lower=0).values
    fig = go.Figure(go.Bar(
        x=df["tissue"], y=df["mean"],
        error_y=dict(type="data", symmetric=False,
                      array=err_plus, arrayminus=err_minus,
                      color="rgba(60,60,60,0.55)", thickness=1, width=2),
        marker=dict(color=df["mean"], colorscale="YlGnBu",
                     line=dict(color="rgba(0,0,0,0.4)", width=0.4)),
        hovertemplate=(
            "tissue: %{x}<br>"
            "mean: %{y:.2f}<br>"
            "median: %{customdata[0]:.2f}<br>"
            "Q1: %{customdata[1]:.2f}<br>"
            "Q3: %{customdata[2]:.2f}<br>"
            "std: %{customdata[3]:.2f}<br>"
            "n samples: %{customdata[4]}<extra></extra>"
        ),
        customdata=df[["median", "q1", "q3", "std", "n_samples"]].values,
    ))
    fig.update_layout(
        title=f"{title_prefix}GTEx tissue expression (mean TPM, IQR error bars)",
        xaxis_title="tissue", yaxis_title="TPM",
        margin=dict(l=60, r=20, t=60, b=140),
        height=460, hovermode="x",
    )
    fig.update_xaxes(tickangle=60, tickfont=dict(size=9), tickmode="linear")
    return fig


def fig_gtex_expression_with_modules(stats: pd.DataFrame,
                                       activity: pd.DataFrame,
                                       modules_df: pd.DataFrame,
                                       title_prefix: str = "") -> go.Figure:
    """Two-panel figure with shared x-axis:
       (top)    bar plot of per-tissue mean TPM with IQR error bars
       (bottom) heatmap of (module × tissue) mean TF TPM
       Tissues sorted by bar TPM desc; module labels colored by their k=10
       dominant program; every module label is shown."""
    if stats.empty:
        return go.Figure()
    bar_df = stats.sort_values("mean", ascending=False).reset_index(drop=True)
    tissues = bar_df["tissue"].tolist()

    # Build the heatmap matrix in the bar's tissue order.
    has_heatmap = (not activity.empty) and (not modules_df.empty)
    if has_heatmap:
        wide = (activity
                 .pivot_table(index=["module_id", "center_offset"],
                              columns="tissue", values="mean_tf_tpm",
                              aggfunc="first")
                 .sort_index(level="center_offset")
                 .reindex(columns=tissues))
        if wide.empty:
            has_heatmap = False

    if has_heatmap:
        mid_to_prog = dict(zip(modules_df["module_id"],
                                modules_df["dominant_program"]))
        # HTML-styled tick labels = colored by program
        mod_labels = []
        for (mid, center) in wide.index:
            # .get(key, 0) does NOT protect here: the key exists, its value is
            # NA, so the default never applies and int(NA) raises. Programs are
            # NULL whenever they come from the genome layer rather than a
            # promoter factorization.
            raw = mid_to_prog.get(int(mid))
            prog = int(raw) if pd.notna(raw) else 0
            color = (PROGRAM_COLORS[(prog - 1) % len(PROGRAM_COLORS)]
                     if prog else "#444")
            tag = f" · P{prog}" if prog else ""
            mod_labels.append(
                f"<span style='color:{color};font-weight:600'>"
                f"M{int(mid)} ({int(center):+d} bp{tag})</span>"
            )

    n_mod = len(wide.index) if has_heatmap else 0
    bar_h    = 280
    heat_h   = max(120, 28 * n_mod + 80)
    total_h  = bar_h + heat_h + 110  # padding for titles/x-labels

    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[bar_h, heat_h], vertical_spacing=0.06,
        subplot_titles=(f"{title_prefix}GTEx tissue expression "
                        "(mean TPM, IQR error bars)",
                        "Module × tissue activity — mean TPM of the "
                        "module's assigned TFs"),
    )

    # ----- bar (row 1) -------------------------------------------------------
    err_minus = (bar_df["mean"] - bar_df["q1"]).clip(lower=0).values
    err_plus  = (bar_df["q3"]   - bar_df["mean"]).clip(lower=0).values
    fig.add_trace(go.Bar(
        x=bar_df["tissue"], y=bar_df["mean"],
        error_y=dict(type="data", symmetric=False,
                      array=err_plus, arrayminus=err_minus,
                      color="rgba(60,60,60,0.55)", thickness=1, width=2),
        marker=dict(color=bar_df["mean"], colorscale="YlGnBu",
                     line=dict(color="rgba(0,0,0,0.4)", width=0.4),
                     showscale=False),
        hovertemplate=(
            "tissue: %{x}<br>mean: %{y:.2f}<br>"
            "median: %{customdata[0]:.2f}<br>"
            "Q1: %{customdata[1]:.2f}<br>Q3: %{customdata[2]:.2f}<br>"
            "std: %{customdata[3]:.2f}<br>"
            "n samples: %{customdata[4]}<extra></extra>"
        ),
        customdata=bar_df[["median", "q1", "q3", "std", "n_samples"]].values,
        showlegend=False,
    ), row=1, col=1)
    fig.update_yaxes(title_text="TPM", row=1, col=1)

    # ----- heatmap (row 2) ---------------------------------------------------
    if has_heatmap:
        z = wide.to_numpy()
        fig.add_trace(go.Heatmap(
            z=z, x=tissues, y=mod_labels,
            colorscale="YlGnBu",
            colorbar=dict(title="mean TF TPM", thickness=12,
                           y=heat_h / (2 * total_h),
                           len=heat_h / total_h * 0.95),
            hovertemplate=("module: %{y}<br>tissue: %{x}<br>"
                            "mean TF TPM: %{z:.2f}<extra></extra>"),
        ), row=2, col=1)
        fig.update_yaxes(autorange="reversed",
                          tickmode="linear", tickfont=dict(size=10),
                          automargin=True, row=2, col=1)

    fig.update_xaxes(tickangle=60, tickfont=dict(size=9),
                      tickmode="linear", row=2, col=1)
    fig.update_xaxes(showticklabels=False, row=1, col=1)
    fig.update_layout(
        height=total_h,
        margin=dict(l=160, r=40, t=60, b=140),
        bargap=0.15, showlegend=False,
    )
    for ann in fig.layout.annotations:
        ann.font.size = 12
    return fig


def fig_gtex_compare(stats_a: pd.DataFrame, stats_b: pd.DataFrame,
                     label_a: str, label_b: str) -> go.Figure:
    """Overlay two transcripts' per-tissue mean TPM across GTEx, tissues
    sorted by the combined mean. Returns an empty figure if both inputs
    are empty."""
    if stats_a.empty and stats_b.empty:
        return go.Figure()

    a = (stats_a[["tissue", "mean"]].rename(columns={"mean": "a"})
         if not stats_a.empty else pd.DataFrame(columns=["tissue", "a"]))
    b = (stats_b[["tissue", "mean"]].rename(columns={"mean": "b"})
         if not stats_b.empty else pd.DataFrame(columns=["tissue", "b"]))
    merged = a.merge(b, on="tissue", how="outer").fillna(0.0)
    merged["sum"] = merged["a"] + merged["b"]
    merged = merged.sort_values("sum", ascending=False).reset_index(drop=True)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=merged["tissue"], y=merged["a"], name=label_a,
        marker_color="#1f77b4", opacity=0.85,
        hovertemplate=f"<b>{label_a}</b><br>%{{x}}<br>"
                       f"mean TPM: %{{y:.2f}}<extra></extra>",
    ))
    fig.add_trace(go.Bar(
        x=merged["tissue"], y=merged["b"], name=label_b,
        marker_color="#d62728", opacity=0.85,
        hovertemplate=f"<b>{label_b}</b><br>%{{x}}<br>"
                       f"mean TPM: %{{y:.2f}}<extra></extra>",
    ))
    fig.update_layout(
        title="GTEx tissue expression — paired comparison "
              "(tissues sorted by combined mean TPM)",
        xaxis_title="tissue", yaxis_title="mean TPM",
        barmode="group", bargap=0.18, bargroupgap=0.05,
        height=460,
        margin=dict(l=60, r=20, t=60, b=140),
        legend=dict(orientation="h", x=0.5, xanchor="center",
                    y=1.06, yanchor="bottom"),
    )
    fig.update_xaxes(tickangle=60, tickfont=dict(size=9), tickmode="linear")
    return fig


def fig_depmap_lineage_bar(lineage_df: pd.DataFrame, gene: str) -> go.Figure:
    """Bar plot of median Chronos per lineage. More-negative = more
    essential. Colored diverging around 0; line at -1 (essential threshold)."""
    if lineage_df.empty:
        return go.Figure()
    df = lineage_df.sort_values("median_chronos").reset_index(drop=True)
    fig = go.Figure(go.Bar(
        x=df["lineage"], y=df["median_chronos"],
        marker=dict(color=df["median_chronos"], colorscale="RdBu",
                     cmin=-2, cmax=2,
                     line=dict(color="rgba(0,0,0,0.4)", width=0.4)),
        hovertemplate=(
            "lineage: %{x}<br>median Chronos: %{y:.2f}<br>"
            "n cell lines: %{customdata[0]}<br>"
            "fraction essential (<-1): %{customdata[1]:.2f}<extra></extra>"
        ),
        customdata=df[["n_lines", "frac_essential"]].values,
    ))
    fig.add_hline(y=-1.0, line_dash="dash", line_color="red", line_width=1,
                   opacity=0.7,
                   annotation_text="essential", annotation_position="right")
    fig.add_hline(y=0, line_color="black", line_width=0.5, opacity=0.4)
    fig.update_layout(
        title=f"{gene} — DepMap CRISPR essentiality (median Chronos per lineage)",
        xaxis_title="lineage", yaxis_title="median Chronos (lower = more essential)",
        margin=dict(l=60, r=20, t=60, b=140),
        height=420, showlegend=False,
    )
    fig.update_xaxes(tickangle=60, tickfont=dict(size=9), tickmode="linear")
    return fig


def fig_depmap_tf_target_corr_bar(
    df: pd.DataFrame, target: str, top_n: int = 25,
) -> go.Figure:
    """Horizontal bar: top TFs by |r(TF Chronos, target expression)| across
    DepMap cell lines. Bars diverge around 0, colored RdBu_r (red = +r,
    blue = -r). Strong negative r is the most mechanistic signal: cells
    that express the target also depend on the TF."""
    if df.empty:
        return go.Figure()
    d = df.head(top_n).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=d["r"], y=d["tf"], orientation="h",
        marker=dict(color=d["r"], colorscale="RdBu_r",
                     cmin=-1, cmax=1,
                     line=dict(color="rgba(0,0,0,0.4)", width=0.4)),
        customdata=d[["n_cell_lines"]].values,
        hovertemplate=("TF: %{y}<br>r: %{x:.3f}<br>"
                       "n cell lines: %{customdata[0]}<extra></extra>"),
    ))
    fig.add_vline(x=0, line_color="black", line_width=0.5, opacity=0.4)
    fig.update_layout(
        title=(f"TF–target essentiality coupling — top {len(d)} TFs<br>"
               f"<sub>r(TF Chronos, {target} expression) across DepMap cell lines</sub>"),
        xaxis_title="Pearson r", yaxis_title="",
        xaxis=dict(range=[-1, 1]),
        margin=dict(l=80, r=20, t=80, b=50),
        height=max(380, 22 * len(d) + 120),
        showlegend=False,
    )
    fig.update_yaxes(tickmode="linear", tickfont=dict(size=10),
                      automargin=True)
    return fig


def fig_depmap_tf_target_scatter(
    pair_df: pd.DataFrame, tf: str, target: str, r: float | None = None,
) -> go.Figure:
    """Per-cell-line scatter: TF Chronos (x) vs target log10(TPM+1) (y),
    colored by lineage. One point per cell line in the intersection of the
    two matrices. Best-fit OLS line shown in grey."""
    if pair_df.empty:
        return go.Figure()
    d = pair_df.copy()
    if "lineage" not in d.columns:
        d["lineage"] = "(unknown)"
    d["lineage"] = d["lineage"].fillna("(unknown)").astype(str)

    fig = go.Figure()
    for lin, g in d.groupby("lineage"):
        fig.add_trace(go.Scatter(
            x=g["tf_chronos"], y=g["target_expr"],
            mode="markers", name=lin,
            marker=dict(size=6, opacity=0.75,
                         line=dict(color="rgba(0,0,0,0.3)", width=0.3)),
            customdata=g[["cell_line"]].fillna("").values if "cell_line" in g.columns else None,
            hovertemplate=(
                "cell line: %{customdata[0]}<br>"
                "lineage: " + lin + "<br>"
                f"{tf} Chronos: %{{x:.2f}}<br>"
                f"{target} log10(TPM+1): %{{y:.2f}}<extra></extra>"
            ),
        ))

    # OLS fit line over all points.
    x = d["tf_chronos"].to_numpy(); y = d["target_expr"].to_numpy()
    if len(x) >= 3 and np.std(x) > 0:
        slope, intercept = np.polyfit(x, y, 1)
        xf = np.array([x.min(), x.max()])
        fig.add_trace(go.Scatter(
            x=xf, y=slope * xf + intercept, mode="lines",
            line=dict(color="rgba(60,60,60,0.6)", width=1.5, dash="dot"),
            name="OLS", showlegend=False, hoverinfo="skip",
        ))

    title = f"{tf} essentiality vs {target} expression (n={len(d)} cell lines)"
    if r is not None and np.isfinite(r):
        title += f"   r={r:.2f}"
    fig.update_layout(
        title=title,
        xaxis_title=f"{tf} Chronos (lower = more essential)",
        yaxis_title=f"{target} log10(TPM+1)",
        margin=dict(l=70, r=20, t=60, b=60),
        height=480,
        legend=dict(font=dict(size=9), itemsizing="constant"),
    )
    fig.add_vline(x=-1.0, line_dash="dash", line_color="red", line_width=1,
                   opacity=0.5,
                   annotation_text="essential", annotation_position="top")
    fig.add_vline(x=0, line_color="black", line_width=0.4, opacity=0.3)
    return fig


def fig_tf_aggregate_profile(matrix: pd.DataFrame, tf: str,
                              other_tfs: list[str] = (),
                              cluster_members: list[str] | None = None,
                              cluster_label: str = "") -> go.Figure:
    """Single TF's aggregate profile, with optional comparison TFs faded and
    an optional shaded mean-of-cluster band overlay."""
    if tf not in matrix.index:
        return go.Figure()
    x = matrix.columns.values
    fig = go.Figure()

    # Cluster-mean ± SEM band drawn behind everything
    if cluster_members:
        members = [t for t in cluster_members if t in matrix.index]
        if len(members) >= 2:
            sub = matrix.loc[members]
            mean = sub.values.mean(axis=0)
            sem  = sub.values.std(axis=0, ddof=1) / np.sqrt(len(sub))
            fig.add_trace(go.Scatter(
                x=np.concatenate([x, x[::-1]]),
                y=np.concatenate([mean + sem, (mean - sem)[::-1]]),
                fill="toself", fillcolor="rgba(120,120,120,0.18)",
                line=dict(color="rgba(0,0,0,0)"),
                name=f"{cluster_label} (n={len(sub)}) ± SEM",
                hoverinfo="skip", showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=x, y=mean, mode="lines",
                line=dict(color="dimgray", width=1.5, dash="dot"),
                name=f"{cluster_label} mean",
            ))

    palette = pc.qualitative.D3
    for i, ot in enumerate(other_tfs):
        if ot in matrix.index and ot != tf:
            fig.add_trace(go.Scatter(
                x=x, y=matrix.loc[ot].values, mode="lines",
                name=ot, line=dict(color=palette[i % len(palette)],
                                    width=1, dash="dot"),
                opacity=0.6,
            ))
    fig.add_trace(go.Scatter(
        x=x, y=matrix.loc[tf].values, mode="lines",
        name=tf, line=dict(color=PRIMARY, width=2.5),
    ))
    fig.add_vline(x=0, line_dash="dash", line_color="red", line_width=1, opacity=0.6)
    fig.update_layout(
        title=f"{tf} — aggregate binding profile",
        xaxis_title="bp from TSS", yaxis_title="mean binary occupancy",
        height=420, margin=dict(l=60, r=20, t=60, b=50),
        hovermode="x unified",
    )
    return fig


def fig_tf_program_loadings(loadings_df: pd.DataFrame, tf: str) -> go.Figure:
    """Bar of this TF's loading across the 10 programs."""
    df = loadings_df.set_index("program").reindex(range(1, 11)).reset_index()
    df["loading"] = df["loading"].fillna(0.0)
    fig = go.Figure(go.Bar(
        x=[f"P{p}" for p in df["program"]],
        y=df["loading"],
        marker_color=[PROGRAM_COLORS[(int(p) - 1) % 10] for p in df["program"]],
        hovertemplate="%{x}<br>H loading: %{y:.3f}<br>rank: %{customdata}<extra></extra>",
        customdata=df["rank"].fillna("—"),
    ))
    fig.update_layout(
        title=f"{tf} — loading across k=10 programs",
        xaxis_title="program", yaxis_title="H loading",
        height=220, margin=dict(l=60, r=20, t=50, b=40),
    )
    return fig


# ---------------------------------------------------------------------------
# Genome-wide element views (distal-capable)
# ---------------------------------------------------------------------------
# The promoter profile above is fixed to +/-OUTER_HALF and cannot show a distal
# element: SOX2's run from 11 kb to 535 kb, a 350x span. Rendering both on one
# axis would compress the promoter region -- the view this site exists for --
# to a few pixels, so distal elements get their own zoom hierarchy.
#
# Zoom levels are QUANTILES of each gene's own element distances, not fixed
# widths. A fixed neighbourhood width would be an arbitrary constant that is
# wrong for most genes (median |dist| is 277 bp at the promoter, 4.8 kb
# proximal, 72 kb distal), and it would clip sprawling genes while wasting
# canvas on compact ones. Quantiles also make the level labels informative:
# "50% of this gene's elements" says something about the gene; "200 kb" does
# not.

# 28 families need more than the 10-colour PROGRAM_COLORS palette above, which
# would silently recycle hues and imply two families are the same one.
FAMILY_COLORS = (pc.qualitative.Dark24 + pc.qualitative.Light24)

# Transcript lanes drawn in the structure track before overflow is summarised.
MAX_STRUCTURE_LANES = 8


def gene_zoom_levels(dist, promoter_half: int = OUTER_HALF,
                     quantiles=(0.25, 0.50, 0.75, 1.00),
                     min_for_levels: int = 8) -> list[dict]:
    """Per-side quantile zoom windows for one gene's elements.

    `dist` is signed distance to the gene's TSS. Windows are computed PER SIDE
    because element distributions are usually lopsided -- a gene may carry 60
    elements upstream and 5 downstream, and a symmetric window would spend half
    the canvas on nothing.

    Degenerate genes are collapsed rather than padded out: the median gene has
    only 5 promoter+proximal elements and 1,419 have exactly one, where five
    quantile levels would render as five near-identical pictures. Below
    `min_for_levels` the caller gets promoter + one full-span view.
    """
    d = np.asarray(dist, dtype=float)
    d = d[np.isfinite(d)]
    levels = [{"label": "promoter", "lo": -promoter_half, "hi": promoter_half,
               "frac": None}]
    if d.size == 0:
        return levels
    up, dn = -d[d < 0], d[d >= 0]          # magnitudes each side
    if d.size < min_for_levels:
        lo = -(up.max() if up.size else promoter_half)
        hi = dn.max() if dn.size else promoter_half
        levels.append({"label": "all", "lo": float(lo), "hi": float(hi),
                       "frac": 1.0})
        return levels
    for q in quantiles:
        lo = -float(np.quantile(up, q)) if up.size else -promoter_half
        hi = float(np.quantile(dn, q)) if dn.size else promoter_half
        # never zoom in past the promoter box, or the levels invert
        lo, hi = min(lo, -promoter_half), max(hi, promoter_half)
        label = "all" if q >= 1.0 else f"{int(q*100)}%"
        if levels and abs(lo - levels[-1]["lo"]) < 1 and abs(hi - levels[-1]["hi"]) < 1:
            continue                        # identical to the previous level
        levels.append({"label": label, "lo": lo, "hi": hi, "frac": float(q)})
    return levels


def fig_gene_neighbourhood(el: pd.DataFrame, lo: float, hi: float,
                           gene: str = "", promoter_half: int = OUTER_HALF,
                           family_labels: dict | None = None) -> go.Figure:
    """Elements on a genomic axis, coloured by family, promoter region boxed.

    Expects columns: dist_to_tss, n_tfs_assigned, stratum, family,
    n_tss_comparably_close, substantive.

    Two columns are rendered rather than dropped, because omitting them would
    overstate what the view shows. `n_tss_comparably_close` >= 2 marks an
    element whose nearest gene is ambiguous -- 56.6% of distal elements have a
    rival TSS within twice the distance, so listing them under a gene without
    the caveat asserts a regulatory link the data does not support. Elements
    whose program is not `substantive` are drawn hollow, so a program pinned to
    three elements does not carry the same visual authority as PRC2.
    """
    v = el[(el.dist_to_tss >= lo) & (el.dist_to_tss <= hi)].copy()
    fig = go.Figure()
    # promoter region, for orientation at every zoom level
    fig.add_vrect(x0=-promoter_half, x1=promoter_half, fillcolor=PRIMARY,
                  opacity=0.10, line_width=0, layer="below",
                  annotation_text="promoter", annotation_position="top left")
    fig.add_vline(x=0, line_width=1, line_dash="dot", line_color=REFERENCE)

    if len(v):
        fams = sorted(v.family.dropna().unique())
        for f in fams:
            g = v[v.family == f]
            name = (family_labels or {}).get(int(f), f"family {int(f)}")
            solid = g[g.substantive.astype(bool)]
            hollow = g[~g.substantive.astype(bool)]
            colour = FAMILY_COLORS[int(f) % len(FAMILY_COLORS)]
            for part, marker in ((solid, dict(size=9, color=colour,
                                              line=dict(width=0))),
                                 (hollow, dict(size=9, color="rgba(0,0,0,0)",
                                               line=dict(width=1.6,
                                                         color=colour)))):
                if not len(part):
                    continue
                amb = part.n_tss_comparably_close.to_numpy() >= 2
                fig.add_trace(go.Scatter(
                    x=part.dist_to_tss, y=part.n_tfs_assigned,
                    mode="markers", name=name,
                    legendgroup=name,
                    showlegend=part is solid or not len(solid),
                    marker=marker,
                    customdata=np.stack([part.stratum,
                                         part.n_tss_comparably_close,
                                         np.where(amb, "ambiguous", "unique")],
                                        axis=-1),
                    hovertemplate=(
                        f"<b>{name}</b><br>%{{x:+,.0f}} bp from TSS<br>"
                        "%{y} TFs assigned<br>%{customdata[0]}<br>"
                        "gene link: %{customdata[2]} "
                        "(%{customdata[1]} TSS within 2x)<extra></extra>")))
    fig.update_layout(
        title=(f"{gene}: elements in view" if gene else "elements in view"),
        xaxis_title="distance from TSS (bp)",
        yaxis_title="TFs assigned",
        height=380, margin=dict(l=60, r=20, t=50, b=50),
        legend=dict(font=dict(size=10)))
    fig.update_xaxes(range=[lo, hi])
    return fig


def fig_program_distance(hist: pd.DataFrame, program: int,
                         bins: int = 40) -> go.Figure:
    """Signed, log-scaled distance distribution for one program's elements.

    Log-scaled because these span three orders of magnitude -- PRC2's distal
    elements have a p90 of 1.67 Mb while its promoter elements sit at 445 bp,
    and a linear axis would put every promoter element in a single bar. Signed
    because upstream and downstream are not interchangeable.
    """
    fig = go.Figure()
    if not hist.empty:
        # bin index -> signed log10 bp, inverted from the SQL binning
        centre = hist["bin"].to_numpy() * 12.0 / bins
        bp = np.sign(centre) * (10 ** np.abs(centre))
        fig.add_trace(go.Bar(
            x=centre, y=hist["n"], marker_color=PRIMARY,
            customdata=bp,
            hovertemplate="%{customdata:+,.0f} bp<br>%{y:,} elements<extra></extra>"))
    ticks = [-6, -5, -4, -3, 0, 3, 4, 5, 6]
    fig.update_layout(
        title=f"Program {program}: element position relative to TSS",
        xaxis_title="distance from TSS (log scale, signed)",
        yaxis_title="elements", height=300, bargap=0.02,
        margin=dict(l=60, r=20, t=50, b=50), showlegend=False)
    fig.update_xaxes(tickmode="array", tickvals=ticks,
                     ticktext=["−1Mb", "−100kb", "−10kb", "−1kb", "TSS",
                               "1kb", "10kb", "100kb", "1Mb"])
    return fig
