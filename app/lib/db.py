"""DuckDB connection + cached query helpers for the Human Promoter Atlas."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional

import duckdb
import numpy as np
import pandas as pd
import streamlit as st


# Resolve data dir relative to repo root (app/lib/db.py -> ../../data/)
APP_DIR  = Path(__file__).resolve().parents[1]
REPO_DIR = APP_DIR.parent
DATA_DIR = REPO_DIR / "data"
DB_PATH  = DATA_DIR / "canonical_promoter.duckdb"
MANIFEST_PATH = DATA_DIR / "manifest.json"
AGG_DIR    = DATA_DIR / "aggregate"
GTEX_DIR   = DATA_DIR / "gtex"
DEPMAP_DIR = DATA_DIR / "depmap"


@st.cache_resource(show_spinner=False)
def get_con() -> duckdb.DuckDBPyConnection:
    if not DB_PATH.exists():
        raise FileNotFoundError(
            f"DuckDB not found at {DB_PATH}. "
            f"Run `python data/build_app_db.py` first."
        )
    return duckdb.connect(str(DB_PATH), read_only=True)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_manifest() -> dict:
    if not MANIFEST_PATH.exists():
        return {}
    return json.loads(MANIFEST_PATH.read_text())


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def build_facts() -> dict:
    """Counts and build axes for the prose in the Methods / Programs tabs.

    These were literals in the tab sources ("1,304 TFs", "76,999 x 1,304",
    "score >= 500"), which is fine until a rebuild moves them: after the
    assignment threshold was recalibrated, the Methods tab described a build
    the app was not serving. Every value here comes from the manifest that
    `data/build_app_db.py` writes alongside the DuckDB, so the text follows
    the data.

    Missing values come back as ``None`` rather than a guess; callers phrase
    around them. `_fmt` is the formatter to use for display.
    """
    m = load_manifest()
    counts = m.get("counts") or {}
    build = m.get("build") or {}
    return {
        "n_tf": counts.get("n_tf"),
        "n_tss": counts.get("n_tss"),
        "n_modules": counts.get("n_modules"),
        "n_programs": counts.get("n_programs") or m.get("k_canonical"),
        "tier": build.get("tier"),
        "qvalue": build.get("qvalue"),
        "tf_set": build.get("tf_set"),
        "min_score_assign": build.get("min_score_assign"),
        "k_canonical": m.get("k_canonical"),
    }


def fmt_count(n: int | None, unknown: str = "?") -> str:
    """Thousands-separated, or a placeholder when the manifest lacks the key."""
    return f"{n:,}" if isinstance(n, int) else unknown


# ---------------------------------------------------------------------------
# Lookups
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def list_genes() -> list[str]:
    return get_con().execute(
        "SELECT DISTINCT gene_name FROM tss WHERE gene_name <> '' ORDER BY gene_name"
    ).df()["gene_name"].tolist()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def list_transcripts() -> list[str]:
    return get_con().execute(
        "SELECT transcript_id FROM tss ORDER BY transcript_id"
    ).df()["transcript_id"].tolist()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def list_tfs() -> list[str]:
    return get_con().execute(
        "SELECT tf FROM tf ORDER BY tf"
    ).df()["tf"].tolist()


# ---------------------------------------------------------------------------
# Per-transcript queries (the hot path)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=3600, show_spinner=False)
def get_tss_meta(transcript_id: str) -> Optional[dict]:
    df = get_con().execute(
        "SELECT * FROM tss WHERE transcript_id = ?", [transcript_id]
    ).df()
    return df.iloc[0].to_dict() if len(df) else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_transcripts_for_gene(gene_name: str) -> pd.DataFrame:
    return get_con().execute(
        "SELECT transcript_id, chrom, tss, strand FROM tss "
        "WHERE gene_name = ? ORDER BY transcript_id",
        [gene_name],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def get_modules_for_transcript(transcript_id: str) -> pd.DataFrame:
    return get_con().execute(
        """
        SELECT m.module_id, m.module_local_idx, m.lo_offset, m.hi_offset,
               m.center_offset, m.width, m.n_peaks_in,
               m.n_tfs_supporting, m.n_tfs_assigned,
               mp.dominant_program, mp.dominant_weight,
               p.reading AS program_reading, p.top_tfs AS program_top_tfs
        -- LEFT joins on purpose. These were inner joins, which silently
        -- returned ZERO modules on a build whose promoter-program tables are
        -- empty -- the promoter profile would have rendered with peaks and no
        -- module blocks, its main content missing and nothing raising to say
        -- so. Modules exist independently of any factorization; the program
        -- columns are annotation on them and come back NULL when absent.
        FROM modules m
        LEFT JOIN module_program mp USING (module_id)
        LEFT JOIN programs p ON mp.dominant_program = p.program
        WHERE m.transcript_id = ?
        ORDER BY m.center_offset
        """,
        [transcript_id],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def get_peaks_for_tss(tss_id: int, min_score: int = 0) -> pd.DataFrame:
    """Peaks at one TSS, read from the parquet sidecar.

    Peaks are not a table in the database: at the q1e-5 tier they are 73.9 M
    rows and 94% of it, which would not fit the container's 2 GiB cap. The
    file is written sorted by tss_id, so the tss_id predicate pushes down to
    row-group statistics and a lookup touches a handful of groups rather than
    scanning. Falls back to an in-database table if one exists, so a database
    built the old way still works.
    """
    fn = DATA_DIR / "peaks.parquet"
    if fn.exists():
        return get_con().execute(
            """
            SELECT t.tf, p.tf_idx, p.local_offset, p.score
            FROM read_parquet(?) p
            JOIN tf t USING (tf_idx)
            WHERE p.tss_id = ? AND p.score >= ?
            ORDER BY t.tf, p.local_offset
            """,
            [str(fn), tss_id, min_score],
        ).df()
    return get_con().execute(
        """
        SELECT t.tf, p.tf_idx, p.local_offset, p.score
        FROM peaks p
        JOIN tf t USING (tf_idx)
        WHERE p.tss_id = ? AND p.score >= ?
        ORDER BY t.tf, p.local_offset
        """,
        [tss_id, min_score],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def get_gene_config(transcript_id: str) -> Optional[dict]:
    df = get_con().execute(
        "SELECT * FROM gene_configs WHERE transcript_id = ?", [transcript_id]
    ).df()
    return df.iloc[0].to_dict() if len(df) else None


# ---------------------------------------------------------------------------
# Programs / GO / TF cluster
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_programs() -> pd.DataFrame:
    return get_con().execute(
        "SELECT * FROM programs ORDER BY program"
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_top_tfs(program: int, limit: int = 30) -> pd.DataFrame:
    return get_con().execute(
        "SELECT rank, tf, loading FROM program_tf_top "
        "WHERE program = ? ORDER BY rank LIMIT ?",
        [program, limit],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_top_tfs_enriched(program: int, limit: int = 30) -> pd.DataFrame:
    """Top TFs per program with TF cluster ids, atlas binding count, and GTEx
    top tissue + TPM joined in one table."""
    base = get_program_top_tfs(program, limit)
    if base.empty:
        return base
    con = get_con()
    tfs = base["tf"].tolist()
    placeholders = ",".join(["?"] * len(tfs))
    meta = con.execute(
        f"SELECT tf, cluster_filtered, cluster_no_filter, "
        f"       n_bound_tss_core "
        f"FROM tf WHERE tf IN ({placeholders})",
        tfs,
    ).df()
    out = base.merge(meta, on="tf", how="left")

    if gtex_available():
        try:
            tf_expr = pd.read_parquet(GTEX_DIR / "tf_tissue_expression.parquet")
            sub = tf_expr.loc[tf_expr.index.isin(tfs)]
            if not sub.empty:
                top = pd.DataFrame({
                    "tf":         sub.index,
                    "top_tissue": sub.idxmax(axis=1).values,
                    "top_tpm":    sub.max(axis=1).values.round(2),
                })
                out = out.merge(top, on="tf", how="left")
        except Exception:
            pass
    return out


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_top_go(program: int, limit: int = 15) -> pd.DataFrame:
    return get_con().execute(
        """
        SELECT rank, term, go_id, fg_in, set_size_in_bg,
               odds_ratio, p_value, q_value
        FROM program_go_top WHERE program = ?
        ORDER BY rank LIMIT ?
        """,
        [program, limit],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def list_go_terms() -> pd.DataFrame:
    """Union of GO terms appearing in either program_go_top or archetype_go_top.
    Returns: term (raw, e.g. GOBP_RIBOSOME_BIOGENESIS), go_id, n_programs,
    n_archetypes, min_q. Used to populate the GO-search autocomplete."""
    return get_con().execute(
        """
        WITH p AS (
            SELECT term, go_id, q_value, "program" AS prog, NULL AS arch
            FROM program_go_top
        ),
        a AS (
            SELECT term, go_id, q_value, NULL, archetype FROM archetype_go_top
        ),
        u AS (SELECT * FROM p UNION ALL SELECT * FROM a)
        SELECT term, ANY_VALUE(go_id) AS go_id,
               COUNT(DISTINCT prog) AS n_programs,
               COUNT(DISTINCT arch) AS n_archetypes,
               MIN(q_value) AS min_q
        FROM u
        GROUP BY term
        ORDER BY n_programs + n_archetypes DESC, min_q ASC
        """
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def programs_for_go_term(term: str) -> pd.DataFrame:
    """Programs that enrich for the given GO term, ordered by q-value."""
    return get_con().execute(
        """
        SELECT "program", rank, go_id, fg_in, set_size_in_bg,
               odds_ratio, p_value, q_value, genes_in_overlap
        FROM program_go_top WHERE term = ?
        ORDER BY q_value
        """,
        [term],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def archetypes_for_go_term(term: str) -> pd.DataFrame:
    """Archetypes that enrich for the given GO term, ordered by q-value."""
    return get_con().execute(
        """
        SELECT archetype, rank, go_id, fg_in, set_size_in_bg,
               odds_ratio, p_value, q_value, genes_in_overlap
        FROM archetype_go_top WHERE term = ?
        ORDER BY q_value
        """,
        [term],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_module_centers(program: int) -> pd.DataFrame:
    """All module centers for one program (for the position-density panel)."""
    return get_con().execute(
        """
        SELECT m.center_offset
        FROM modules m
        JOIN module_program mp USING (module_id)
        WHERE mp.dominant_program = ?
        """,
        [program],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_archetypes() -> pd.DataFrame:
    return get_con().execute(
        "SELECT * FROM archetypes ORDER BY archetype"
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_archetype_program_loadings() -> pd.DataFrame:
    return get_con().execute(
        "SELECT * FROM archetype_program_loading "
        "ORDER BY archetype, program"
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_archetype_top_go(archetype: int, limit: int = 15) -> pd.DataFrame:
    return get_con().execute(
        "SELECT rank, term, go_id, fg_in, set_size_in_bg, "
        "       odds_ratio, p_value, q_value "
        "FROM archetype_go_top WHERE archetype = ? "
        "ORDER BY rank LIMIT ?",
        [archetype, limit],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def get_gene_archetype(transcript_id: str):
    df = get_con().execute(
        "SELECT * FROM gene_archetypes WHERE transcript_id = ?",
        [transcript_id],
    ).df()
    return df.iloc[0].to_dict() if len(df) else None


@st.cache_data(ttl=3600, show_spinner=False)
def get_genes_in_archetype(archetype: int, limit: int = 200) -> pd.DataFrame:
    return get_con().execute(
        """
        SELECT ga.transcript_id, ga.gene_name, ga.n_modules,
               ga.dominant_weight, t.chrom, t.tss, t.strand
        FROM gene_archetypes ga
        JOIN tss t ON ga.transcript_id = t.transcript_id
        WHERE ga.dominant_archetype = ?
        ORDER BY ga.dominant_weight DESC
        LIMIT ?
        """,
        [archetype, limit],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_cooccurrence() -> pd.DataFrame:
    """For every (P_i, P_j), count canonical TSSs whose modules contain at
    least one module of program i AND at least one of program j (i may equal j
    when a gene has ≥2 modules in the same program).

    Returns a long-format DataFrame: prog_i, prog_j, n_genes_both,
    n_genes_either, lift (= n_both / n_expected_under_independence).
    """
    return get_con().execute(
        """
        WITH gene_progs AS (
            SELECT DISTINCT m.transcript_id, mp.dominant_program AS p
            FROM modules m JOIN module_program mp USING (module_id)
        ),
        n_genes_total AS (
            SELECT COUNT(DISTINCT transcript_id) AS n FROM gene_progs
        ),
        n_per_prog AS (
            SELECT p, COUNT(DISTINCT transcript_id) AS n
            FROM gene_progs GROUP BY p
        ),
        pair AS (
            SELECT a.p AS p_i, b.p AS p_j,
                   COUNT(DISTINCT a.transcript_id) AS n_both
            FROM gene_progs a JOIN gene_progs b USING (transcript_id)
            GROUP BY a.p, b.p
        )
        SELECT pair.p_i, pair.p_j, pair.n_both,
               n_per_prog_i.n AS n_i, n_per_prog_j.n AS n_j,
               t.n AS n_total,
               (pair.n_both * 1.0 / NULLIF(
                    (n_per_prog_i.n * n_per_prog_j.n * 1.0 / t.n), 0
               )) AS lift
        FROM pair
        JOIN n_per_prog n_per_prog_i ON pair.p_i = n_per_prog_i.p
        JOIN n_per_prog n_per_prog_j ON pair.p_j = n_per_prog_j.p
        CROSS JOIN n_genes_total t
        ORDER BY pair.p_i, pair.p_j
        """
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_cluster_members(cluster_set: str, cluster_id: int) -> list[str]:
    """All TFs in the given filtered/no_filter cluster."""
    col = "cluster_filtered" if cluster_set == "filtered" else "cluster_no_filter"
    return get_con().execute(
        f"SELECT tf FROM tf WHERE {col} = ? ORDER BY tf",
        [cluster_id],
    ).df()["tf"].tolist()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_tf_meta(tf: str) -> Optional[dict]:
    df = get_con().execute("SELECT * FROM tf WHERE tf = ?", [tf]).df()
    return df.iloc[0].to_dict() if len(df) else None


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_tf_program_loadings(tf: str) -> pd.DataFrame:
    """How much does this TF load on each of the 10 programs (top_tfs view)."""
    return get_con().execute(
        "SELECT program, rank, loading FROM program_tf_top "
        "WHERE tf = ? ORDER BY program",
        [tf],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def get_top_tss_for_tf(tf: str, limit: int = 100) -> pd.DataFrame:
    """TSSs with the most score≥500 peaks for this TF."""
    return get_con().execute(
        """
        SELECT t.gene_name, t.transcript_id, t.chrom, t.tss, t.strand,
               COUNT(*) AS n_peaks_500,
               MIN(p.local_offset) AS min_offset,
               MAX(p.local_offset) AS max_offset
        FROM peaks p
        JOIN tf  ON p.tf_idx = tf.tf_idx
        JOIN tss t ON p.tss_id = t.tss_id
        WHERE tf.tf = ? AND p.score >= 500
        GROUP BY t.gene_name, t.transcript_id, t.chrom, t.tss, t.strand
        ORDER BY n_peaks_500 DESC
        LIMIT ?
        """,
        [tf, limit],
    ).df()


# ---------------------------------------------------------------------------
# Aggregate matrices (parquet)
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# DepMap — CRISPR Chronos essentiality per (gene, lineage)
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def depmap_available() -> bool:
    return (DEPMAP_DIR / "gene_essentiality_summary.parquet").exists()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def depmap_lineage_index() -> pd.DataFrame:
    fn = DEPMAP_DIR / "lineage_index.tsv"
    if not fn.exists():
        return pd.DataFrame()
    return pd.read_csv(fn, sep="\t")


@st.cache_data(ttl=3600, show_spinner=False)
def depmap_gene_lineage(gene: str) -> pd.DataFrame:
    """Per-lineage essentiality for one gene."""
    fn = DEPMAP_DIR / "gene_lineage_essentiality.parquet"
    if not fn.exists():
        return pd.DataFrame()
    return get_con().execute(
        "SELECT lineage, n_lines, mean_chronos, median_chronos, "
        "       frac_essential "
        "FROM read_parquet(?) WHERE gene = ? ORDER BY median_chronos",
        [str(fn), gene],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def depmap_gene_summary(genes: tuple) -> pd.DataFrame:
    """Per-gene global essentiality summary for a tuple of gene names."""
    fn = DEPMAP_DIR / "gene_essentiality_summary.parquet"
    if not fn.exists() or not genes:
        return pd.DataFrame()
    placeholders = ",".join(["?"] * len(genes))
    return get_con().execute(
        f"SELECT gene, n_lines, median_chronos, frac_essential, "
        f"       most_essential_lineage, most_essential_chronos "
        f"FROM read_parquet('{fn}') WHERE gene IN ({placeholders})",
        list(genes),
    ).df()


# ---------------------------------------------------------------------------
# DepMap raw matrices — Chronos × cell line and expression × cell line
# ---------------------------------------------------------------------------
# Override with HPA_DEPMAP_RAW if the raw DepMap CSVs live elsewhere.
DEPMAP_RAW_DIR = Path(os.environ.get("HPA_DEPMAP_RAW", "data/raw/depmap"))
DEPMAP_CHRONOS_CSV    = DEPMAP_RAW_DIR / "CRISPRGeneEffect.csv"
DEPMAP_EXPRESSION_CSV = DEPMAP_RAW_DIR / "protein_coding_expr" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
DEPMAP_MODEL_CSV      = DEPMAP_RAW_DIR / "Model.csv"
DEPMAP_CORR_DIR       = DEPMAP_DIR / "tf_target_corr"
# Slim parquet versions built by data/build_depmap_pair_matrices.py — when
# present these are preferred over the raw CSVs (much faster cold-load,
# tiny disk footprint, deployable to production).
DEPMAP_CHRONOS_PARQUET = DEPMAP_DIR / "chronos_matrix.parquet"
DEPMAP_EXPR_PARQUET    = DEPMAP_DIR / "expression_matrix.parquet"
DEPMAP_META_PARQUET    = DEPMAP_DIR / "cell_line_meta.parquet"


def _safe_gene_filename(name: str) -> str:
    """Mirror the sanitization used by build_depmap_tf_target_correlations.py."""
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def _strip_entrez(col: str) -> str:
    """`A1BG (1)` -> `A1BG`. Column-name parser shared by both matrices."""
    i = col.find(" (")
    return col[:i] if i > 0 else col


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def depmap_raw_available() -> bool:
    """True if either the slim parquet matrices OR the raw CSVs are
    available. Both code paths produce the same data; parquet is preferred
    when present (smaller, faster cold-load, production-deployable)."""
    parquet_ok = (DEPMAP_CHRONOS_PARQUET.exists()
                  and DEPMAP_EXPR_PARQUET.exists()
                  and DEPMAP_META_PARQUET.exists())
    csv_ok = (DEPMAP_CHRONOS_CSV.exists()
              and DEPMAP_EXPRESSION_CSV.exists()
              and DEPMAP_MODEL_CSV.exists())
    return parquet_ok or csv_ok


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def depmap_tf_target_corr_available() -> bool:
    """True if either the precomputed shards OR the per-cell-line matrices
    are present."""
    return DEPMAP_CORR_DIR.is_dir() or depmap_raw_available()


@st.cache_resource(show_spinner="Loading DepMap Chronos matrix…")
def depmap_chronos_matrix() -> pd.DataFrame:
    """Chronos essentiality: cell lines (ModelID) × genes (symbol).
    Prefers the slim parquet when present (built by
    data/build_depmap_pair_matrices.py); falls back to the raw CSV."""
    if DEPMAP_CHRONOS_PARQUET.exists():
        df = pd.read_parquet(DEPMAP_CHRONOS_PARQUET)
        return df.set_index("ModelID")
    if DEPMAP_CHRONOS_CSV.exists():
        df = pd.read_csv(DEPMAP_CHRONOS_CSV, index_col=0)
        df.columns = [_strip_entrez(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index.name = "ModelID"
        return df
    return pd.DataFrame()


@st.cache_resource(show_spinner="Loading DepMap expression matrix…")
def depmap_expression_matrix() -> pd.DataFrame:
    """log10(TPM+1) expression: cell lines (ModelID) × genes (symbol).
    Prefers the slim parquet when present."""
    if DEPMAP_EXPR_PARQUET.exists():
        df = pd.read_parquet(DEPMAP_EXPR_PARQUET)
        return df.set_index("ModelID")
    if DEPMAP_EXPRESSION_CSV.exists():
        df = pd.read_csv(DEPMAP_EXPRESSION_CSV, index_col=0)
        df.columns = [_strip_entrez(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()]
        df.index.name = "ModelID"
        return df
    return pd.DataFrame()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def depmap_cell_line_metadata() -> pd.DataFrame:
    """ModelID → CellLineName, OncotreeLineage. Prefers parquet when present."""
    if DEPMAP_META_PARQUET.exists():
        return pd.read_parquet(DEPMAP_META_PARQUET).set_index("ModelID")
    if DEPMAP_MODEL_CSV.exists():
        return (pd.read_csv(DEPMAP_MODEL_CSV,
                             usecols=["ModelID", "CellLineName", "OncotreeLineage"])
                  .set_index("ModelID"))
    return pd.DataFrame()


@st.cache_data(ttl=3600, show_spinner=False)
def depmap_tf_target_correlation(
    target: str, tfs: tuple[str, ...],
    min_cell_lines: int = 50,
) -> pd.DataFrame:
    """For a target gene and a list of TFs, compute Pearson r between each
    TF's Chronos essentiality and the target's expression across the
    cell lines common to both matrices.

    Returns DataFrame [tf, r, n_cell_lines] sorted by |r| descending. TFs
    with fewer than `min_cell_lines` non-NaN paired observations are dropped.
    Returns empty DataFrame if matrices are missing or the target is absent
    from expression.

    Fast path: if data/depmap/tf_target_corr/{target}.parquet exists,
    read the slim per-target shard (~5 KB) instead of the full CSVs.
    """
    if not tfs:
        return pd.DataFrame(columns=["tf", "r", "n_cell_lines"])

    # --- Fast path: precomputed per-target shard --------------------------
    shard = DEPMAP_CORR_DIR / f"{_safe_gene_filename(target)}.parquet"
    if shard.exists():
        df = pd.read_parquet(shard)
        # Schema: tf (categorical), r (int8 = r*100), n (int16).
        df = df[df["tf"].astype(str).isin(set(tfs))]
        df = df[df["n"] >= int(min_cell_lines)]
        if df.empty:
            return pd.DataFrame(columns=["tf", "r", "n_cell_lines"])
        out = pd.DataFrame({
            "tf":           df["tf"].astype(str).values,
            "r":            (df["r"].astype(np.int16).values / 100.0)
                                .round(2),
            "n_cell_lines": df["n"].astype(int).values,
        })
        out = out.reindex(out["r"].abs().sort_values(ascending=False).index)
        return out.reset_index(drop=True)

    # --- Fallback: runtime computation from raw CSVs ----------------------
    chro = depmap_chronos_matrix()
    expr = depmap_expression_matrix()
    if chro.empty or expr.empty or target not in expr.columns:
        return pd.DataFrame(columns=["tf", "r", "n_cell_lines"])

    tfs_in = [t for t in tfs if t in chro.columns]
    if not tfs_in:
        return pd.DataFrame(columns=["tf", "r", "n_cell_lines"])

    common = chro.index.intersection(expr.index)
    if len(common) < min_cell_lines:
        return pd.DataFrame(columns=["tf", "r", "n_cell_lines"])

    e = expr.loc[common, target].to_numpy(dtype=np.float64)
    C = chro.loc[common, tfs_in].to_numpy(dtype=np.float64)

    # Pairwise pearson with row-wise NaN masking. Vectorized per TF.
    rows = []
    for j, tf in enumerate(tfs_in):
        c = C[:, j]
        mask = np.isfinite(c) & np.isfinite(e)
        n = int(mask.sum())
        if n < min_cell_lines:
            continue
        ce = c[mask]; ee = e[mask]
        ce -= ce.mean(); ee -= ee.mean()
        denom = np.sqrt((ce * ce).sum() * (ee * ee).sum())
        r = float((ce * ee).sum() / denom) if denom > 0 else float("nan")
        rows.append({"tf": tf, "r": round(r, 3), "n_cell_lines": n})

    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.reindex(out["r"].abs().sort_values(ascending=False).index)
        out = out.reset_index(drop=True)
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def depmap_tf_target_pair_values(target: str, tf: str) -> pd.DataFrame:
    """Per-cell-line (tf_chronos, target_expr, lineage, cell_line_name) for
    one (TF, target) pair. Used by the scatter plot. Empty if either is
    missing from its matrix."""
    chro = depmap_chronos_matrix()
    expr = depmap_expression_matrix()
    if (chro.empty or expr.empty
            or tf not in chro.columns or target not in expr.columns):
        return pd.DataFrame()
    common = chro.index.intersection(expr.index)
    df = pd.DataFrame({
        "tf_chronos":  chro.loc[common, tf].astype(float),
        "target_expr": expr.loc[common, target].astype(float),
    }, index=common)
    df = df.dropna()
    meta = depmap_cell_line_metadata()
    if not meta.empty:
        df = df.join(meta, how="left")
    df = df.rename(columns={"OncotreeLineage": "lineage",
                              "CellLineName":  "cell_line"})
    return df.reset_index()


# ---------------------------------------------------------------------------
# GTEx — loaded as parquet on demand and cached
# ---------------------------------------------------------------------------
@st.cache_data(ttl=24 * 3600, show_spinner=False)
def gtex_available() -> bool:
    return (GTEX_DIR / "transcript_tissue_mean.parquet").exists()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def gtex_tissue_index() -> pd.DataFrame:
    fn = GTEX_DIR / "tissue_index.tsv"
    if not fn.exists():
        return pd.DataFrame()
    return pd.read_csv(fn, sep="\t")


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_transcript_stats(transcript_id: str) -> pd.DataFrame:
    """Per-tissue stats for one transcript."""
    fn = GTEX_DIR / "transcript_tissue_stats.parquet"
    if not fn.exists():
        return pd.DataFrame()
    return get_con().execute(
        "SELECT tissue, n_samples, mean, median, q1, q3, std "
        "FROM read_parquet(?) WHERE transcript_id = ? ORDER BY mean DESC",
        [str(fn), transcript_id],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_tf_expression(tf: str) -> pd.DataFrame:
    """Per-tissue mean TPM (max-of-transcripts) for one TF gene."""
    fn = GTEX_DIR / "tf_tissue_expression.parquet"
    if not fn.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fn)
    if tf not in df.index:
        return pd.DataFrame()
    s = df.loc[tf].sort_values(ascending=False)
    return s.reset_index().rename(columns={"index": "tissue", tf: "tpm"})


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_module_activity_for_transcript(transcript_id: str) -> pd.DataFrame:
    """Module × tissue activity scores for the given transcript's modules."""
    fn = GTEX_DIR / "module_tissue_activity.parquet"
    if not fn.exists():
        return pd.DataFrame()
    return get_con().execute(
        """
        SELECT mta.module_id, m.center_offset, mta.tissue,
               mta.mean_tf_tpm, mta.n_tfs_in_module
        FROM read_parquet(?) mta
        JOIN modules m ON mta.module_id = m.module_id
        WHERE m.transcript_id = ?
        """,
        [str(fn), transcript_id],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_module_target_evidence(transcript_id: str) -> pd.DataFrame:
    """For all modules at this transcript: r between module-mean TF TPM and
    target across 66 tissues, plus the top-r 'driver' TF and a count of TFs
    in the module with |r|>=0.5, plus n_supporting_tissues."""
    fn = GTEX_DIR / "module_target_correlation.parquet"
    if not fn.exists():
        return pd.DataFrame()
    # module_program exists only when the promoter build carries its own
    # factorization. Under the hybrid architecture it does not -- programs come
    # from the genome build, and modules have no promoter-program assignment.
    # The program column was decoration here; the module/target evidence is the
    # point, so it degrades instead of failing the whole panel.
    prog_sel, prog_join = "", ""
    if _table_exists("module_program"):
        prog_sel = "mp.dominant_program,"
        prog_join = "JOIN module_program mp ON m.module_id = mp.module_id"
    return get_con().execute(
        f"""
        SELECT m.module_id, m.center_offset, m.width,
               {prog_sel}
               mt.r_module_target, mt.top_driver_tf, mt.top_driver_r,
               mt.n_tfs_high_r, mt.n_supporting_tissues,
               mt.top_supporting_tissue
        FROM read_parquet(?) mt
        JOIN modules m ON mt.module_id = m.module_id
        {prog_join}
        WHERE m.transcript_id = ?
        ORDER BY m.center_offset
        """,
        [str(fn), transcript_id],
    ).df()


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_module_tf_evidence(module_id: int) -> pd.DataFrame:
    """Per-TF correlation against the focal target, for the TFs in this
    module."""
    fn = GTEX_DIR / "module_tf_target_correlation.parquet"
    if not fn.exists():
        return pd.DataFrame()
    df = get_con().execute(
        "SELECT tf, r FROM read_parquet(?) WHERE module_id = ? "
        "ORDER BY ABS(r) DESC",
        [str(fn), int(module_id)],
    ).df()
    df["r"] = df["r"].astype(float).round(2)
    return df


DRIVER_CLASS_ORDER = ["no-driver", "single-driver", "multi-driver"]


def driver_class_label(n) -> str:
    """Categorize a module by its `n_tfs_high_r` count.

    no-driver:     n == 0   (no TF in the module correlates with the target)
    single-driver: n == 1   (one TF drives the regulatory signal)
    multi-driver:  n >= 2   (combinatorial / redundant)
    Returns "" for NaN/None (e.g. when GTEx evidence is missing).
    """
    if n is None:
        return ""
    if isinstance(n, float) and not np.isfinite(n):
        return ""
    n = int(n)
    if n == 0:
        return "no-driver"
    if n == 1:
        return "single-driver"
    return "multi-driver"


def driver_class_series(s: pd.Series) -> pd.Series:
    """Vectorized driver_class_label over a Series."""
    return s.map(driver_class_label).astype("string")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_representative_modules(
    program: int, n: int = 6,
) -> pd.DataFrame:
    """Top-N modules of a program ranked by dominant_weight, with their
    parent transcript info + the module's top-5 TFs by |r| against target.

    Returns DataFrame [module_id, transcript_id, gene_name, dominant_weight,
    center_offset, width, top_tfs]. `top_tfs` is a comma-joined string of
    up to 5 TFs, ordered by |r| descending — the canonical 'fingerprint'.
    """
    mt_fn = GTEX_DIR / "module_tf_target_correlation.parquet"
    if not mt_fn.exists():
        return get_con().execute(
            """
            SELECT mp.module_id, m.transcript_id, t.gene_name,
                   mp.dominant_weight, m.center_offset, m.width
            FROM module_program mp
            JOIN modules m ON mp.module_id = m.module_id
            JOIN tss     t ON m.transcript_id = t.transcript_id
            WHERE mp.dominant_program = ?
            ORDER BY mp.dominant_weight DESC
            LIMIT ?
            """, [int(program), int(n)]
        ).df()
    return get_con().execute(
        f"""
        WITH top_modules AS (
            SELECT mp.module_id, mp.dominant_weight,
                   m.transcript_id, m.center_offset, m.width,
                   t.gene_name
            FROM module_program mp
            JOIN modules m ON mp.module_id = m.module_id
            JOIN tss     t ON m.transcript_id = t.transcript_id
            WHERE mp.dominant_program = ?
            ORDER BY mp.dominant_weight DESC
            LIMIT ?
        ),
        ranked AS (
            SELECT mt.module_id, mt.tf, mt.r,
                   ROW_NUMBER() OVER (PARTITION BY mt.module_id
                                       ORDER BY ABS(mt.r) DESC) AS rn
            FROM read_parquet('{mt_fn}') mt
            WHERE mt.module_id IN (SELECT module_id FROM top_modules)
        ),
        tfs AS (
            SELECT module_id,
                   STRING_AGG(tf, ', ' ORDER BY ABS(r) DESC) AS top_tfs
            FROM ranked WHERE rn <= 5
            GROUP BY module_id
        )
        SELECT tm.module_id, tm.transcript_id, tm.gene_name,
               tm.dominant_weight, tm.center_offset, tm.width,
               COALESCE(tfs.top_tfs, '') AS top_tfs
        FROM top_modules tm
        LEFT JOIN tfs USING (module_id)
        ORDER BY tm.dominant_weight DESC
        """, [int(program), int(n)]
    ).df()


# ---------------------------------------------------------------------------
# TF × TF co-occurrence — precomputed atlas-wide pair table
# ---------------------------------------------------------------------------
TF_PAIR_PARQUET = DATA_DIR / "network" / "tf_pair_cooccurrence.parquet"


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def tf_pair_table_available() -> bool:
    return TF_PAIR_PARQUET.exists()


@st.cache_resource(show_spinner="Loading TF pair table…")
def _tf_pair_table_full() -> pd.DataFrame:
    """All ~330k atlas-wide TF pairs (lower-triangular, n_shared ≥ 5)."""
    if not TF_PAIR_PARQUET.exists():
        return pd.DataFrame()
    df = pd.read_parquet(TF_PAIR_PARQUET)
    # Cast categoricals back to plain strings for downstream filtering.
    df["tf_a"] = df["tf_a"].astype(str)
    df["tf_b"] = df["tf_b"].astype(str)
    return df


@st.cache_data(ttl=3600, show_spinner=False)
def tf_pair_query(
    search: str = "", min_shared: int = 100, min_jaccard: float = 0.0,
    sort_by: str = "n_shared", ascending: bool = False,
    limit: int = 200,
) -> pd.DataFrame:
    """Filtered + sorted slice of the TF-pair table.

    search:      case-insensitive substring; matches either TF in the pair.
                 Empty = no filter.
    min_shared:  minimum n_shared modules to keep.
    min_jaccard: minimum Jaccard overlap to keep.
    sort_by:     'n_shared', 'jaccard', or 'lift'.
    """
    df = _tf_pair_table_full()
    if df.empty:
        return df

    if min_shared > 0:
        df = df[df["n_shared"] >= int(min_shared)]
    if min_jaccard > 0:
        df = df[df["jaccard"] >= float(min_jaccard)]

    if search:
        s = search.strip().upper()
        mask = (df["tf_a"].str.upper().str.contains(s, regex=False) |
                df["tf_b"].str.upper().str.contains(s, regex=False))
        df = df[mask]

    if sort_by not in df.columns:
        sort_by = "n_shared"
    df = df.sort_values(sort_by, ascending=bool(ascending))
    return df.head(int(limit)).reset_index(drop=True)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def tf_pair_table_stats(lift_min_shared: int = 1000) -> dict:
    """Summary stats for the intro / table-context line.

    `lift_min_shared` is the n_shared floor used when picking the max
    lift. The raw table includes pairs where both TFs are individually
    rare; those pairs can have astronomical lift (observed/expected
    blows up when expected is tiny) without being biologically
    interesting. Matching the floor to the default table filter (1 000)
    keeps the header metric aligned with what the user actually sees.
    """
    df = _tf_pair_table_full()
    if df.empty:
        return {}
    lift_pool = df[df["n_shared"] >= int(lift_min_shared)]
    if lift_pool.empty:
        lift_pool = df
    return {
        "n_pairs":        int(len(df)),
        "n_tfs":          int(len(set(df["tf_a"]) | set(df["tf_b"]))),
        "max_share":      int(df["n_shared"].max()),
        "max_lift":       float(lift_pool["lift"].max()),
        "lift_min_shared": int(lift_min_shared),
    }


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_tf_cobinding_partners(focal_tf: str, limit: int = 20) -> pd.DataFrame:
    """For a focal TF, top partner TFs ranked by number of shared modules.

    Uses module_tf_target_correlation.parquet as the comprehensive
    (module_id, tf) listing (~2 M rows, 27 TFs per module avg).

    Returns DataFrame [partner, n_shared, partner_total, focal_total,
    pct_of_partner_modules, pct_of_focal_modules, jaccard]. The two
    directional percentages tell different stories: pct_of_partner answers
    'how dependent is partner on co-binding focal?' (high values name
    obligate partners like CTCF↔RAD21); pct_of_focal answers 'how broad
    is partner's reach across focal's modules?'.
    """
    fn = GTEX_DIR / "module_tf_target_correlation.parquet"
    if not fn.exists():
        return pd.DataFrame()
    return get_con().execute(
        f"""
        WITH focal_modules AS (
            SELECT DISTINCT module_id FROM read_parquet(?)
            WHERE tf = ?
        ),
        partner_counts AS (
            SELECT mt.tf AS partner,
                   COUNT(*) AS n_shared
            FROM read_parquet(?) mt
            JOIN focal_modules fm ON mt.module_id = fm.module_id
            WHERE mt.tf <> ?
            GROUP BY mt.tf
        ),
        partner_total AS (
            SELECT tf, COUNT(*) AS n_total
            FROM read_parquet(?) GROUP BY tf
        ),
        n_focal AS (SELECT COUNT(*) AS n FROM focal_modules)
        SELECT pc.partner,
               pc.n_shared,
               pt.n_total                            AS partner_total,
               (SELECT n FROM n_focal)               AS focal_total,
               ROUND(100.0 * pc.n_shared / pt.n_total, 1)
                   AS pct_of_partner_modules,
               ROUND(100.0 * pc.n_shared / (SELECT n FROM n_focal), 1)
                   AS pct_of_focal_modules,
               ROUND(1.0 * pc.n_shared /
                     (pt.n_total + (SELECT n FROM n_focal) - pc.n_shared),
                     3) AS jaccard
        FROM partner_counts pc
        JOIN partner_total pt ON pc.partner = pt.tf
        ORDER BY pc.n_shared DESC
        LIMIT ?
        """,
        [str(fn), focal_tf, str(fn), focal_tf, str(fn), int(limit)],
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_strand_distribution() -> pd.DataFrame:
    """Per-program strand distribution of parent transcripts.

    Returns DataFrame [program, n_plus, n_minus, total, frac_plus]. Used by
    the strand-asymmetry verification panel — expected ~50/50 within each
    program; any large skew would suggest a chromatin / orientation
    artifact in the upstream pipeline.
    """
    df = get_con().execute(
        """
        SELECT mp.dominant_program AS program,
               t.strand            AS strand,
               COUNT(*)            AS n
        FROM module_program mp
        JOIN modules m ON mp.module_id = m.module_id
        JOIN tss     t ON m.transcript_id = t.transcript_id
        WHERE mp.dominant_program IS NOT NULL
        GROUP BY mp.dominant_program, t.strand
        """
    ).df()
    if df.empty:
        return pd.DataFrame(columns=["program", "n_plus", "n_minus",
                                       "total", "frac_plus"])
    piv = (df.pivot(index="program", columns="strand", values="n")
             .fillna(0).astype(int))
    n_plus  = piv["+"] if "+" in piv.columns else 0
    n_minus = piv["-"] if "-" in piv.columns else 0
    out = pd.DataFrame({
        "program":  piv.index.astype(int),
        "n_plus":   n_plus,
        "n_minus":  n_minus,
    }).reset_index(drop=True)
    out["total"] = out["n_plus"] + out["n_minus"]
    out["frac_plus"] = (out["n_plus"] / out["total"]).round(3)
    return out.sort_values("program").reset_index(drop=True)


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def gtex_program_driver_class_distribution() -> pd.DataFrame:
    """Per-program count of modules in each driver class.

    Returns long DataFrame [program, driver_class, n_modules].
    Modules missing from the correlation parquet (e.g. excluded for
    insufficient tissue support) are skipped.
    """
    fn = GTEX_DIR / "module_target_correlation.parquet"
    if not fn.exists():
        return pd.DataFrame()
    df = get_con().execute(
        """
        SELECT mp.dominant_program AS program,
               mt.n_tfs_high_r AS n_tfs_high_r
        FROM read_parquet(?) mt
        JOIN module_program mp ON mt.module_id = mp.module_id
        WHERE mp.dominant_program IS NOT NULL
          AND mt.n_tfs_high_r IS NOT NULL
        """,
        [str(fn)],
    ).df()
    if df.empty:
        return df
    df["driver_class"] = driver_class_series(df["n_tfs_high_r"])
    out = (df.groupby(["program", "driver_class"])
             .size().rename("n_modules").reset_index())
    return out


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_program_tf_tissue_matrix(
    program: int, limit: int = 20,
) -> tuple[pd.DataFrame, pd.Series]:
    """Top-N TFs of a program × 66 GTEx tissues.

    Returns (M, loadings):
      M:        wide DataFrame, index=TF (loading desc), columns=tissue, values=TPM.
      loadings: Series of H loadings indexed by TF, aligned to M.index.
    Returns ({}, {}) if GTEx data missing or program has no TFs in the matrix.
    """
    fn = GTEX_DIR / "tf_tissue_expression.parquet"
    if not fn.exists():
        return pd.DataFrame(), pd.Series(dtype="float64")
    top = get_program_top_tfs(int(program), limit=int(limit))
    if top.empty:
        return pd.DataFrame(), pd.Series(dtype="float64")
    tf_expr = pd.read_parquet(fn)
    keep = top[top["tf"].isin(tf_expr.index)].copy()
    if keep.empty:
        return pd.DataFrame(), pd.Series(dtype="float64")
    M = tf_expr.loc[keep["tf"].tolist()]
    loadings = keep.set_index("tf")["loading"].astype(float)
    return M, loadings


@st.cache_data(ttl=24 * 3600, show_spinner=False)
# ORPHANED by the move to one program vocabulary. These read outputs of
# gtex_regulatory_evidence.001.py, which keys everything on the PROMOTER
# factorization's program numbers -- a numbering the site no longer uses. They
# have no callers. Left in place rather than deleted because re-deriving
# tissue specificity for the genome programs is a real and wanted analysis
# (per program, the tissue specificity of the genes near its elements), and
# these are the shape it should land in.
def gtex_program_tissue_specificity() -> pd.DataFrame:
    fn = GTEX_DIR / "program_tissue_specificity.tsv"
    if not fn.exists():
        return pd.DataFrame()
    return pd.read_csv(fn, sep="\t")


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def gtex_archetype_tissue_specificity(min_samples: int = 40) -> pd.DataFrame:
    """Yanai tau per archetype over the mean GTEx expression of each
    archetype's member transcripts. Tissues with fewer than `min_samples`
    average donor samples are excluded — matches MIN_SAMPLES_FOR_TAU in the
    program-tau upstream computation.

    Returns: archetype, tau, top_tissue, top_tissue_mean_tpm, median_tpm,
    n_genes, n_tissues_used, min_samples_for_tau.
    """
    ttm_fn = GTEX_DIR / "transcript_tissue_mean.parquet"
    ti_fn  = GTEX_DIR / "tissue_index.tsv"
    if not ttm_fn.exists():
        return pd.DataFrame()

    arch = get_con().execute(
        "SELECT transcript_id, dominant_archetype FROM gene_archetypes "
        "WHERE dominant_archetype IS NOT NULL"
    ).df()

    ttm = pd.read_parquet(ttm_fn)
    tissues = list(ttm.columns)

    well = set(tissues)
    if ti_fn.exists():
        ti = pd.read_csv(ti_fn, sep="\t")
        well = set(ti.loc[ti["mean_samples"] >= min_samples, "tissue"])
    keep_cols = [t for t in tissues if t in well]
    if not keep_cols:
        return pd.DataFrame()

    df = arch.set_index("transcript_id").join(ttm, how="inner")
    mean_per_arch = df.groupby("dominant_archetype")[tissues].mean()
    n_per_arch = df.groupby("dominant_archetype").size()

    rows = []
    for arch_id, row in mean_per_arch.iterrows():
        v_keep = row[keep_cols].to_numpy(np.float64)
        v_keep_f = v_keep[np.isfinite(v_keep)]
        if v_keep_f.size == 0 or v_keep_f.max() <= 0:
            tau = float("nan"); top = ""; top_v = float("nan")
        else:
            n = v_keep_f.size
            tau = float((1.0 - v_keep_f / v_keep_f.max()).sum() / (n - 1))
            ix = int(np.nanargmax(v_keep))
            top = keep_cols[ix]
            top_v = float(v_keep[ix])
        rows.append({
            "archetype":          int(arch_id),
            "tau":                round(tau, 3) if np.isfinite(tau) else float("nan"),
            "top_tissue":         top,
            "top_tissue_mean_tpm":round(top_v, 2) if np.isfinite(top_v) else float("nan"),
            "median_tpm":         round(float(np.nanmedian(v_keep)), 2),
            "n_genes":            int(n_per_arch.loc[arch_id]),
            "n_tissues_used":     len(keep_cols),
            "min_samples_for_tau":int(min_samples),
        })
    return pd.DataFrame(rows).sort_values("archetype").reset_index(drop=True)


@st.cache_data(ttl=3600, show_spinner=False)
def gtex_tf_target_correlations(transcript_id: str,
                                  min_abs_r: float = 0.3) -> pd.DataFrame:
    """All TFs with |r| >= min_abs_r against this transcript's expression
    profile across GTEx tissues."""
    fn = GTEX_DIR / "tf_target_correlation.parquet"
    if not fn.exists():
        return pd.DataFrame()
    df = get_con().execute(
        "SELECT tf, r FROM read_parquet(?) "
        "WHERE target_transcript = ? AND ABS(r) >= ? "
        "ORDER BY r DESC",
        [str(fn), transcript_id, float(min_abs_r)],
    ).df()
    df["r"] = df["r"].astype(float).round(2)
    return df


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def load_aggregate_matrix(flavor: str = "binary") -> pd.DataFrame:
    """Load TF×position matrix from parquet. flavor in
    {binary, score, raw, raw_score1000}."""
    fn = AGG_DIR / f"tf_x_position.{flavor}.parquet"
    if not fn.exists():
        return pd.DataFrame()
    df = pd.read_parquet(fn)
    df = df.set_index("TF")
    df.columns = df.columns.astype(int)
    return df


# ---------------------------------------------------------------------------
# Genome-wide element layer
# ---------------------------------------------------------------------------
# These read the annotation-free build (elements, genome_programs,
# program_families) that sits behind the gene-centric front door. The promoter
# tables above are unchanged: the regression gate showed both layers see the
# same promoters (98.2% recovery at 12 bp median offset), so a gene page can
# draw on either without them contradicting each other.


def _table_exists(name: str) -> bool:
    try:
        get_con().execute(f"SELECT 1 FROM {name} LIMIT 1")
        return True
    except Exception:
        return False


def has_genome_layer() -> bool:
    """False on a database built before the genome layer, so tabs can degrade
    rather than raise. Not cached: it gates the cached readers below."""
    try:
        get_con().execute("SELECT 1 FROM elements LIMIT 1")
        return True
    except Exception:
        return False


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_families() -> pd.DataFrame:
    return get_con().execute(
        "SELECT * FROM program_families ORDER BY n_elements DESC"
    ).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_genome_programs(family: int | None = None) -> pd.DataFrame:
    if family is None:
        return get_con().execute(
            "SELECT * FROM genome_programs ORDER BY program").df()
    return get_con().execute(
        "SELECT * FROM genome_programs WHERE family = ? ORDER BY n_elements DESC",
        [family]).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_elements_for_gene(gene_name: str) -> pd.DataFrame:
    """Every element whose NEAREST gene is this one, with program and family.

    n_tss_comparably_close travels with the row deliberately: 56.6% of distal
    elements have a rival TSS within twice the distance, so a caller that drops
    it turns a locator into a regulatory assignment the data cannot support.
    """
    return get_con().execute(
        """SELECT e.element_id, e.chrom, e.start, e."end", e.peak, e.width,
                  e.dist_to_tss, e.stratum, e.n_tfs_assigned,
                  e.n_tss_comparably_close, e.cluster_id, e.cluster_size,
                  p.dominant_program AS program, p.dominant_weight AS weight,
                  g.family, g.substantive, g.seed_stability, g.top_tfs
           FROM elements e
           JOIN element_program p USING (element_id)
           JOIN genome_programs g ON g.program = p.dominant_program
           WHERE e.nearest_gene_name = ?
           ORDER BY ABS(e.dist_to_tss)""", [gene_name]).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_family_labels() -> dict:
    """{family: 'TF1 / TF2 / TF3'} for legends and axis labels."""
    df = get_con().execute(
        "SELECT family, top_tfs FROM program_families").df()
    return {int(r.family): " / ".join(str(r.top_tfs).split(", ")[:3])
            for _, r in df.iterrows()}


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_genes_in_family(family: int, limit: int = 200) -> pd.DataFrame:
    """Genes with the most elements in a family.

    PROMOTER-stratum elements only, ranked by count then by total loading.

    Two ranking traps, both measured rather than anticipated. Counting ALL
    strata returns segmental duplications and repeat clusters -- for the PRC2
    family it gave SRGAP2C (217 elements), TUBA3C, TPTE, ZNF716, 95-100% of
    them distal -- because repetitive regions accumulate spurious peaks. And
    counting promoter elements alone is a near-total tie, since most genes have
    exactly one, so the gene name silently becomes the sort key and the list
    runs alphabetically from the third entry.

    Restricting to the promoter stratum also sidesteps the ambiguity that makes
    distal attribution unsafe (56.6% of distal elements have a rival TSS within
    twice the distance). n_distal is still returned, for display, not ranking.

    This is a lookup, not an assignment: diag_gene_coherence.py found gene
    identity carries no information about an element's program once genomic
    distance is controlled.
    """
    return get_con().execute(
        """SELECT e.nearest_gene_name AS gene_name,
                  COUNT(*) FILTER (WHERE e.stratum = 'promoter') AS n_promoter,
                  COUNT(*) FILTER (WHERE e.stratum = 'distal') AS n_distal,
                  COUNT(*) AS n_elements,
                  SUM(CASE WHEN e.stratum = 'promoter'
                           THEN p.dominant_weight ELSE 0 END) AS promoter_weight
           FROM elements e
           JOIN element_program p USING (element_id)
           JOIN genome_programs g ON g.program = p.dominant_program
           WHERE g.family = ? AND e.nearest_gene_name IS NOT NULL
           GROUP BY 1
           HAVING COUNT(*) FILTER (WHERE e.stratum = 'promoter') > 0
           ORDER BY n_promoter DESC, promoter_weight DESC
           LIMIT ?""",
        [family, limit]).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_family_terms(family: int, limit: int = 25,
                     lib: str | None = None) -> pd.DataFrame:
    """All significant enrichments for a family, best evidence first.

    get_program_families() carries one headline label; this is everything that
    cleared FDR. Worth showing, because the headline hides structure: family 8
    is labelled "ESC/E(Z) complex" but its strongest hit is "PcG protein
    complex" driven by CBX7, EZH2, JARID2, KDM2B, MTF2, PCGF1 and SUZ12 -- PRC1
    and PRC2 members together, which the single label conceals.

    `is_label` marks the row the headline came from. It is usually NOT row 1:
    the list ranks by q, the label is chosen by tier-then-odds (median rank 4).
    A UI that shows the headline above this list should highlight that row, or
    the two will look inconsistent.

    `overlap_tfs` is the evidence -- which member TFs are in the term -- so a
    reader can judge a 3/7 hit rather than trust it.
    """
    sql = ("SELECT rank, is_label, label, lib, go_id, url, overlap, set_size, "
           "odds, q, overlap_tfs FROM family_terms WHERE family = ?")
    params: list = [family]
    if lib:
        sql += " AND lib = ?"
        params.append(lib)
    sql += " ORDER BY rank LIMIT ?"
    params.append(limit)
    return get_con().execute(sql, params).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_genome_program_tfs(program: int, limit: int = 30) -> pd.DataFrame:
    return get_con().execute(
        "SELECT rank, tf, loading FROM genome_program_tf_top "
        "WHERE program = ? ORDER BY rank LIMIT ?", [program, limit]).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_element_stats(program: int) -> pd.DataFrame:
    """Stratum breakdown and distance spread for one program's elements.

    Distances are |dist_to_tss| quantiles rather than a mean: the distribution
    runs from a few hundred bp to several hundred kb and is heavily skewed, so
    a mean would describe no actual element.
    """
    return get_con().execute(
        """SELECT e.stratum,
                  COUNT(*)                          AS n,
                  MEDIAN(ABS(e.dist_to_tss))        AS median_dist,
                  QUANTILE_CONT(ABS(e.dist_to_tss), 0.9) AS p90_dist,
                  MEDIAN(e.n_tfs_assigned)          AS median_tfs,
                  MEDIAN(e.width)                   AS median_width
           FROM elements e JOIN element_program p USING (element_id)
           WHERE p.dominant_program = ?
           GROUP BY 1 ORDER BY n DESC""", [program]).df()


@st.cache_data(ttl=24 * 3600, show_spinner=False)
def get_program_distance_hist(program: int, bins: int = 40) -> pd.DataFrame:
    """Signed log-scaled distance histogram, for the position panel.

    Signed, because upstream and downstream are not interchangeable, and
    log-scaled because a linear axis over +/-500 kb puts every promoter
    element in one bar.
    """
    return get_con().execute(
        """SELECT CAST(SIGN(e.dist_to_tss) *
                       LEAST(6, LOG10(GREATEST(ABS(e.dist_to_tss), 1)))
                       * ? / 12 AS INTEGER) AS bin,
                  COUNT(*) AS n
           FROM elements e JOIN element_program p USING (element_id)
           WHERE p.dominant_program = ?
           GROUP BY 1 ORDER BY 1""", [bins, program]).df()
