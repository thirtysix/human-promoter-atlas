#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Pack the canonical-promoter analysis outputs into a single DuckDB file plus a
small aggregate-matrix parquet directory, used by the Streamlit explorer app.

Reads from:
    {ANALYSIS_DN}/tss_modules/                  modules + NMF
    {ANALYSIS_DN}/matrices/                     aggregate TF×position matrices
    {ANALYSIS_DN}/clustering/                   filtered K=8 TF clusters
    {ANALYSIS_DN}/clustering_no_filter/         no-filter K=12 TF clusters
    {ANALYSIS_DN}/enrichment_msigdb_gobp_modules/k10/   per-program GO BP

Writes:
    {APP_DN}/data/canonical_promoter.duckdb
    {APP_DN}/data/aggregate/{tf_x_position.binary,score,raw,raw_score1000}.parquet
    {APP_DN}/data/manifest.json   (versions + parameters for the Methods tab)

Tables in DuckDB:
    tss               canonical TSS metadata          (~19,745 rows)
    tf                TF metadata + cluster ids       (~1,304 rows)
    peaks             per-peak records                (~12 M rows)  [hot table]
    modules           per-gene regulatory modules     (~77 K rows)
    module_program    module → program (k=10)         (~77 K rows)
    programs          program-level summary (k=10)    (10 rows)
    program_tf_top    top-30 TFs per program          (300 rows)
    program_go_top    top-15 GO BP terms per program  (150 rows)
    tf_clusters       TF cluster assignments          (~1,400 rows)
    gene_configs      per-gene program_path summary   (~18 K rows)

Indexes are created on the JOIN-hot columns (tss_id, tf_idx, gene_name, tf,
transcript_id, dominant_program). DuckDB's planner is good enough that we
don't need complex covering indexes.
"""

################################################################################
# Libraries ####################################################################
################################################################################
import json
import os
import shutil
from datetime import datetime
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd


################################################################################
# Paths ########################################################################
################################################################################
APP_DN      = Path(__file__).resolve().parents[1]


def _dotenv(path: Path) -> dict:
    """Minimal KEY=VALUE reader for the repo-root .env.

    Every pipeline script reads .env through config.py, but this one used plain
    os.environ, so HPA_ANALYSIS_DIR set in .env was ignored and the default
    silently applied -- pointing the build at data/raw/analyses instead of the
    real analysis tree. Not imported from config because that module hard-
    requires the raw-input paths (GTF, ChIP-Atlas, DNA-binding table) that this
    packing step never touches.
    """
    vals = {}
    if not path.is_file():
        return vals
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        vals[key.strip()] = val.strip().strip('"').strip("'")
    return vals


_FILE_ENV = _dotenv(APP_DN / ".env")


def _env(name: str, default: str) -> str:
    """Real environment wins, then .env, then the default."""
    return os.environ.get(name) or _FILE_ENV.get(name) or default


# Override with HPA_ANALYSIS_DIR if the upstream analysis lives elsewhere.
ANALYSIS_DN = Path(_env("HPA_ANALYSIS_DIR", "data/raw/analyses"))
DATA_DN     = APP_DN / "data"
DUCKDB_FN   = DATA_DN / "canonical_promoter.duckdb"
AGG_DN      = DATA_DN / "aggregate"
MANIFEST_FN = DATA_DN / "manifest.json"

# Must match the rank the build was actually factorized at. Kept as an env
# read rather than a literal because raising the rank otherwise packages
# nmf.k10.* tables into an app whose build has none -- or, worse, stale ones.
# Mirrors pipeline/config.py's HPA_K_CANONICAL; the default stays 10 so the
# published 1,304-TF atlas repackages unchanged.
K_CANONICAL = int(_env("HPA_K_CANONICAL", "10"))
ARCHETYPE_DN = ANALYSIS_DN / "tss_archetypes"
ARCHETYPE_GO_DN = ANALYSIS_DN / "enrichment_msigdb_gobp_archetypes"


################################################################################
# Helpers ######################################################################
################################################################################
def _ts() -> str:
    return datetime.now().strftime("%H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}")


def _exec(con, sql: str, *params):
    """Execute and ignore (for DDL / one-shot statements)."""
    con.execute(sql, list(params))


def _register_df(con, name: str, df: pd.DataFrame):
    """Register a pandas DataFrame as a temporary view for INSERT INTO ... SELECT."""
    con.register(name, df)


################################################################################
# Loaders ######################################################################
################################################################################
def load_tss(con):
    """tss table from matrices/tss_table.tsv. Adds an integer tss_id matching
    the per-peak records in tss_modules/peaks.parquet."""
    fn = ANALYSIS_DN / "tss_modules" / "tss_table.tsv"
    if not fn.exists():
        fn = ANALYSIS_DN / "matrices" / "tss_table.tsv"
    df = pd.read_csv(fn, sep="\t")
    df = df.reset_index(drop=True)
    df["tss_id"] = df.index.astype(np.int32)
    df = df[["tss_id", "transcript_id", "gene_id", "gene_name",
             "chrom", "tss", "strand"]]
    df["tss"] = df["tss"].astype(np.int32)
    df["transcript_id"] = df["transcript_id"].astype(str)
    df["gene_id"]       = df["gene_id"].astype(str)
    df["gene_name"]     = df["gene_name"].fillna("").astype(str)
    df["chrom"]         = df["chrom"].astype(str)
    df["strand"]        = df["strand"].astype(str)

    _exec(con, """
        CREATE TABLE tss (
            tss_id        INTEGER PRIMARY KEY,
            transcript_id VARCHAR,
            gene_id       VARCHAR,
            gene_name     VARCHAR,
            chrom         VARCHAR,
            tss           INTEGER,
            strand        VARCHAR
        );
    """)
    _register_df(con, "tss_df", df)
    _exec(con, "INSERT INTO tss SELECT * FROM tss_df;")
    _exec(con, "CREATE INDEX idx_tss_gene  ON tss(gene_name);")
    _exec(con, "CREATE INDEX idx_tss_tx    ON tss(transcript_id);")
    _log(f"  tss: {len(df):,} rows")
    return df


def load_tf(con):
    """tf metadata + cluster ids."""
    tf_idx = pd.read_csv(ANALYSIS_DN / "tss_modules" / "tf_index.tsv", sep="\t")
    tf_idx = tf_idx.sort_values("tf_idx").reset_index(drop=True)

    # tf_summary from main aggregate run (n_peaks_total etc.)
    summary_fn = ANALYSIS_DN / "matrices" / "tf_summary.tsv"
    if summary_fn.exists():
        tf_sum = pd.read_csv(summary_fn, sep="\t")
        tf_idx = tf_idx.merge(
            tf_sum[["TF", "n_peaks_total", "n_peaks_kept", "binary_total",
                    "score_total"]],
            on="TF", how="left",
        )
    else:
        for c in ("n_peaks_total", "n_peaks_kept", "binary_total", "score_total"):
            tf_idx[c] = np.nan

    # Per-TF bound-TSS count from tss_modules
    core_summary_fn = ANALYSIS_DN / "tss_programs" / "tf_summary.core100.tsv"
    if core_summary_fn.exists():
        tfb = pd.read_csv(core_summary_fn, sep="\t")
        tf_idx = tf_idx.merge(
            tfb[["TF", "n_bound_tss", "frac_tss"]].rename(
                columns={"n_bound_tss": "n_bound_tss_core",
                         "frac_tss":    "frac_tss_core"}),
            on="TF", how="left",
        )
    else:
        tf_idx["n_bound_tss_core"] = np.nan
        tf_idx["frac_tss_core"]     = np.nan

    # Cluster assignments
    cf = pd.read_csv(ANALYSIS_DN / "clustering" / "tf_cluster_table.tsv", sep="\t")
    cf = cf.rename(columns={"cluster": "cluster_filtered",
                            "Peak distance from TSS": "peak_dist_filtered"})
    tf_idx = tf_idx.merge(cf[["TF", "cluster_filtered", "peak_dist_filtered"]],
                          on="TF", how="left")
    cn = pd.read_csv(ANALYSIS_DN / "clustering_no_filter" / "tf_cluster_table.tsv", sep="\t")
    cn = cn.rename(columns={"cluster": "cluster_no_filter",
                            "Peak distance from TSS": "peak_dist_no_filter"})
    tf_idx = tf_idx.merge(cn[["TF", "cluster_no_filter", "peak_dist_no_filter"]],
                          on="TF", how="left")

    tf_idx = tf_idx.rename(columns={"TF": "tf"})
    tf_idx["tf_idx"] = tf_idx["tf_idx"].astype(np.int32)
    for c in ("cluster_filtered", "cluster_no_filter",
              "peak_dist_filtered", "peak_dist_no_filter",
              "n_peaks_total", "n_peaks_kept",
              "n_bound_tss_core"):
        if c in tf_idx.columns:
            tf_idx[c] = tf_idx[c].astype("Int64")

    _exec(con, """
        CREATE TABLE tf (
            tf_idx              INTEGER PRIMARY KEY,
            tf                  VARCHAR UNIQUE,
            n_peaks_total       BIGINT,
            n_peaks_kept        BIGINT,
            binary_total        DOUBLE,
            score_total         DOUBLE,
            n_bound_tss_core    BIGINT,
            frac_tss_core       DOUBLE,
            cluster_filtered    INTEGER,
            peak_dist_filtered  INTEGER,
            cluster_no_filter   INTEGER,
            peak_dist_no_filter INTEGER
        );
    """)
    _register_df(con, "tf_df", tf_idx)
    _exec(con, """
        INSERT INTO tf SELECT
            tf_idx, tf, n_peaks_total, n_peaks_kept, binary_total, score_total,
            n_bound_tss_core, frac_tss_core,
            cluster_filtered, peak_dist_filtered,
            cluster_no_filter, peak_dist_no_filter
        FROM tf_df;
    """)
    _exec(con, "CREATE INDEX idx_tf_name ON tf(tf);")
    _log(f"  tf: {len(tf_idx):,} rows")
    return tf_idx


def load_peaks(con):
    """The hot table — 12 M rows from peaks.parquet."""
    fn = ANALYSIS_DN / "tss_modules" / "peaks.parquet"
    _exec(con, f"""
        CREATE TABLE peaks AS
        SELECT
            CAST(tss_id AS INTEGER) AS tss_id,
            CAST(tf_idx AS INTEGER) AS tf_idx,
            CAST(local AS SMALLINT) AS local_offset,
            CAST(score AS SMALLINT) AS score
        FROM read_parquet('{fn}');
    """)
    _exec(con, "CREATE INDEX idx_peaks_tss ON peaks(tss_id);")
    _exec(con, "CREATE INDEX idx_peaks_tf  ON peaks(tf_idx);")
    n = con.execute("SELECT COUNT(*) FROM peaks;").fetchone()[0]
    _log(f"  peaks: {n:,} rows")


def load_modules_only(con):
    """Just the modules table.

    Used when the promoter build has no factorization of its own: the modules
    are still the gene-facing unit, the programs simply come from the
    genome-wide build instead. Shares its table definition with
    load_modules_and_programs rather than restating it, so the two cannot drift.
    """
    return _load_modules(con)


def load_modules_and_programs(con):
    """modules + module_program (canonical k) + programs."""
    _load_modules(con)
    _load_programs(con)


def _load_modules(con):
    mods = pd.read_csv(ANALYSIS_DN / "tss_modules" / "modules.tsv", sep="\t")

    keep = ["module_id", "tss_id", "transcript_id", "gene_name",
            "module_local_idx", "lo_offset", "hi_offset", "center_offset",
            "width", "n_peaks_in", "n_tfs_supporting", "n_tfs_assigned"]
    mods = mods[keep].copy()
    for c in ("module_id", "tss_id", "module_local_idx", "lo_offset",
              "hi_offset", "center_offset", "width", "n_peaks_in",
              "n_tfs_supporting", "n_tfs_assigned"):
        mods[c] = mods[c].astype(np.int32)
    mods["transcript_id"] = mods["transcript_id"].astype(str)
    mods["gene_name"]     = mods["gene_name"].fillna("").astype(str)

    _exec(con, """
        CREATE TABLE modules (
            module_id        INTEGER PRIMARY KEY,
            tss_id           INTEGER,
            transcript_id    VARCHAR,
            gene_name        VARCHAR,
            module_local_idx INTEGER,
            lo_offset        INTEGER,
            hi_offset        INTEGER,
            center_offset    INTEGER,
            width            INTEGER,
            n_peaks_in       INTEGER,
            n_tfs_supporting INTEGER,
            n_tfs_assigned   INTEGER
        );
    """)
    _register_df(con, "mods_df", mods)
    _exec(con, "INSERT INTO modules SELECT * FROM mods_df;")
    _exec(con, "CREATE INDEX idx_modules_tss ON modules(tss_id);")
    _exec(con, "CREATE INDEX idx_modules_tx  ON modules(transcript_id);")
    _exec(con, "CREATE INDEX idx_modules_gene ON modules(gene_name);")
    _log(f"  modules: {len(mods):,} rows")


def _load_programs(con):
    # module_program (canonical k)
    mp = pd.read_csv(
        ANALYSIS_DN / "tss_modules" / f"nmf.k{K_CANONICAL}.module_program.tsv",
        sep="\t",
    )
    keep_mp = ["module_id", "dominant_program", "dominant_weight"]
    keep_mp += [c for c in mp.columns if c.startswith("prog") and c.endswith("_w")]
    mp = mp[keep_mp].copy()
    mp["module_id"]        = mp["module_id"].astype(np.int32)
    mp["dominant_program"] = mp["dominant_program"].astype(np.int32)
    mp["dominant_weight"]  = mp["dominant_weight"].astype(np.float32)
    weight_cols = [c for c in mp.columns if c.startswith("prog") and c.endswith("_w")]
    for c in weight_cols:
        mp[c] = mp[c].astype(np.float32)

    cols_sql = ", ".join(f'"{c}" REAL' for c in weight_cols)
    _exec(con, f"""
        CREATE TABLE module_program (
            module_id        INTEGER PRIMARY KEY,
            dominant_program INTEGER,
            dominant_weight  REAL,
            {cols_sql}
        );
    """)
    _register_df(con, "mp_df", mp)
    _exec(con, "INSERT INTO module_program SELECT * FROM mp_df;")
    _exec(con, "CREATE INDEX idx_mp_program ON module_program(dominant_program);")
    _log(f"  module_program (k={K_CANONICAL}): {len(mp):,} rows")

    # programs summary
    summary = pd.read_csv(ANALYSIS_DN / "tss_modules" / f"nmf.k{K_CANONICAL}.summary.tsv",
                           sep="\t")
    summary["program"]         = summary["program"].astype(np.int32)
    summary["n_modules"]       = summary["n_modules"].astype(np.int32)
    summary["median_center"]   = summary["median_center"].astype(np.int32)
    summary["median_width"]    = summary["median_width"].astype(np.int32)
    summary["mean_dom_weight"] = summary["mean_dom_weight"].astype(np.float32)
    summary["top_tfs"] = summary["top_tfs"].astype(str)
    summary["reading"] = summary["reading"].astype(str)

    _exec(con, """
        CREATE TABLE programs (
            program          INTEGER PRIMARY KEY,
            n_modules        INTEGER,
            median_center    INTEGER,
            median_width     INTEGER,
            mean_dom_weight  REAL,
            top_tfs          VARCHAR,
            reading          VARCHAR
        );
    """)
    _register_df(con, "prog_df", summary)
    _exec(con, """
        INSERT INTO programs SELECT
            program, n_modules, median_center, median_width,
            mean_dom_weight, top_tfs, reading
        FROM prog_df;
    """)
    _log(f"  programs (k={K_CANONICAL}): {len(summary)} rows")


def load_program_tf_top(con):
    """Top-30 TFs per program."""
    fn = ANALYSIS_DN / "tss_modules" / f"nmf.k{K_CANONICAL}.top_tfs.tsv"
    df = pd.read_csv(fn, sep="\t")
    df["program"] = df["program"].astype(np.int32)
    df["rank"]    = df["rank"].astype(np.int32)
    df["tf"]      = df["tf"].astype(str)
    df["loading"] = df["loading"].astype(np.float32)

    _exec(con, """
        CREATE TABLE program_tf_top (
            program  INTEGER,
            rank     INTEGER,
            tf       VARCHAR,
            loading  REAL,
            PRIMARY KEY (program, rank)
        );
    """)
    _register_df(con, "ptt_df", df)
    _exec(con, "INSERT INTO program_tf_top SELECT * FROM ptt_df;")
    _exec(con, "CREATE INDEX idx_ptt_tf ON program_tf_top(tf);")
    _log(f"  program_tf_top: {len(df):,} rows")


def load_program_go_top(con):
    """Top-15 GO BP terms per program (genome_bg)."""
    fn = (ANALYSIS_DN / "enrichment_msigdb_gobp_modules"
                     / f"k{K_CANONICAL}" / "program_top_terms.tsv")
    if not fn.exists():
        _log(f"  program_go_top: SKIPPED (file not found: {fn})")
        return
    df = pd.read_csv(fn, sep="\t")
    df["program"]        = df["program"].astype(np.int32)
    df["rank"]           = df["rank"].astype(np.int32)
    df["term"]           = df["term"].astype(str)
    df["go_id"]          = df["go_id"].fillna("").astype(str)
    df["fg_in"]          = df["fg_in"].astype(np.int32)
    df["set_size_in_bg"] = df["set_size_in_bg"].astype(np.int32)
    df["odds_ratio"]     = df["odds_ratio"].astype(np.float32)
    df["p_value"]        = df["p_value"].astype(np.float64)
    df["q_value"]        = df["q_value"].astype(np.float64)
    if "genes_in_overlap" in df.columns:
        df["genes_in_overlap"] = df["genes_in_overlap"].astype(str)
    else:
        df["genes_in_overlap"] = ""

    _exec(con, """
        CREATE TABLE program_go_top (
            program          INTEGER,
            rank             INTEGER,
            term             VARCHAR,
            go_id            VARCHAR,
            fg_in            INTEGER,
            set_size_in_bg   INTEGER,
            odds_ratio       REAL,
            p_value          DOUBLE,
            q_value          DOUBLE,
            genes_in_overlap VARCHAR,
            PRIMARY KEY (program, rank)
        );
    """)
    _register_df(con, "pgo_df", df)
    _exec(con, """
        INSERT INTO program_go_top SELECT
            program, rank, term, go_id, fg_in, set_size_in_bg,
            odds_ratio, p_value, q_value, genes_in_overlap
        FROM pgo_df;
    """)
    _log(f"  program_go_top: {len(df):,} rows")


def load_archetypes(con) -> int:
    """Read tss_archetypes outputs, return canonical A used."""
    canonical_fn = ARCHETYPE_DN / "canonical_A.txt"
    if not canonical_fn.exists():
        _log("  archetypes: canonical_A.txt not found — skipping")
        return 0
    A = int(canonical_fn.read_text().strip())
    _log(f"  canonical archetype rank A = {A}")

    # gene_archetypes
    ga = pd.read_csv(ARCHETYPE_DN / f"nmf.A{A}.gene_archetype.tsv", sep="\t")
    keep = ["transcript_id", "gene_name", "n_modules",
            "dominant_archetype", "dominant_weight"]
    weight_cols = [c for c in ga.columns
                   if c.startswith("A") and c.endswith("_w")]
    ga = ga[keep + weight_cols].copy()
    ga["transcript_id"] = ga["transcript_id"].astype(str)
    ga["gene_name"]     = ga["gene_name"].fillna("").astype(str)
    ga["n_modules"]          = ga["n_modules"].astype(np.int32)
    ga["dominant_archetype"] = ga["dominant_archetype"].astype(np.int32)
    ga["dominant_weight"]    = ga["dominant_weight"].astype(np.float32)
    for c in weight_cols:
        ga[c] = ga[c].astype(np.float32)

    weight_sql = ", ".join(f'"{c}" REAL' for c in weight_cols)
    _exec(con, f"""
        CREATE TABLE gene_archetypes (
            transcript_id      VARCHAR PRIMARY KEY,
            gene_name          VARCHAR,
            n_modules          INTEGER,
            dominant_archetype INTEGER,
            dominant_weight    REAL,
            {weight_sql}
        );
    """)
    _register_df(con, "ga_df", ga)
    _exec(con, "INSERT INTO gene_archetypes SELECT * FROM ga_df;")
    _exec(con, "CREATE INDEX idx_ga_arch ON gene_archetypes(dominant_archetype);")
    _exec(con, "CREATE INDEX idx_ga_gene ON gene_archetypes(gene_name);")
    _log(f"  gene_archetypes: {len(ga):,} rows")

    # archetypes summary
    arsum = pd.read_csv(ARCHETYPE_DN / f"nmf.A{A}.archetype_summary.tsv",
                         sep="\t")
    arsum["archetype"] = arsum["archetype"].astype(np.int32)
    arsum["n_genes"]   = arsum["n_genes"].astype(np.int32)
    arsum["frac_genes"] = arsum["frac_genes"].astype(np.float32)
    arsum["mean_modules_per_gene"] = arsum["mean_modules_per_gene"].astype(np.float32)
    arsum["top_programs"] = arsum["top_programs"].astype(str)
    arsum["top_loadings"] = arsum["top_loadings"].astype(str)
    _exec(con, """
        CREATE TABLE archetypes (
            archetype             INTEGER PRIMARY KEY,
            n_genes               INTEGER,
            frac_genes            REAL,
            mean_modules_per_gene REAL,
            top_programs          VARCHAR,
            top_loadings          VARCHAR
        );
    """)
    _register_df(con, "as_df", arsum)
    _exec(con, "INSERT INTO archetypes SELECT * FROM as_df;")
    _log(f"  archetypes: {len(arsum)} rows")

    # Archetype × program H matrix → long form
    H = pd.read_csv(ARCHETYPE_DN / f"nmf.A{A}.H.tsv.gz",
                     sep="\t", index_col=0)
    long_rows = []
    for arch_label, row in H.iterrows():
        a = int(arch_label.replace("A", ""))
        for col, v in row.items():
            p = int(col.replace("P", ""))
            long_rows.append({"archetype": a, "program": p,
                              "loading": float(v)})
    arch_prog = pd.DataFrame(long_rows)
    _exec(con, """
        CREATE TABLE archetype_program_loading (
            archetype INTEGER,
            program   INTEGER,
            loading   REAL,
            PRIMARY KEY (archetype, program)
        );
    """)
    _register_df(con, "ap_df", arch_prog)
    _exec(con, "INSERT INTO archetype_program_loading SELECT * FROM ap_df;")
    _log(f"  archetype_program_loading: {len(arch_prog)} rows")

    # Top GO BP terms per archetype
    go_fn = ARCHETYPE_GO_DN / f"A{A}" / "archetype_top_terms.tsv"
    if go_fn.exists():
        go = pd.read_csv(go_fn, sep="\t")
        go["archetype"]      = go["archetype"].astype(np.int32)
        go["rank"]           = go["rank"].astype(np.int32)
        go["term"]           = go["term"].astype(str)
        go["go_id"]          = go["go_id"].fillna("").astype(str)
        go["fg_in"]          = go["fg_in"].astype(np.int32)
        go["set_size_in_bg"] = go["set_size_in_bg"].astype(np.int32)
        go["odds_ratio"]     = go["odds_ratio"].astype(np.float32)
        go["p_value"]        = go["p_value"].astype(np.float64)
        go["q_value"]        = go["q_value"].astype(np.float64)
        if "genes_in_overlap" in go.columns:
            go["genes_in_overlap"] = go["genes_in_overlap"].astype(str)
        else:
            go["genes_in_overlap"] = ""
        _exec(con, """
            CREATE TABLE archetype_go_top (
                archetype        INTEGER,
                rank             INTEGER,
                term             VARCHAR,
                go_id            VARCHAR,
                fg_in            INTEGER,
                set_size_in_bg   INTEGER,
                odds_ratio       REAL,
                p_value          DOUBLE,
                q_value          DOUBLE,
                genes_in_overlap VARCHAR,
                PRIMARY KEY (archetype, rank)
            );
        """)
        _register_df(con, "ag_df", go)
        _exec(con, """
            INSERT INTO archetype_go_top SELECT
                archetype, rank, term, go_id, fg_in, set_size_in_bg,
                odds_ratio, p_value, q_value, genes_in_overlap
            FROM ag_df;
        """)
        _log(f"  archetype_go_top: {len(go):,} rows")
    else:
        _log("  archetype_go_top: (file missing)")

    return A


def load_gene_configs(con):
    """Per-gene program_path configurations."""
    fn = ANALYSIS_DN / "tss_modules" / f"nmf.k{K_CANONICAL}.gene_configurations.tsv"
    df = pd.read_csv(fn, sep="\t")
    df["transcript_id"] = df["transcript_id"].astype(str)
    df["gene_name"]     = df["gene_name"].fillna("").astype(str)
    df["n_modules"]     = df["n_modules"].astype(np.int32)
    df["program_path"]  = df["program_path"].astype(str)
    df["centers"]       = df["centers"].astype(str)
    df["widths"]        = df["widths"].astype(str)

    _exec(con, """
        CREATE TABLE gene_configs (
            transcript_id VARCHAR PRIMARY KEY,
            gene_name     VARCHAR,
            n_modules     INTEGER,
            program_path  VARCHAR,
            centers       VARCHAR,
            widths        VARCHAR
        );
    """)
    _register_df(con, "gc_df", df)
    _exec(con, "INSERT INTO gene_configs SELECT * FROM gc_df;")
    _exec(con, "CREATE INDEX idx_gc_gene ON gene_configs(gene_name);")
    _exec(con, "CREATE INDEX idx_gc_path ON gene_configs(program_path);")
    _log(f"  gene_configs: {len(df):,} rows")


def copy_aggregate_matrices():
    """Copy the aggregate TF×position parquets into data/aggregate/."""
    AGG_DN.mkdir(parents=True, exist_ok=True)
    src_dir = ANALYSIS_DN / "matrices"
    for stem in ("tf_x_position.binary", "tf_x_position.score",
                 "tf_x_position.raw", "tf_x_position.raw_score1000"):
        src = src_dir / f"{stem}.parquet"
        if not src.exists():
            _log(f"  [warn] missing {src.name}")
            continue
        dst = AGG_DN / src.name
        shutil.copy2(src, dst)
        _log(f"  aggregate: {src.name} -> data/aggregate/")


def copy_depmap_outputs():
    """Copy DepMap parquets into data/depmap/."""
    src_dir = ANALYSIS_DN / "tss_depmap"
    if not src_dir.exists():
        _log("  depmap: tss_depmap/ not found — skipping")
        return
    dst_dir = DATA_DN / "depmap"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("gene_lineage_essentiality",
                 "gene_essentiality_summary"):
        src = src_dir / f"{stem}.parquet"
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
            _log(f"  depmap: {src.name} ({dst.stat().st_size/1e6:.1f} MB) "
                 f"-> data/depmap/" if (dst := dst_dir / src.name).exists() else "")
    lin = src_dir / "lineage_index.tsv"
    if lin.exists():
        shutil.copy2(lin, dst_dir / lin.name)
        _log(f"  depmap: lineage_index.tsv -> data/depmap/")


def copy_gtex_outputs():
    """Copy the GTEx parquets into data/gtex/ for the app to load on demand.

    These are ~50 MB total and are kept as parquet (not duckdb tables) since
    the app's hot path reads them as full numpy arrays for plotting/sorting,
    not via SQL filters.
    """
    src_dir = ANALYSIS_DN / "tss_gtex"
    if not src_dir.exists():
        _log("  gtex: tss_gtex/ not found — skipping")
        return
    dst_dir = DATA_DN / "gtex"
    dst_dir.mkdir(parents=True, exist_ok=True)
    for stem in ("transcript_tissue_stats",
                 "transcript_tissue_mean",
                 "tf_tissue_expression",
                 "tf_target_correlation",
                 "module_tissue_activity",
                 "module_target_correlation",
                 "module_tf_target_correlation",
                 "module_supporting_tissues"):
        src = src_dir / f"{stem}.parquet"
        if not src.exists():
            _log(f"  gtex: missing {src.name}")
            continue
        dst = dst_dir / src.name
        shutil.copy2(src, dst)
        _log(f"  gtex: {src.name} ({dst.stat().st_size/1e6:.1f} MB) "
             f"-> data/gtex/")
    tsvs = ["tissue_index.tsv"]
    # program_tissue_specificity.tsv is keyed on the PROMOTER factorization's
    # program numbers. Under the hybrid architecture the site numbers programs
    # from the genome build, so shipping this file would put a table of
    # "program 7" tissue specificity next to a completely different program 7.
    # A stale copy on disk is worse than none: it would look current.
    if _promoter_programs_available():
        tsvs.append("program_tissue_specificity.tsv")
    else:
        stale = src_dir / "program_tissue_specificity.tsv"
        if stale.exists():
            _log("  gtex: skipping program_tissue_specificity.tsv "
                 "(keyed on promoter program numbers the site no longer uses)")
    for tsv_name in tsvs:
        src = src_dir / tsv_name
        if src.exists():
            shutil.copy2(src, dst_dir / src.name)
            _log(f"  gtex: {src.name} -> data/gtex/")


# Fields copied out of the upstream build stamp, by explicit whitelist. The
# stamp also records per_tf_dn, which is a cluster scratch path -- manifest.json
# is committed to a public repo, so this stays an allowlist and never becomes a
# blanket copy.
_BUILD_STAMP_KEYS = ("tier", "qvalue", "tf_set", "min_score_assign")


def read_build_stamp() -> dict:
    """The build's own ``_BUILD.json``, which is the authority on its axes.

    ``config.py`` derives the build directory from tier/tf_set/score, and the
    one rule of that layout is that nothing re-implements it -- a second copy of
    the path rule is what once made a rerun skip the stage it was asked to redo
    and exit 0. So the axes are read from the stamp the pipeline wrote, neither
    re-derived here nor imported from ``config``: importing it would demand
    HPA_CHIP_ATLAS_DIR and friends, which this packing step has no use for.

    Returns ``{}`` for the original flat layout, which predates the stamp.
    """
    stamp_fn = ANALYSIS_DN / "_BUILD.json"
    if not stamp_fn.is_file():
        _log(f"  manifest: no _BUILD.json under {ANALYSIS_DN} — "
             f"build axes recorded as null")
        return {}
    stamp = json.loads(stamp_fn.read_text())
    return {k: stamp[k] for k in _BUILD_STAMP_KEYS if k in stamp}


def write_manifest(canonical_A: int = 0, counts: dict | None = None):
    """Versions + parameters for the Methods tab.

    ``counts`` are read back out of the DuckDB just written, so the numbers the
    Methods tab prints describe the data this app actually serves. They used to
    be prose constants in the tab source, and they drifted: the text still said
    1,304 TFs and score >= 500 after the assignment threshold was recalibrated.
    """
    canA_path = ARCHETYPE_DN / "canonical_A.txt"
    if canonical_A == 0 and canA_path.exists():
        canonical_A = int(canA_path.read_text().strip())
    stamp = read_build_stamp()
    counts = counts or {}
    manifest = {
        "built_at": datetime.now().isoformat(timespec="seconds"),
        "analysis_dn": str(ANALYSIS_DN),
        "k_canonical": K_CANONICAL,
        "a_canonical": canonical_A,
        "build": {
            "tier": stamp.get("tier"),
            "qvalue": stamp.get("qvalue"),
            "tf_set": stamp.get("tf_set"),
            "min_score_assign": stamp.get("min_score_assign"),
        },
        "counts": {
            "n_tf": counts.get("tf"),
            "n_tss": counts.get("tss"),
            "n_modules": counts.get("modules"),
            "n_programs": counts.get("programs"),
        },
        "datasets": {
            "ensembl": "GRCh38.114",
            "chip_atlas": "TF/per_TF (snapshot in analyses/canonical_promoter)",
            "msigdb": "c5.go.bp.v2026.1.Hs",
        },
        "parameters": {
            "tss_window_aggregate_bp": [-1000, 1000],
            "peak_recenter_half_bp": 12,
            "tss_modules_outer_half_bp": 1500,
            "tss_modules_kde_bw_bp": 25,
            "tss_modules_min_support_tfs": 2,
            # The same quantity as the tier (score = -10*log10(Q)), so it comes
            # from the build stamp rather than being restated as a literal here.
            "tss_modules_min_score_assign": stamp.get("min_score_assign"),
            "tss_modules_boundary_frac": 0.20,
            "valid_chroms": ["1-22", "X", "Y", "MT"],
        },
        "tables": [
            "tss", "tf", "peaks", "modules", "module_program",
            "programs", "program_tf_top", "program_go_top",
            "tf_clusters", "gene_configs",
        ],
    }
    with open(MANIFEST_FN, "w") as f:
        json.dump(manifest, f, indent=2)
    _log(f"  manifest: {MANIFEST_FN}")


################################################################################
# Execution ####################################################################
################################################################################
# When the promoter build carries no factorization of its own, everything
# downstream of it has no source: module_program, programs, program_tf_top,
# program_go_top, gene_configs and the archetypes all derive from a promoter
# NMF. That is the intended state for the hybrid architecture -- programs come
# from the genome-wide build via build_app_db_genome.py, one vocabulary across
# the site -- so the absence is detected and skipped rather than crashing.
#
# EXPLICIT, not detected. Detecting by file existence looked tidier but gives
# the wrong answer: stale nmf.k*.module_program.tsv files from earlier runs sit
# in the build directory, so detection would silently re-enable promoter
# programs and ship two competing program vocabularies. Default off; set
# HPA_PROMOTER_PROGRAMS=1 to build them, and that fails loudly if the inputs
# are absent rather than quietly falling back.
def _promoter_programs_available() -> bool:
    if _env("HPA_PROMOTER_PROGRAMS", "0") not in ("1", "true", "True"):
        return False
    fn = (ANALYSIS_DN / "tss_modules" /
          f"nmf.k{K_CANONICAL}.module_program.tsv")
    if not fn.exists():
        raise SystemExit(
            f"HPA_PROMOTER_PROGRAMS=1 but {fn} does not exist. Either run the "
            f"promoter factorization at k={K_CANONICAL} or leave the flag off "
            f"and take programs from the genome build.")
    return True


def main():
    DATA_DN.mkdir(parents=True, exist_ok=True)
    if DUCKDB_FN.exists():
        DUCKDB_FN.unlink()

    _log(f"building {DUCKDB_FN}")
    con = duckdb.connect(str(DUCKDB_FN))
    try:
        load_tss(con)
        load_tf(con)
        load_peaks(con)
        canonical_A = 0
        if _promoter_programs_available():
            load_modules_and_programs(con)
            load_program_tf_top(con)
            load_program_go_top(con)
            load_gene_configs(con)
            canonical_A = load_archetypes(con)
        else:
            _log(f"no nmf.k{K_CANONICAL}.module_program.tsv in "
                 f"{ANALYSIS_DN.name}: loading modules WITHOUT programs.")
            _log("  programs come from the genome build "
                 "(data/build_app_db_genome.py) -- one vocabulary site-wide.")
            load_modules_only(con)

        # tf_clusters convenience view (long format for the Aggregate tab)
        _exec(con, """
            CREATE VIEW tf_clusters AS
            SELECT tf, 'filtered'  AS cluster_set,
                   cluster_filtered    AS cluster,
                   peak_dist_filtered  AS peak_distance_from_tss
            FROM tf WHERE cluster_filtered IS NOT NULL
            UNION ALL
            SELECT tf, 'no_filter' AS cluster_set,
                   cluster_no_filter   AS cluster,
                   peak_dist_no_filter AS peak_distance_from_tss
            FROM tf WHERE cluster_no_filter IS NOT NULL;
        """)
        # Sanity counts. Also the source of the manifest's `counts` block, so
        # the Methods tab quotes the data it is serving rather than a literal.
        _log("table sizes:")
        counts = {}
        tables = ["tss", "tf", "peaks", "modules"]
        if _promoter_programs_available():
            tables += ["module_program", "programs", "program_tf_top",
                       "program_go_top", "gene_configs"]
        if canonical_A:
            tables += ["gene_archetypes", "archetypes",
                        "archetype_program_loading", "archetype_go_top"]
        for t in tables:
            try:
                n = con.execute(f"SELECT COUNT(*) FROM {t};").fetchone()[0]
                counts[t] = n
                _log(f"  {t:28s}  {n:>12,}")
            except Exception:
                pass
    finally:
        con.close()

    copy_aggregate_matrices()
    copy_gtex_outputs()
    copy_depmap_outputs()
    write_manifest(canonical_A, counts)
    size_mb = DUCKDB_FN.stat().st_size / 1e6
    _log(f"DONE — {DUCKDB_FN.name} = {size_mb:.1f} MB")


if __name__ == "__main__":
    main()
