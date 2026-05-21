"""Slim parquet versions of the DepMap matrices used for the per-cell-line
TF–target scatter plot. The raw DepMap CSVs are ~834 MB together and not
suitable for production deploy. This script trims them to the atlas-relevant
subset and writes:

  data/depmap/chronos_matrix.parquet     cells × atlas TFs        (~1186 × 1293)
  data/depmap/expression_matrix.parquet  cells × atlas targets    (~1450 × 18878)
  data/depmap/cell_line_meta.parquet     ModelID → CellLineName, OncotreeLineage

All values stored as float32 (sufficient precision for log10(TPM+1) and Chronos),
gene-symbol columns only (Entrez stripped), gene/cell-line indices preserved.

Run once:

    python data/build_depmap_pair_matrices.py
"""
from __future__ import annotations

import sys
import os
import time
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
DB_PATH  = REPO_DIR / "data" / "canonical_promoter.duckdb"

# Override with HPA_DEPMAP_RAW if the raw DepMap CSVs live elsewhere.
DEPMAP_RAW  = Path(os.environ.get("HPA_DEPMAP_RAW", "data/raw/depmap"))
CHRONOS_CSV = DEPMAP_RAW / "CRISPRGeneEffect.csv"
EXPR_CSV    = DEPMAP_RAW / "protein_coding_expr" / "OmicsExpressionProteinCodingGenesTPMLogp1.csv"
MODEL_CSV   = DEPMAP_RAW / "Model.csv"

OUT_DIR = REPO_DIR / "data" / "depmap"
OUT_CHRONOS = OUT_DIR / "chronos_matrix.parquet"
OUT_EXPR    = OUT_DIR / "expression_matrix.parquet"
OUT_META    = OUT_DIR / "cell_line_meta.parquet"


def _strip_entrez(col: str) -> str:
    i = col.find(" (")
    return col[:i] if i > 0 else col


def _save_wide(df: pd.DataFrame, out: Path, label: str) -> None:
    """Write a wide [cells × genes] matrix to parquet with float32 values
    and the cell-line ID as a normal column (so column-projection reads
    don't need to materialize an index)."""
    t0 = time.time()
    df = df.astype("float32")
    df = df.reset_index().rename(columns={df.index.name or "index": "ModelID"})
    df.to_parquet(out, index=False, compression="zstd")
    size_mb = out.stat().st_size / 1024 / 1024
    print(f"  wrote {label}: {out.name}  ({df.shape[0]}×{df.shape[1]-1} "
          f"cells×genes, {size_mb:.1f} MB, {time.time()-t0:.1f}s)")


def main() -> None:
    if not CHRONOS_CSV.exists() or not EXPR_CSV.exists() or not MODEL_CSV.exists():
        sys.exit(f"DepMap raw CSVs not found under {DEPMAP_RAW}")
    if not DB_PATH.exists():
        sys.exit(f"DuckDB not found at {DB_PATH}. Run build_app_db.py first.")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    con = duckdb.connect(str(DB_PATH), read_only=True)
    atlas_tfs = set(con.execute("SELECT tf FROM tf").df()["tf"])
    atlas_targets = set(
        con.execute("SELECT DISTINCT gene_name FROM tss "
                    "WHERE gene_name IS NOT NULL").df()["gene_name"]
    )
    print(f"atlas TFs: {len(atlas_tfs):,}   atlas targets: {len(atlas_targets):,}")

    # ----- Chronos --------------------------------------------------------
    print("Loading Chronos CSV…", flush=True)
    t0 = time.time()
    chro = pd.read_csv(CHRONOS_CSV, index_col=0)
    chro.columns = [_strip_entrez(c) for c in chro.columns]
    chro = chro.loc[:, ~chro.columns.duplicated()]
    chro = chro[[c for c in chro.columns if c in atlas_tfs]]
    chro.index.name = "ModelID"
    print(f"  loaded in {time.time()-t0:.1f}s, shape {chro.shape}")
    _save_wide(chro, OUT_CHRONOS, "Chronos")

    # ----- Expression -----------------------------------------------------
    print("Loading expression CSV…", flush=True)
    t0 = time.time()
    expr = pd.read_csv(EXPR_CSV, index_col=0)
    expr.columns = [_strip_entrez(c) for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    expr = expr[[c for c in expr.columns if c in atlas_targets]]
    expr.index.name = "ModelID"
    print(f"  loaded in {time.time()-t0:.1f}s, shape {expr.shape}")
    _save_wide(expr, OUT_EXPR, "Expression")

    # ----- Cell-line metadata --------------------------------------------
    print("Loading Model.csv…", flush=True)
    t0 = time.time()
    meta = pd.read_csv(MODEL_CSV,
                       usecols=["ModelID", "CellLineName", "OncotreeLineage"])
    meta.to_parquet(OUT_META, index=False, compression="zstd")
    print(f"  wrote {OUT_META.name}: {len(meta):,} rows, "
          f"{OUT_META.stat().st_size / 1024:.1f} KB, "
          f"{time.time()-t0:.1f}s")

    total = sum(p.stat().st_size for p in
                [OUT_CHRONOS, OUT_EXPR, OUT_META])
    print(f"\nTotal disk: {total / 1024 / 1024:.1f} MB  "
          f"(vs ~834 MB of source CSVs)")


if __name__ == "__main__":
    main()
