"""Precompute Pearson r between each atlas TF's DepMap Chronos and each
atlas target's log10(TPM+1) expression across DepMap cell lines.

Output sharding: one parquet per target gene at
  data/depmap/tf_target_corr/{GENE}.parquet

Schema per file (long):
  tf : categorical string
  r  : int8     (= round(r * 100), range −100…+100; ~0.5% precision)
  n  : int16    (n cell lines with paired non-NaN observations)

Pairs with n < 50 are dropped. Each file is ~1–8 KB. Total atlas:
≤ 19,354 files. Run once:

    python data/build_depmap_tf_target_correlations.py
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

OUT_DIR = REPO_DIR / "data" / "depmap" / "tf_target_corr"
MIN_N   = 50


def _strip_entrez(col: str) -> str:
    i = col.find(" (")
    return col[:i] if i > 0 else col


def _safe_filename(name: str) -> str:
    return "".join(c if (c.isalnum() or c in "._-") else "_" for c in name)


def main() -> None:
    if not CHRONOS_CSV.exists() or not EXPR_CSV.exists():
        sys.exit(f"DepMap raw CSVs not found under {DEPMAP_RAW}")
    if not DB_PATH.exists():
        sys.exit(f"DuckDB not found at {DB_PATH}. Run build_app_db.py first.")

    print("Reading atlas TF + target lists from DuckDB…", flush=True)
    con = duckdb.connect(str(DB_PATH), read_only=True)
    atlas_tfs = set(con.execute("SELECT tf FROM tf").df()["tf"])
    atlas_targets = set(
        con.execute("SELECT DISTINCT gene_name FROM tss "
                    "WHERE gene_name IS NOT NULL").df()["gene_name"]
    )
    print(f"  atlas TFs:     {len(atlas_tfs):>6,}")
    print(f"  atlas targets: {len(atlas_targets):>6,}")

    print("Loading Chronos matrix…", flush=True)
    t0 = time.time()
    chro = pd.read_csv(CHRONOS_CSV, index_col=0)
    chro.columns = [_strip_entrez(c) for c in chro.columns]
    chro = chro.loc[:, ~chro.columns.duplicated()]
    chro = chro[[c for c in chro.columns if c in atlas_tfs]]
    print(f"  shape: {chro.shape}  ({time.time()-t0:.1f}s)")

    print("Loading expression matrix…", flush=True)
    t0 = time.time()
    expr = pd.read_csv(EXPR_CSV, index_col=0)
    expr.columns = [_strip_entrez(c) for c in expr.columns]
    expr = expr.loc[:, ~expr.columns.duplicated()]
    expr = expr[[c for c in expr.columns if c in atlas_targets]]
    print(f"  shape: {expr.shape}  ({time.time()-t0:.1f}s)")

    common = chro.index.intersection(expr.index)
    print(f"Cell-line intersection: {len(common)}")
    if len(common) < MIN_N:
        sys.exit("Not enough overlapping cell lines.")

    chro_a = chro.loc[common].to_numpy(dtype=np.float64)
    expr_a = expr.loc[common].to_numpy(dtype=np.float64)
    tfs     = chro.columns.to_numpy()
    targets = expr.columns.to_numpy()
    n_tfs, n_targets = len(tfs), len(targets)

    # Preallocate result matrices [n_tfs × n_targets]. Total ~100 MB.
    R_pct = np.zeros((n_tfs, n_targets), dtype=np.int8)   # r * 100
    N_obs = np.zeros((n_tfs, n_targets), dtype=np.int16)  # n cell lines
    KEEP  = np.zeros((n_tfs, n_targets), dtype=bool)

    print(f"Computing r for {n_tfs:,} TFs × {n_targets:,} targets…", flush=True)
    t0 = time.time()
    for j in range(n_tfs):
        if j % 100 == 0:
            print(f"  TF {j:>5}/{n_tfs}  ({tfs[j]})", flush=True)
        c = chro_a[:, j]
        mask_c = np.isfinite(c)
        if mask_c.sum() < MIN_N:
            continue
        c_v = c[mask_c]
        E_v = expr_a[mask_c, :]            # cells × targets, may contain NaN
        valid = np.isfinite(E_v)           # per-(cell, target) validity
        n = valid.sum(axis=0).astype(np.int32)

        # Zero-fill NaN so the sums skip them naturally.
        Ez = np.where(valid, E_v, 0.0)
        c_col = c_v[:, None]
        sum_y  = Ez.sum(axis=0)
        sum_y2 = (Ez * Ez).sum(axis=0)
        sum_x  = (c_col * valid).sum(axis=0)
        sum_x2 = (c_col * c_col * valid).sum(axis=0)
        sum_xy = (c_col * Ez).sum(axis=0)

        numer   = n * sum_xy - sum_x * sum_y
        denom_x = n * sum_x2 - sum_x * sum_x
        denom_y = n * sum_y2 - sum_y * sum_y
        denom   = np.sqrt(denom_x * denom_y)
        with np.errstate(divide="ignore", invalid="ignore"):
            r = np.where(denom > 0, numer / denom, np.nan)

        keep = (n >= MIN_N) & np.isfinite(r)
        if not keep.any():
            continue
        r_int = np.clip(np.round(np.where(keep, r, 0.0) * 100),
                        -100, 100).astype(np.int8)
        n_int = np.where(keep, n, 0).astype(np.int16)
        R_pct[j] = r_int
        N_obs[j] = n_int
        KEEP[j]  = keep
    print(f"  done in {time.time()-t0:.1f}s")

    print(f"Writing one parquet per target to {OUT_DIR}…", flush=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    # Clean any stale shards from a prior run so we don't keep ghosts.
    for old in OUT_DIR.glob("*.parquet"):
        old.unlink()

    t0 = time.time()
    n_files = 0
    n_rows  = 0
    for k in range(n_targets):
        mask = KEEP[:, k]
        if not mask.any():
            continue
        df = pd.DataFrame({
            "tf": pd.Categorical(tfs[mask]),
            "r":  R_pct[mask, k],
            "n":  N_obs[mask, k],
        })
        fn = OUT_DIR / f"{_safe_filename(targets[k])}.parquet"
        df.to_parquet(fn, index=False, compression="zstd")
        n_files += 1
        n_rows  += len(df)
    print(f"  wrote {n_files:,} files, {n_rows:,} total rows "
          f"({time.time()-t0:.1f}s)")

    total = sum(p.stat().st_size for p in OUT_DIR.glob("*.parquet"))
    print(f"Total disk: {total / 1024 / 1024:.1f} MB  "
          f"(mean {total / max(n_files, 1) / 1024:.1f} KB/file)")


if __name__ == "__main__":
    main()
