"""Precompute the atlas-wide TF × TF co-occurrence table.

Source: data/gtex/module_tf_target_correlation.parquet — the per-(module, TF)
membership table (~2.1 M rows, ~27 TFs / module avg over 76,788 modules).

For every unordered pair of TFs that co-occur in at least one module we
compute:
  n_shared : # modules containing BOTH TFs
  n_a      : total modules containing TF A
  n_b      : total modules containing TF B
  jaccard  : n_shared / (n_a + n_b - n_shared)       — symmetric overlap
  lift     : observed / expected under independence,
             where expected = N_total_modules * (n_a / N) * (n_b / N).

Output: data/tf_pair_cooccurrence.parquet (long format, lower-triangle
only — TF A < TF B alphabetically — so each unordered pair appears once).

Run once:

    python data/build_tf_pair_table.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_PQ   = REPO_DIR / "data" / "gtex" / "module_tf_target_correlation.parquet"
OUT_PQ   = REPO_DIR / "data" / "network" / "tf_pair_cooccurrence.parquet"
OUT_GZ   = REPO_DIR / "data" / "network" / "tf_pair_cooccurrence.tsv.gz"

# Drop pairs that almost never co-occur — keeps the file small without
# losing any analytically interesting signal. 5 shared modules is the
# practical floor (anything below it has too much noise to interpret).
MIN_SHARED = 5


def main() -> None:
    if not SRC_PQ.exists():
        sys.exit(f"source parquet not found at {SRC_PQ}")

    print("Loading module-TF membership…", flush=True)
    t0 = time.time()
    df = pd.read_parquet(SRC_PQ, columns=["module_id", "tf"])
    print(f"  {len(df):,} rows in {time.time()-t0:.1f}s")

    # Build a sparse boolean module × TF matrix. modules are dense int ids;
    # TFs we encode as a Categorical → small int code.
    df["tf"] = df["tf"].astype("category")
    tfs = list(df["tf"].cat.categories)
    n_tfs = len(tfs)
    n_modules = int(df["module_id"].max()) + 1
    print(f"  {n_tfs} unique TFs · {n_modules} modules (max id + 1)")

    t0 = time.time()
    rows = df["module_id"].to_numpy(dtype=np.int32)
    cols = df["tf"].cat.codes.to_numpy(dtype=np.int32)
    # int32 matters: B.T @ B is computed in B's dtype, and int8 overflows
    # at 127 — many co-binding pairs exceed that (CTCF/cohesin ~14k).
    data = np.ones(len(df), dtype=np.int32)
    B = csr_matrix((data, (rows, cols)), shape=(n_modules, n_tfs))
    # Modules that are completely absent from the source (id holes) just
    # have zero rows — they contribute nothing to the co-occurrence matrix.
    N_used = int((B.sum(axis=1) > 0).sum())
    print(f"  binary matrix {B.shape} built in {time.time()-t0:.1f}s "
          f"({N_used} populated modules)")

    print("Computing co-occurrence (B.T @ B)…", flush=True)
    t0 = time.time()
    C = (B.T @ B).toarray().astype(np.int32)   # n_tfs × n_tfs
    marginals = np.diag(C).copy().astype(np.int32)
    print(f"  co-occurrence matrix in {time.time()-t0:.1f}s, "
          f"shape {C.shape}, dtype {C.dtype}")

    # Lower-triangular extraction — only i < j to dedupe symmetric pairs.
    print("Building long-format pair table…", flush=True)
    t0 = time.time()
    ii, jj = np.triu_indices(n_tfs, k=1)              # upper triangle (i < j)
    n_shared = C[ii, jj]
    keep = n_shared >= MIN_SHARED
    ii, jj, n_shared = ii[keep], jj[keep], n_shared[keep]
    n_a = marginals[ii]
    n_b = marginals[jj]
    # Jaccard
    denom_j = (n_a.astype(np.float64) + n_b - n_shared)
    jaccard = np.where(denom_j > 0, n_shared / denom_j, 0.0)
    # Lift = observed / expected. expected = N * (n_a/N) * (n_b/N) = n_a*n_b/N.
    expected = (n_a.astype(np.float64) * n_b) / max(N_used, 1)
    lift = np.where(expected > 0, n_shared / expected, np.nan)

    tf_arr = np.array(tfs, dtype=object)
    out = pd.DataFrame({
        "tf_a":     pd.Categorical(tf_arr[ii]),
        "tf_b":     pd.Categorical(tf_arr[jj]),
        "n_shared": n_shared.astype(np.int32),
        "n_a":      n_a.astype(np.int32),
        "n_b":      n_b.astype(np.int32),
        "jaccard":  np.round(jaccard, 4).astype(np.float32),
        "lift":     np.round(lift,    3).astype(np.float32),
    })
    # Sort by n_shared desc so the parquet's "natural" order is already
    # the most useful sort for the app.
    out = out.sort_values("n_shared", ascending=False).reset_index(drop=True)
    print(f"  {len(out):,} pairs kept (n_shared ≥ {MIN_SHARED}) "
          f"in {time.time()-t0:.1f}s")

    OUT_PQ.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(OUT_PQ, index=False, compression="zstd")
    size_mb = OUT_PQ.stat().st_size / 1024 / 1024
    print(f"Wrote {OUT_PQ}  ({size_mb:.1f} MB)")

    # The app offers this whole table as a TSV.gz download. Building it at
    # request time meant materialising all 517k rows and serialising them on
    # the first render of the page -- 83 MB of anon memory for a 6.8 MB file,
    # and the expander renders even when collapsed, so every cold start paid
    # it. Write it once here instead.
    import gzip
    t0 = time.time()
    with gzip.open(OUT_GZ, "wb", compresslevel=6) as fh:
        fh.write(out.to_csv(sep="\t", index=False).encode())
    print(f"Wrote {OUT_GZ}  ({OUT_GZ.stat().st_size / 1024 / 1024:.1f} MB, "
          f"{time.time()-t0:.1f}s)")
    print("\nHead:")
    print(out.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
