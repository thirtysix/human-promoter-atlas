"""Memory-mapped sidecars for the TF×position aggregate matrices.

The app reads these matrices on its landing page, so their cost is paid by
every visitor. Out of parquet that is ~104 MB of resident memory per flavour
for 27.5 MB of data, and it does not come back when the frame is released.
As a memory-mapped `.npy` it is ~0 MB of anon memory, and the pages that are
touched are file-backed and therefore reclaimable.

Values are copied verbatim -- same dtype, same row order, no float32
downcast -- so the sidecar is bit-identical to the parquet it came from and
nothing downstream can shift. `app/lib/matrix_sidecar.py` owns the format and
is imported here rather than reimplemented, so the fingerprint the reader
checks is the one the writer wrote.

Safe to re-run, and safe to skip: `load_aggregate_matrix` falls back to the
parquet whenever a sidecar is missing or does not match it.

Run after data/build_app_db.py, whenever the aggregate parquets change:

    python data/build_aggregate_npy.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_DIR))

from app.lib import matrix_sidecar  # noqa: E402

AGG_DIR = REPO_DIR / "data" / "aggregate"
FLAVORS = ("binary", "score", "raw", "raw_score1000")


def main() -> None:
    if not AGG_DIR.is_dir():
        sys.exit(f"No aggregate directory at {AGG_DIR}. Run build_app_db.py first.")

    built = skipped = 0
    for flavor in FLAVORS:
        src = AGG_DIR / f"tf_x_position.{flavor}.parquet"
        if not src.exists():
            print(f"  {flavor:16s} no parquet — skipped")
            skipped += 1
            continue
        t0 = time.time()
        df = pd.read_parquet(src).set_index("TF")
        df.columns = df.columns.astype(int)
        out = AGG_DIR / f"tf_x_position.{flavor}.npy"
        matrix_sidecar.write(out, df.to_numpy(), df.index.to_numpy(),
                             df.columns.to_numpy())

        # Read it back through the same door the app uses. A sidecar that the
        # reader rejects is worse than none: the app would silently fall back
        # to parquet and this script would still have reported success.
        check = matrix_sidecar.load(out, [str(t) for t in df.index])
        if check is None:
            sys.exit(f"  {flavor}: wrote a sidecar the loader rejects — aborting")
        import numpy as np
        if not np.array_equal(np.asarray(check[0]), df.to_numpy()):
            sys.exit(f"  {flavor}: sidecar values differ from the parquet — aborting")

        size = out.stat().st_size / 1024 / 1024
        print(f"  {flavor:16s} {df.shape[0]}×{df.shape[1]}  {df.to_numpy().dtype}  "
              f"{size:6.1f} MB  {time.time()-t0:.1f}s  verified")
        built += 1
        del df

    print(f"\n{built} sidecar(s) written, {skipped} skipped.")


if __name__ == "__main__":
    main()
