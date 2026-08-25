"""Memory-mapped sidecars for wide numeric matrices.

Reading a [1793 x 2001] matrix out of parquet costs ~104 MB of resident
memory for 27.5 MB of data, and the overhead is not transient: releasing the
frame gives back a quarter of it. The decode allocates one buffer per column
chunk -- 2002 of them -- and the freed holes stay with the process. Measured:
+172 MB RSS for the two flavours the UI can reach, against 55 MB of actual
data. Arena tuning does not help (MALLOC_ARENA_MAX=2 measured slightly worse
than default), because the memory is not fragmentation -- it is simply never
handed back.

The same values in a plain `.npy`, memory-mapped, cost ~0 MB of anon memory.
The pages that do get touched are file-backed, which is the reclaimable kind:
under pressure the kernel can drop them, where anon memory on a swapless host
can only be OOM-killed.

This module owns BOTH ends of that format -- writing and verifying -- because
a positionally-aligned artifact is exactly the kind that fails silently. The
`.npy` carries no row keys: if the parquet is rebuilt with a different TF set
and the sidecar is stale, every row would take its neighbour's data and
nothing would raise. So `load()` demands the authoritative row labels from
its caller and refuses to return anything whose fingerprint disagrees.

The fingerprint is computed by `fingerprint()` on both sides, deliberately.
A fingerprint derived two ways is worse than none: it fails closed on correct
data and teaches everyone to ignore it.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np

# Bumped when the on-disk layout changes, so an old sidecar is rejected rather
# than misread.
FORMAT_VERSION = 1


def fingerprint(labels) -> str:
    """Order-sensitive digest of a matrix's row labels.

    Order-sensitive on purpose: the failure this guards against is a
    reordering that keeps the same membership, which a set hash would wave
    through. The NUL separator stops ("AB", "C") and ("A", "BC") colliding.
    """
    h = hashlib.sha256()
    for label in labels:
        h.update(str(label).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _meta_path(npy_path: Path) -> Path:
    return npy_path.with_suffix(".meta.json")


def write(npy_path: Path, values: np.ndarray, row_labels, col_labels) -> None:
    """Write `values` plus the metadata needed to verify it later.

    The metadata is written AFTER the array, so a crash between the two reads
    as "no sidecar" -- absent, which falls back to parquet -- rather than as a
    sidecar that claims to describe data it does not.
    """
    npy_path = Path(npy_path)
    npy_path.parent.mkdir(parents=True, exist_ok=True)
    values = np.ascontiguousarray(values)
    np.save(npy_path, values, allow_pickle=False)
    meta = {
        "format_version": FORMAT_VERSION,
        "shape": list(values.shape),
        "dtype": values.dtype.str,
        "row_fingerprint": fingerprint(row_labels),
        "rows": [str(r) for r in row_labels],
        "cols": [int(c) for c in col_labels],
    }
    _meta_path(npy_path).write_text(json.dumps(meta))


def load(npy_path: Path, expected_rows) -> Optional[tuple]:
    """`(values_memmap, rows, cols)`, or None if there is no usable sidecar.

    None is not an error — every caller has the source file to fall back to.
    That is what makes this safe to ship before the build step has run
    anywhere, and what makes a stale sidecar a slow path rather than a wrong
    answer.
    """
    npy_path = Path(npy_path)
    meta_path = _meta_path(npy_path)
    if not npy_path.exists() or not meta_path.exists():
        return None
    try:
        meta = json.loads(meta_path.read_text())
    except (OSError, ValueError):
        return None
    if meta.get("format_version") != FORMAT_VERSION:
        return None
    expected_rows = [str(r) for r in expected_rows]
    if meta.get("row_fingerprint") != fingerprint(expected_rows):
        return None
    if meta.get("rows") != expected_rows:          # belt and braces; cheap here
        return None
    try:
        values = np.load(npy_path, mmap_mode="r", allow_pickle=False)
    except (OSError, ValueError):
        return None
    if list(values.shape) != list(meta.get("shape", [])):
        return None
    if values.dtype.str != meta.get("dtype"):
        return None
    cols = np.asarray(meta.get("cols", []), dtype=np.int64)
    if values.shape[1] != cols.size or values.shape[0] != len(expected_rows):
        return None
    return values, np.asarray(expected_rows, dtype=object), cols
