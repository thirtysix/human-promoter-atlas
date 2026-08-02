#!/usr/bin/env python3
"""
Split each per-TF peak BED into two halves by EXPERIMENT (SRX), producing two
peak trees that share no experiments.

This is the input to the split-half threshold calibration. Splitting by
experiment rather than by peak is the whole point: two halves of the same
experiment would replicate trivially, telling us nothing. Two disjoint sets of
experiments replicate an assignment only if the binding is real.

Layout is chosen so pipeline/config.py accepts each half unchanged -- it
requires the peak directory to be named after the tier, so each half gets its
own parent:

    <out>/A/per_TF/<tier>/<TF>.bed.gz
    <out>/B/per_TF/<tier>/<TF>.bed.gz

Assignment to halves is deterministic (hash of the SRX id), so the split is
reproducible and does not depend on file or listing order. Experiments are
split per TF, so every TF is halved rather than whole TFs landing on one side.

TFs with fewer than --min-srx experiments cannot be split meaningfully and are
written to neither half; they are listed in the report so they can be excluded
from the calibration rather than silently contributing zeros.

Usage:
    python acquire/split_by_srx.py --tier q1e-5 --out /path/to/split
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import multiprocessing as mp
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import config  # noqa: E402

SRX_COL = 5  # 0-based: chrom start end tf score SRX cell_group


def half_of(srx: str) -> int:
    """Deterministic, order-independent 50/50 assignment."""
    return hashlib.blake2b(srx.encode(), digest_size=8).digest()[0] & 1


def split_one(args) -> tuple:
    """Stream a BED into two halves. Returns (tf, n_srx, n_a, n_b, n, n_missing).

    Written streaming rather than by accumulating rows: CTCF alone is 65.6M rows
    (~5 GB of text), so buffering both halves in lists would need ~10 GB for one
    TF. Output goes to temporary names and is renamed only on success, so an
    interrupted run cannot leave a half-written BED that later looks complete.
    Cache SRX->half, since a TF has thousands of rows per experiment and the
    hash is otherwise recomputed for every one.
    """
    src, out_a, out_b, min_srx = args
    tf = src.name[: -len(".bed.gz")]
    tmp_a, tmp_b = Path(f"{out_a}.tmp"), Path(f"{out_b}.tmp")
    srxs, cache = set(), {}
    n = miss = na = nb = 0
    try:
        with gzip.open(src, "rt") as fh, \
             gzip.open(tmp_a, "wt", compresslevel=6) as fa, \
             gzip.open(tmp_b, "wt", compresslevel=6) as fb:
            for line in fh:
                n += 1
                f = line.split("\t")
                if len(f) <= SRX_COL:
                    miss += 1
                    continue
                srx = f[SRX_COL].strip()
                if not srx:
                    miss += 1
                    continue
                h = cache.get(srx)
                if h is None:
                    h = cache[srx] = half_of(srx)
                    srxs.add(srx)
                if h == 0:
                    fa.write(line); na += 1
                else:
                    fb.write(line); nb += 1
    except Exception:
        tmp_a.unlink(missing_ok=True); tmp_b.unlink(missing_ok=True)
        raise

    # A TF with too few experiments cannot be split meaningfully -- drop it
    # rather than let it contribute a degenerate 1-vs-1 comparison.
    if len(srxs) < min_srx:
        tmp_a.unlink(missing_ok=True); tmp_b.unlink(missing_ok=True)
        return tf, len(srxs), 0, 0, n, miss
    tmp_a.replace(out_a); tmp_b.replace(out_b)
    return tf, len(srxs), na, nb, n, miss


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=config.TIER)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--min-srx", type=int, default=4,
                    help="TFs with fewer experiments are skipped (default 4)")
    args = ap.parse_args()

    src_dn = config.CHIP_ATLAS_DN / "per_TF" / args.tier
    beds = sorted(src_dn.glob("*.bed.gz"))
    if len(beds) < 1000:
        raise SystemExit(f"only {len(beds)} beds under {src_dn} -- wrong tier?")

    a_dn = args.out / "A" / "per_TF" / args.tier
    b_dn = args.out / "B" / "per_TF" / args.tier
    a_dn.mkdir(parents=True, exist_ok=True)
    b_dn.mkdir(parents=True, exist_ok=True)

    # gzip compression dominates, so this parallelises almost linearly.
    # Largest first: CTCF is ~15x the median and would otherwise straggle.
    beds.sort(key=lambda p: -p.stat().st_size)
    jobs = [(b, a_dn / b.name, b_dn / b.name, args.min_srx) for b in beds]

    skipped, kept = [], []
    tot_a = tot_b = tot_n = tot_miss = 0
    with mp.Pool(processes=config.WORKERS) as pool:
        for i, (tf, n_srx, na, nb, n, miss) in enumerate(
                pool.imap_unordered(split_one, jobs, chunksize=1), 1):
            tot_a += na; tot_b += nb; tot_n += n; tot_miss += miss
            (kept if na or nb else skipped).append((tf, n_srx))
            if i % 200 == 0 or i == len(jobs):
                print(f"  {i}/{len(jobs)} ...", flush=True)

    report = args.out / f"split_report.{args.tier}.tsv"
    with open(report, "w") as fh:
        fh.write("tf\tn_srx\tsplit\n")
        for tf, n in kept:
            fh.write(f"{tf}\t{n}\tyes\n")
        for tf, n in skipped:
            fh.write(f"{tf}\t{n}\tno\n")

    print(f"\n  TFs split      {len(kept):>6}")
    print(f"  TFs skipped    {len(skipped):>6}  (< {args.min_srx} experiments)")
    print(f"  rows total     {tot_n:>12,}")
    print(f"  -> half A      {tot_a:>12,}")
    print(f"  -> half B      {tot_b:>12,}")
    print(f"  no SRX (dropped){tot_miss:>11,}  ({100*tot_miss/max(tot_n,1):.1f}%)")
    print(f"  report         {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
