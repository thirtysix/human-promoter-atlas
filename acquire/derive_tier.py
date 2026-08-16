#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Derive a stricter ChIP-Atlas tier from a looser one already on disk.

ChIP-Atlas peak score is -10*log10(Q) capped at 1000, so a tier IS a score
threshold on the same quantity:

    q1e-50  ->  score >= 500
    q1e-20  ->  score >= 200
    q1e-10  ->  score >= 100
    q1e-5   ->  score >=  50

So q1e-20 is exactly the score>=200 subset of the q1e-5 files, and deriving it
costs one filtering pass instead of a multi-GB download.

Why derive rather than download
-------------------------------
The separately-downloaded tiers are NOT coordinate-identical. Measured on
GATA1/CTCF: identical score values appear in both q1e-5 and q1e-50, but the
q1e-5 intervals are ~1.4x wider, and exact-interval overlap is near zero. They
are the same peak calls with different interval processing (midpoints agree to
a median of 1 bp, and 95% within +-25 bp, which is why the pipeline -- KDE on
midpoints, sigma 25 bp -- is unaffected).

Deriving therefore buys two things a download cannot:

  * one processing run. No mixing of interval conventions across tiers.
  * the per-experiment columns. The q1e-5 files carry SRX accession and cell
    class per peak; the bulk tier files do not. That metadata is what makes
    split-half validation across DISJOINT experiments possible, so a bulk
    download would silently cost the ability to reproduce that analysis.

The result is a correct Q<threshold subset of a self-consistent dataset. It is
deliberately not expected to match a ChIP-Atlas bulk download byte for byte.

Usage:
    python acquire/derive_tier.py --from q1e-5 --to q1e-20
    python acquire/derive_tier.py --from q1e-5 --to q1e-20 --dry-run
"""

################################################################################
# Libraries ####################################################################
################################################################################
import argparse
import gzip
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
from config import CHIP_ATLAS_DN, TIER_TO_QVALUE          # noqa: E402

SCORE_COL = 4          # 0-based; ChIP-Atlas BED column 5
# Deliberately below the core count of a typical workstation: this is a short
# gzip-bound pass, and leaving headroom keeps it from saturating a laptop.
# Raise with --workers on a machine that can take it.
DEFAULT_WORKERS = 8


def tier_score(tier: str) -> int:
    """The tier expressed as a peak-score threshold."""
    import math
    return int(round(-10 * math.log10(TIER_TO_QVALUE[tier])))


def filter_one(job):
    """Copy one per-TF BED keeping rows at or above the score threshold."""
    src, dst, min_score = job
    n_in = n_out = 0
    tmp = dst.with_suffix(dst.suffix + ".part")
    try:
        with gzip.open(src, "rt") as fh, gzip.open(tmp, "wt", compresslevel=6) as out:
            for line in fh:
                n_in += 1
                f = line.split("\t")
                try:
                    if int(f[SCORE_COL]) >= min_score:
                        out.write(line)
                        n_out += 1
                except (IndexError, ValueError):
                    continue          # malformed row: skip, do not abort the TF
        # Rename only on success, so an interrupted run never leaves a
        # short file that looks complete to the next stage.
        tmp.replace(dst)
    except Exception as exc:
        tmp.unlink(missing_ok=True)
        return (src.name, n_in, n_out, f"FAILED: {exc}")
    return (src.name, n_in, n_out, None)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="src_tier", required=True)
    ap.add_argument("--to", dest="dst_tier", required=True)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--dest", help="write here instead of <chip_atlas>/per_TF/<to>; "
                                   "must still end in the tier name")
    ap.add_argument("--force", action="store_true",
                    help="replace a DOWNLOADED tier (destructive)")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    for t in (args.src_tier, args.dst_tier):
        if t not in TIER_TO_QVALUE:
            raise SystemExit(f"unknown tier {t!r}; known: {sorted(TIER_TO_QVALUE)}")
    s_score, d_score = tier_score(args.src_tier), tier_score(args.dst_tier)
    if d_score <= s_score:
        raise SystemExit(
            f"--to {args.dst_tier} (score>={d_score}) is not stricter than "
            f"--from {args.src_tier} (score>={s_score}). A looser tier cannot be "
            f"derived: the peaks it needs are not in the source file.")

    src_dn = CHIP_ATLAS_DN / "per_TF" / args.src_tier
    dst_dn = Path(args.dest) if args.dest else CHIP_ATLAS_DN / "per_TF" / args.dst_tier
    if dst_dn.name != args.dst_tier:
        raise SystemExit(
            f"--dest must end in the tier name: {dst_dn} does not end in "
            f"{args.dst_tier!r}. config.py enforces the same rule, and it is "
            f"what stops a build being labelled with the wrong tier.")
    beds = sorted(src_dn.glob("*.bed.gz"))
    if not beds:
        raise SystemExit(f"no *.bed.gz under {src_dn}")

    # Never silently replace a DOWNLOADED tier. q1e-50 holds the peaks the
    # published atlas was built from; overwriting it with a derived set would
    # destroy the only copy of a different processing run (the bulk files use
    # ~1.4x narrower intervals) and no downstream check would notice.
    existing = list(dst_dn.glob("*.bed.gz")) if dst_dn.is_dir() else []
    if existing and not (dst_dn / "_DERIVED.json").is_file() and not args.force:
        raise SystemExit(
            f"{dst_dn} already holds {len(existing):,} beds and has no "
            f"_DERIVED.json, so it was DOWNLOADED, not derived. Refusing to "
            f"overwrite it.\n"
            f"  To derive this tier without touching the download, write it "
            f"elsewhere and point the pipeline at it:\n"
            f"    --dest <somewhere>/per_TF_derived/{args.dst_tier}\n"
            f"    HPA_PER_TF_DIR=<somewhere>/per_TF_derived/{args.dst_tier}\n"
            f"  Use --force only if you intend to replace the download.")

    print(f"deriving {args.dst_tier} (score>={d_score}) from {args.src_tier} "
          f"(score>={s_score})")
    print(f"  source: {len(beds):,} beds, "
          f"{sum(b.stat().st_size for b in beds)/1e9:.1f} GB")
    print(f"  dest:   {dst_dn}")
    if args.dry_run:
        print("  [dry-run] nothing written")
        return 0

    dst_dn.mkdir(parents=True, exist_ok=True)
    jobs = [(b, dst_dn / b.name, d_score) for b in beds]
    t0 = time.time()
    rows, failures = [], []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(filter_one, jobs, chunksize=4), 1):
            rows.append(r)
            if r[3]:
                failures.append(r)
            if i % 200 == 0:
                kept = sum(x[2] for x in rows); seen = sum(x[1] for x in rows)
                print(f"  {i:>5}/{len(jobs)}  peaks {kept:,}/{seen:,} "
                      f"({kept/max(seen,1):.1%} kept)  {time.time()-t0:.0f}s",
                      flush=True)

    seen = sum(r[1] for r in rows); kept = sum(r[2] for r in rows)
    prov = {
        "derived_from_tier": args.src_tier,
        "tier": args.dst_tier,
        "min_score": d_score,
        "qvalue": TIER_TO_QVALUE[args.dst_tier],
        "n_files": len(rows),
        "n_peaks_in": seen,
        "n_peaks_kept": kept,
        "fraction_kept": round(kept / max(seen, 1), 6),
        "n_failed": len(failures),
        "note": ("Derived by score filter, not downloaded. Score is "
                 "-10*log10(Q), so this is exactly the Q<qvalue subset of the "
                 "source tier. Retains the source's per-experiment columns "
                 "(SRX, cell class). Not expected to match a ChIP-Atlas bulk "
                 "download of this tier byte for byte -- the bulk files use a "
                 "different interval convention (~1.4x wider), though midpoints "
                 "agree to a median of 1 bp."),
    }
    (dst_dn / "_DERIVED.json").write_text(json.dumps(prov, indent=2) + "\n")

    print(f"\n{len(rows):,} files, {seen:,} peaks in -> {kept:,} kept "
          f"({kept/max(seen,1):.1%}) in {(time.time()-t0)/60:.1f} min")
    print(f"  size: {sum(f.stat().st_size for f in dst_dn.glob('*.bed.gz'))/1e9:.2f} GB")
    if failures:
        print(f"  {len(failures)} FAILED:")
        for f in failures[:10]:
            print(f"    {f[0]}: {f[3]}")
        return 1
    print(f"  provenance: {dst_dn/'_DERIVED.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
