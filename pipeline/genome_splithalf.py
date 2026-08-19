#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Split-half replication of genome-wide programs across disjoint experiments.

Seed stability asks whether independent STARTS converge on the same programs.
That is a solver property. This asks whether independent EXPERIMENTS do, which
is a property of the biology, and it is the stronger claim: a program built
from an artifact shared by every experiment will be perfectly seed-stable and
will still fail here.

Design
------
Elements are NOT rediscovered per half. Boundaries come from the existing
elements.tsv and are held fixed, so the two halves are two occupancy matrices
over the SAME rows. If each half rediscovered its own elements, program
mismatch would confound "different programs" with "different elements", and
the comparison would mean nothing.

ChIP-Atlas peaks carry an SRX accession (column 5 of the 7-column tier form).
config.PEAK_USECOLS keeps only 0,1,2,4, and genome_modules._read_tf likewise
takes only the score, so experiment identity is discarded before any analysis
sees it -- it is recovered here without touching either, exactly as
diag_cell_heterogeneity.py recovers cell_class.

Each SRX is assigned to half A or B by a deterministic hash of the accession,
so the split is reproducible without a metadata pass and is independent of file
order. A TF whose experiments all land on one side has an empty column on the
other; such TFs are EXCLUDED from the cosine comparison and counted, because a
zero column would otherwise drag every program's similarity toward zero and be
read as failure to replicate.

Matching is a global optimal assignment (Hungarian), not greedy nearest --
greedy lets two A-programs claim the same B-program and inflates the count.

Usage:
    python pipeline/genome_splithalf.py --genome-dir <dir> --stage all \\
        --ranks 60,90,110,125,150
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")

import argparse
import datetime as dt
import gzip
import hashlib
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.optimize import linear_sum_assignment

import nmf_fit
from config import (MIN_SCORE_ASSIGN, TIER, TF_SET, discover_tf_files,
                    write_analysis_readme)
import genome_modules as gm          # plain import: Pool must be able to pickle

REPLICATE_COSINE = 0.90              # same bar as the seed-stability criterion
SEEDS_PER_HALF = 3


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _srx_key(srx: str) -> bytes:
    """Deterministic sort key, so the split does not depend on file order."""
    return hashlib.md5(srx.encode()).digest()


def _split_within_tf(srxs) -> dict:
    """Assign one TF's OWN experiments alternately to halves A and B.

    Hashing each accession independently loses any TF whose experiments all
    land on one side: certain for a single-experiment TF, 50% for two, still
    12.5% for four. Measured on this build that discarded 497 of 1,793 TFs --
    and precisely the sparsely-assayed ones, whose programs are the least
    likely to replicate, so it biased replication UPWARD. Blocking the
    randomisation within TF keeps every TF with >=2 experiments on both sides.
    Order is by hash of the accession, not file order, so it stays reproducible.
    """
    ordered = sorted(srxs, key=_srx_key)
    return {srx: i & 1 for i, srx in enumerate(ordered)}


def _read_tf_half(args):
    """(tf_idx, packed_midpoints, scores, halves) for one TF."""
    tf_idx, paths, keep = args
    mids, scs, sidx = [], [], []
    seen = {}                     # SRX -> small int, interned per TF
    for p in paths:
        try:
            with gzip.open(p, "rt") as fh:
                for line in fh:
                    f = line.rstrip("\n").split("\t")
                    if len(f) < 6:
                        continue
                    c = f[0][3:] if f[0].startswith("chr") else f[0]
                    if c not in keep:
                        continue
                    mid = (int(f[1]) + int(f[2])) // 2
                    mid = (mid // (2 * gm.RECENTER_HALF + 1)) * (2 * gm.RECENTER_HALF + 1)
                    mids.append((keep[c] << 40) | mid)
                    scs.append(int(f[4]))
                    srx = f[5].strip()
                    j = seen.get(srx)
                    if j is None:
                        j = seen[srx] = len(seen)
                    sidx.append(j)
        except Exception:
            continue
    # Stratify WITHIN this TF, using only what this worker already read.
    assign = _split_within_tf(seen.keys())
    lut = np.empty(len(seen), np.int8)
    for srx, j in seen.items():
        lut[j] = assign[srx]
    hlf = lut[np.asarray(sidx, np.int64)] if sidx else []
    if not mids:
        return (tf_idx, np.empty(0, np.int64), np.empty(0, np.int16),
                np.empty(0, np.int8))
    return (tf_idx, np.asarray(mids, np.int64), np.asarray(scs, np.int16),
            np.asarray(hlf, np.int8))


def _assert_srx_column(paths):
    """Refuse to run on a tier where column 5 is not an SRX accession.

    config.PEAK_USECOLS keeps only 0,1,2,4 precisely because that is ALL the
    tiers agree on: q1e-5 is the 7-column stripped form (chrom start end
    antigen score SRX cell_class) but q1e-50 is BED9, whose column 5 is STRAND.
    Splitting a BED9 tier on column 5 would partition by +/- and report a
    confident replication figure for a partition with no experimental meaning.
    """
    import itertools
    for p in paths[:1]:
        with gzip.open(p, "rt") as fh:
            for line in itertools.islice(fh, 5):
                f = line.rstrip("\n").split("\t")
                if len(f) < 6 or not f[5].strip()[:3] in ("SRX", "ERX", "DRX"):
                    raise SystemExit(
                        f"{p}: column 5 is {f[5]!r}, not an SRX/ERX/DRX "
                        f"accession. This tier does not carry experiment "
                        f"identity in that column (q1e-50 is BED9, where it is "
                        f"strand). Split-half cannot run on it.")
    return True


def build(root: Path, workers: int):
    el = pd.read_csv(root / "elements.tsv", sep="\t", dtype={"chrom": str})
    chroms = list(dict.fromkeys(el.chrom))
    keep = {c: i for i, c in enumerate(chroms)}
    tf_files = discover_tf_files()
    tf_names = [s for s, _ in tf_files]
    _assert_srx_column(tf_files[0][1])
    _log(f"{len(el):,} elements over {len(chroms)} chroms, {len(tf_names)} TFs")

    jobs = [(i, paths, keep) for i, (_s, paths) in enumerate(tf_files)]
    mids, scs, tfi, hlf = [], [], [], []
    t0 = time.time()
    with Pool(workers) as pool:
        for n, (ti, m, s, h) in enumerate(
                pool.imap_unordered(_read_tf_half, jobs, chunksize=4), 1):
            if m.size:
                mids.append(m); scs.append(s); hlf.append(h)
                tfi.append(np.full(m.size, ti, np.int16))
            if n % 400 == 0:
                _log(f"  {n}/{len(jobs)} TFs ({time.time()-t0:.0f}s)")
    mid = np.concatenate(mids); sc = np.concatenate(scs)
    ti = np.concatenate(tfi); hf = np.concatenate(hlf)
    del mids, scs, tfi, hlf
    _log(f"  {len(mid):,} peaks with SRX in {(time.time()-t0)/60:.1f} min; "
         f"half A {int((hf==0).sum()):,} / half B {int((hf==1).sum()):,}")

    order = np.argsort(mid, kind="stable")
    mid, sc, ti, hf = mid[order], sc[order], ti[order], hf[order]

    rows = {0: ([], []), 1: ([], [])}
    for cname, sub in el.groupby("chrom", sort=False):
        ci = keep[cname]
        lo_pack = (ci << 40)
        c0 = np.searchsorted(mid, lo_pack, "left")
        c1 = np.searchsorted(mid, lo_pack | ((1 << 40) - 1), "right")
        pos = mid[c0:c1] - lo_pack
        s_, t_, h_ = sc[c0:c1], ti[c0:c1], hf[c0:c1]
        a = np.searchsorted(pos, sub.start.values, "left")
        b = np.searchsorted(pos, sub.end.values, "right")
        for eid, i0, i1 in zip(sub.element_id.values, a, b):
            if i1 <= i0:
                continue
            tt, vv, hh = t_[i0:i1], s_[i0:i1], h_[i0:i1]
            ok = vv >= MIN_SCORE_ASSIGN
            for half in (0, 1):
                asg = np.unique(tt[ok & (hh == half)])
                rows[half][0].extend([eid] * asg.size)
                rows[half][1].extend(asg.tolist())
        _log(f"  chr{cname}: assigned ({time.time()-t0:.0f}s)")

    eid_index = {e: i for i, e in enumerate(el.element_id.values)}
    out = {}
    for half, name in ((0, "A"), (1, "B")):
        r = np.array([eid_index[e] for e in rows[half][0]], np.int64)
        c = np.asarray(rows[half][1], np.int64)
        M = sp.coo_matrix((np.ones(len(r), np.float32), (r, c)),
                          shape=(len(el), len(tf_names))).tocsr()
        M.sum_duplicates()
        M.data[:] = 1.0
        sp.save_npz(str(root / f"occupancy.half_{name}.npz"), M)
        out[name] = M
        _log(f"  half {name}: {M.nnz:,} nonzeros, "
             f"{int((np.asarray(M.sum(0)).ravel() > 0).sum())} TFs present")
    pd.DataFrame({"TF": tf_names, "tf_idx": range(len(tf_names))}).to_csv(
        root / "tf_index.splithalf.tsv", sep="\t", index=False)
    return out["A"], out["B"], tf_names


def compare(root: Path, ranks, tf_names):
    A = sp.load_npz(str(root / "occupancy.half_A.npz")).tocsr()
    B = sp.load_npz(str(root / "occupancy.half_B.npz")).tocsr()
    inA = np.asarray(A.sum(0)).ravel() > 0
    inB = np.asarray(B.sum(0)).ravel() > 0
    shared = inA & inB
    _log(f"TFs present in both halves: {int(shared.sum()):,} of {len(tf_names):,} "
         f"({int((~shared).sum()):,} excluded -- all experiments on one side)")

    rows = []
    for k in ranks:
        Hs = {}
        for name, M in (("A", A), ("B", B)):
            cand = []
            for s in range(SEEDS_PER_HALF):
                _W, H, err, _r = nmf_fit.fit_nmf_stable(M, k, s,
                                                        max_iter=gm_max_iter())
                del _W
                cand.append(H)
            ref = medoid(cand)
            Hs[name] = cand[ref]
            _log(f"  k={k} half {name}: medoid of {SEEDS_PER_HALF} seeds")
        Ha = _unit_rows(Hs["A"][:, shared])
        Hb = _unit_rows(Hs["B"][:, shared])
        C = Ha @ Hb.T                                   # cosine, rows unit
        r_i, c_i = linear_sum_assignment(-C)            # optimal, not greedy
        best = C[r_i, c_i]
        rows.append(dict(k=k, replicated=int((best >= REPLICATE_COSINE).sum()),
                         frac=float((best >= REPLICATE_COSINE).mean()),
                         median_cosine=float(np.median(best)),
                         mean_cosine=float(best.mean())))
        _log(f"  k={k:>4}  replicated {rows[-1]['replicated']:>4}/{k}  "
             f"median cosine {rows[-1]['median_cosine']:.3f}")
    return pd.DataFrame(rows)


def gm_max_iter():
    from tss_modules_consensus import NMF_MAX_ITER
    return NMF_MAX_ITER


def _unit_rows(H):
    n = np.linalg.norm(H, axis=1, keepdims=True)
    return H / np.where(n > 0, n, 1.0)


def medoid(Hs):
    U = [_unit_rows(H) for H in Hs]
    n = len(U)
    S = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            C = U[i] @ U[j].T
            r, c = linear_sum_assignment(-C)
            S[i, j] = S[j, i] = C[r, c].mean()
    return int(np.argmax(S.sum(1)))


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--ranks", default="60,90,110,125,150")
    ap.add_argument("--stage", choices=["build", "compare", "all"], default="all")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    root = Path(args.genome_dir)
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]

    tf_names = [s for s, _ in discover_tf_files()]
    if args.stage in ("build", "all"):
        build(root, args.workers)
    if args.stage in ("compare", "all"):
        d = compare(root, ranks, tf_names)
        out = root / "k_selection"
        out.mkdir(parents=True, exist_ok=True)
        d.to_csv(out / "splithalf_replication.tsv", sep="\t", index=False,
                 float_format="%.6f")
        print()
        _log("=== split-half replication across disjoint SRX ===")
        print(d.to_string(index=False, float_format=lambda x: f"{x:,.3f}"))
        best = d.loc[d.replicated.idxmax()]
        print(f"\n  most replicated programs: k={int(best.k)} "
              f"({int(best.replicated)} of {int(best.k)})")
        write_analysis_readme(
            out / "splithalf",
            title="Split-half replication of genome-wide programs",
            rationale=(
                "Seed stability asks whether independent starts converge on the "
                "same programs -- a solver property. This asks whether "
                "independent EXPERIMENTS do, which is the stronger claim: a "
                "program built from an artifact common to every experiment is "
                "perfectly seed-stable and still fails here.\n\n"
                "Elements are held fixed from elements.tsv, so the halves are "
                "two matrices over the same rows; rediscovering elements per "
                "half would confound different programs with different "
                "elements. TFs whose experiments all fall on one side are "
                "excluded and counted, since a zero column would drag every "
                "similarity toward zero and read as failure to replicate. "
                "Programs are matched by global optimal assignment, not greedy "
                "nearest, which would let two programs claim the same partner."),
            params={"replicate_cosine": REPLICATE_COSINE,
                    "seeds_per_half": SEEDS_PER_HALF, "ranks": str(ranks),
                    "min_score_assign": MIN_SCORE_ASSIGN},
            inputs={"genome_dir": str(root), "tier": TIER, "tf_set": TF_SET},
            stats={f"k={int(r.k)}": f"{int(r.replicated)}/{int(r.k)} replicated, "
                                    f"median cosine {r.median_cosine:.3f}"
                   for _, r in d.iterrows()})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
