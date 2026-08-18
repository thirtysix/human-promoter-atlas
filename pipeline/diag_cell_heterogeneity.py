#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Are the unstable programs pooling artifacts?

ChIP-Atlas aggregates experiments across every assayed cell type, and
config.py's PEAK_USECOLS = [0,1,2,4] discards the cell_class column before any
analysis sees it. So when two TFs are both "assigned" to a module, that can mean
either

    they co-bind that element in the same cellular context, or
    one was assayed in liver and the other in breast and they have never been
    in the same nucleus.

The occupancy matrix cannot tell those apart. If a program is assembled from TFs
studied in disjoint contexts, its "co-binding" is an artifact of pooling, and
there is no reason for NMF to recover it consistently.

This predicts the observed failures specifically: AP-1 (P6: CEBPB/JUND/FOSL/FOS)
is stimulus-responsive and nuclear receptors (P3: FOXA1/AR/ESR1/PGR) are
tissue-restricted, whereas cohesin (P7) and ETS/RUNX (P5) bind consistently
across contexts -- and it is exactly P6 and P3 that fail while P7 and P5 are the
most reproducible.

Measured here as the loading-weighted mean pairwise cosine between the
cell-class composition vectors of a program's TFs. High = the program's factors
were assayed in the same contexts, so co-binding is credible. Low = the program
is stitched together across experiments that never co-occurred.

This is a cheaper and more direct test than re-scanning peaks per module: it
asks whether the TFs COULD have co-occurred, which is the actual question. It
does not require touching PEAK_USECOLS -- the production reader is untouched and
the cell_class column is read here only.

Usage:
    python pipeline/diag_cell_heterogeneity.py --k 18
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
import time
from collections import Counter
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from config import CHIP_ATLAS_DN, OUT_DN, K_CANONICAL

CELL_COL = 6                # 0-based; the 7th column of the q1e-5 per-TF BEDs
DEFAULT_WORKERS = 8
GZ_RATIO, BYTES_PER_ROW = 4.0, 50
TOP_TFS = 20                # TFs per program entering the pairwise comparison


def _log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def tf_cell_profile(args):
    """Cell-class composition of one TF's peaks (stride-sampled)."""
    path, target = args
    tf = path.name.replace(".bed.gz", "")
    try:
        est = max(1, int(path.stat().st_size * GZ_RATIO / BYTES_PER_ROW))
        stride = max(1, est // target)
        c = Counter()
        with gzip.open(path, "rt") as fh:
            for i, line in enumerate(fh):
                if i % stride:
                    continue
                f = line.rstrip("\n").split("\t")
                if len(f) > CELL_COL:
                    v = f[CELL_COL].strip()
                    if v:
                        c[v] += 1
        return tf, dict(c)
    except Exception as exc:
        return tf, {"__error__": str(exc)}


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="q1e-5")
    ap.add_argument("--k", type=int, default=None)
    ap.add_argument("--target-peaks", type=int, default=200000)
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    K = args.k or K_CANONICAL
    root = OUT_DN / "tss_modules"

    beds = sorted((CHIP_ATLAS_DN / "per_TF" / args.tier).glob("*.bed.gz"))
    _log(f"tier {args.tier}: {len(beds):,} TFs")
    t0 = time.time()
    profiles = {}
    with Pool(args.workers) as pool:
        jobs = [(b, args.target_peaks) for b in beds]
        for i, (tf, prof) in enumerate(pool.imap_unordered(tf_cell_profile, jobs,
                                                           chunksize=4), 1):
            if prof and "__error__" not in prof:
                profiles[tf] = prof
            if i % 300 == 0:
                _log(f"  {i}/{len(beds)}  ({time.time()-t0:.0f}s)")
    _log(f"cell profiles for {len(profiles):,} TFs in {(time.time()-t0)/60:.1f} min")

    classes = sorted({c for p in profiles.values() for c in p})
    cidx = {c: i for i, c in enumerate(classes)}
    _log(f"  {len(classes)} distinct cell classes: {', '.join(classes[:12])}"
         + (" ..." if len(classes) > 12 else ""))

    # L2-normalised composition vector per TF
    V = {}
    for tf, p in profiles.items():
        v = np.zeros(len(classes), dtype=np.float64)
        for c, n in p.items():
            v[cidx[c]] = n
        s = np.linalg.norm(v)
        if s > 0:
            V[tf] = v / s

    # per-TF diversity, for the record
    div = []
    for tf, p in profiles.items():
        n = np.array(list(p.values()), float)
        q = n / n.sum()
        div.append(dict(TF=tf, n_classes=len(p), n_sampled=int(n.sum()),
                        entropy=float(-(q * np.log(q + 1e-12)).sum())))
    div = pd.DataFrame(div)

    # ---- per-program coherence -------------------------------------------
    H = pd.read_csv(root / f"nmf.k{K}.H.tsv.gz", sep="\t", index_col=0)
    stab = pd.read_csv(root / f"nmf.k{K}.stability.tsv", sep="\t").sort_values("program")
    rows = []
    for p in H.index:
        load = H.loc[p].sort_values(ascending=False)
        tfs = [t for t in load.index[:TOP_TFS] if t in V]
        if len(tfs) < 3:
            rows.append(dict(program=int(str(p).replace("prog", "")),
                             coherence=np.nan, n_tf=len(tfs)))
            continue
        w = load[tfs].to_numpy(float)
        M = np.stack([V[t] for t in tfs])
        S = M @ M.T                                   # pairwise cosine
        iu = np.triu_indices(len(tfs), k=1)
        wp = np.outer(w, w)[iu]                       # loading-weighted mean
        rows.append(dict(program=int(str(p).replace("prog", "")),
                         coherence=float((S[iu] * wp).sum() / wp.sum()),
                         n_tf=len(tfs),
                         mean_classes=float(div.set_index("TF")
                                            .reindex(tfs)["n_classes"].mean())))
    coh = pd.DataFrame(rows)
    out_df = stab.merge(coh, on="program")

    out = args.out or (root / "k_selection" / f"cell_coherence.k{K}.tsv")
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    out_df.to_csv(out, sep="\t", index=False, float_format="%.6f")
    div.sort_values("entropy").to_csv(str(out).replace(".tsv", ".per_tf.tsv"),
                                      sep="\t", index=False, float_format="%.6f")

    print()
    _log(f"=== per-program: stability vs cell-class coherence (k={K}) ===")
    print(f"{'prog':>5}{'stability':>11}{'coherence':>11}{'meanCls':>9}  top TFs")
    for _, r in out_df.sort_values("coherence").iterrows():
        flag = "" if r["median_cosine"] >= 0.90 else "  <-- unstable"
        print(f"{int(r['program']):>5}{r['median_cosine']:>11.3f}{r['coherence']:>11.3f}"
              f"{r['mean_classes']:>9.1f}  {str(r['top_tfs'])[:38]}{flag}")
    d = out_df.dropna(subset=["coherence"])
    rho = d[["median_cosine", "coherence"]].corr(method="spearman").iloc[0, 1]
    pear = d[["median_cosine", "coherence"]].corr().iloc[0, 1]
    print(f"\n  Spearman(stability, coherence) = {rho:+.3f}   n={len(d)}")
    print(f"  Pearson                         = {pear:+.3f}")
    print("  Positive => programs whose TFs share cellular context are the ones")
    print("  that reproduce, i.e. the unstable ones are pooling artifacts.")
    print(f"\n  wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
