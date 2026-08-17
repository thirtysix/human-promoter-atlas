#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Rank selection by external complex recovery.

Asks whether a rank-k factorization keeps known protein complexes together.
Cohesin subunits (RAD21/SMC1A/SMC3/STAG1/STAG2) either land in one program or
they do not, and GO Cellular Component knows the answer independently of any
ChIP data -- so unlike GO-BP program enrichment, this cannot be gamed by
assigning more TFs to more modules.

Scoring: Adjusted Mutual Information between two partitions of the same TFs,
one by complex membership and one by dominant program. AMI is used rather than
raw MI or a co-membership rate for one specific reason: it is corrected for
chance AND comparable across partitions with DIFFERENT NUMBERS OF CLUSTERS
(Vinh, Epps & Bailey 2010). Every earlier rank statistic in this project failed
exactly there -- reproducible-fraction has k in its denominator and favours
small k, reproducible-count grows with k mechanically. An uncorrected
co-membership score is degenerate at k=1, where every complex is trivially
"together".

Complex labels: each TF is assigned the SMALLEST GO-CC set containing it, so
nested terms (SWI/SNF superfamily vs its sub-complexes) give one specific label
per TF rather than several overlapping ones.

Seeds are averaged because a single fit's program assignment is one draw; the
spread across seeds is reported so a rank whose AMI is unstable is visible
rather than hidden behind a mean.

Usage:
    python pipeline/nmf_complex_recovery.py --ranks 5,8,10,12,15,18,20,25,30 --seeds 5
"""

################################################################################
# Libraries ####################################################################
################################################################################
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "8")
os.environ.setdefault("OMP_NUM_THREADS", "8")
os.environ.setdefault("MKL_NUM_THREADS", "8")

import argparse
import datetime as dt
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.metrics import adjusted_mutual_info_score

import nmf_fit
from config import OUT_DN, TIER, TF_SET, MIN_SCORE_ASSIGN, MSIGDB_FN

MIN_MEMBERS = 3          # complexes with fewer present TFs carry no signal


def _log(msg):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def complex_labels(tf_names, cc_path: Path):
    """TF -> smallest containing GO-CC set. Returns (labels, index, n_sets)."""
    cc = json.loads(cc_path.read_text())
    present = set(tf_names)
    sets = {}
    for name, rec in cc.items():
        members = set(rec.get("geneSymbols", [])) & present
        if len(members) >= MIN_MEMBERS:
            sets[name] = members
    # smallest-first so the most specific complex wins the assignment
    lab = {}
    for name in sorted(sets, key=lambda n: len(sets[n])):
        for tf in sets[name]:
            lab.setdefault(tf, name)
    idx = [i for i, t in enumerate(tf_names) if t in lab]
    labels = np.array([lab[tf_names[i]] for i in idx])
    return labels, np.array(idx), len(sets)


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", default="5,8,10,12,15,18,20,25,30")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--cc", default=None, help="path to c5.go.cc json")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ranks = [int(r) for r in args.ranks.split(",") if r.strip()]

    root = OUT_DN / "tss_modules"
    M = sp.load_npz(str(root / "occupancy.modules.npz")).tocsr()
    tf_names = (pd.read_csv(root / "tf_index.tsv", sep="\t")
                  .sort_values("tf_idx")["TF"].tolist())
    cc_path = Path(args.cc) if args.cc else Path(str(MSIGDB_FN)).parent / "c5.go.cc.v2026.1.Hs.json"
    labels, idx, n_sets = complex_labels(tf_names, cc_path)

    _log(f"build {OUT_DN.name} (tier={TIER} tf_set={TF_SET} score={MIN_SCORE_ASSIGN})")
    _log(f"  occupancy {M.shape}")
    _log(f"  GO-CC sets with >={MIN_MEMBERS} present TFs: {n_sets}")
    _log(f"  TFs with a complex label: {len(idx)} / {len(tf_names)} "
         f"across {len(set(labels))} distinct complexes")

    rows = []
    t_all = time.time()
    for k in ranks:
        amis, retries_tot = [], 0
        t0 = time.time()
        for s in range(args.seeds):
            _W, H, _err, retries = nmf_fit.fit_nmf_stable(M, k, s, max_iter=300)
            retries_tot += retries
            # dominant program per TF, on the complex-labelled TFs only
            prog = H[:, idx].argmax(axis=0)
            amis.append(adjusted_mutual_info_score(labels, prog))
            del _W, H
        rows.append(dict(k=k, ami_mean=float(np.mean(amis)),
                         ami_sd=float(np.std(amis, ddof=1)) if len(amis) > 1 else 0.0,
                         ami_max=float(np.max(amis)), collapses=retries_tot,
                         secs=round(time.time() - t0, 1)))
        _log(f"  k={k:>3}  AMI={np.mean(amis):.4f} +/- "
             f"{(np.std(amis, ddof=1) if len(amis)>1 else 0):.4f}  "
             f"collapses={retries_tot}  ({time.time()-t0:.0f}s)")

    d = pd.DataFrame(rows)
    out = args.out or (root / "k_selection" / "complex_recovery.tsv")
    os.makedirs(os.path.dirname(str(out)), exist_ok=True)
    d.to_csv(out, sep="\t", index=False, float_format="%.6f")

    print()
    _log(f"=== complex recovery (AMI vs GO-CC), {args.seeds} seeds ===")
    print(f"{'k':>5}{'AMI':>10}{'sd':>9}{'collapses':>11}")
    best = int(d.loc[d.ami_mean.idxmax(), "k"])
    for _, r in d.iterrows():
        print(f"{int(r.k):>5}{r.ami_mean:>10.4f}{r.ami_sd:>9.4f}{int(r.collapses):>11}"
              + ("  <-- best" if r.k == best else ""))
    print(f"\n  best k by complex recovery: {best}")
    print(f"  wrote {out}")
    _log(f"total {(time.time()-t_all)/60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
