#!/usr/bin/env python3
"""
Compare two atlas builds and say whether the difference is signal or dilution.

Designed for the A/B/C ladder, where two things change and must not be
confounded:

    A  q1e-50 / whitelist   baseline
    B  q1e-5  / whitelist   A->B isolates the THRESHOLD effect
    C  q1e-5  / all         B->C isolates the TF-AXIS effect

Deliberate methodological choices
---------------------------------
* Program agreement is measured on the **gene axis**, not the module axis.
  Modules are re-discovered per build, so module ids are not comparable at all
  and their count is expected to move; the ~19.7k TSS axis is stable, so
  "dominant program per TSS" is the only honest basis for an ARI.
* Program identity is matched by **cosine similarity of TF loadings** and a
  greedy assignment, restricted to TFs the two builds share. NMF program numbers
  are arbitrary, so comparing prog3 to prog3 across builds is meaningless.
* Dilution is reported as several independent quantities rather than one score,
  because "more peaks" legitimately increases some of them.

Usage:
    python pipeline/compare_builds.py A_DIR B_DIR [--k 12] [--label-a A] [--label-b B]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def _load(build: Path, k: int) -> dict:
    mod_dn = build / "tss_modules"
    out = {"root": build, "stamp": None}
    stamp = build / "_BUILD.json"
    if stamp.is_file():
        out["stamp"] = json.loads(stamp.read_text())
    out["modules"] = pd.read_csv(mod_dn / "modules.tsv", sep="\t")
    h = mod_dn / f"nmf.k{k}.H.tsv.gz"
    out["H"] = pd.read_csv(h, sep="\t", index_col=0) if h.is_file() else None
    g = mod_dn / f"nmf.k{k}.gene_configurations.tsv"
    out["genes"] = pd.read_csv(g, sep="\t") if g.is_file() else None
    t = mod_dn / f"nmf.k{k}.top_tfs.tsv"
    out["top"] = pd.read_csv(t, sep="\t") if t.is_file() else None
    agg = build / "matrices" / "tf_x_position.binary.parquet"
    out["agg"] = pd.read_parquet(agg) if agg.is_file() else None
    return out


def anchors(d: dict) -> dict:
    """Positional sanity, the pass/fail gate. A build whose CTCF or TBP peak has
    moved is wrong, not merely different."""
    m = d["agg"]
    if m is None:
        return {}
    pos = np.array([float(c) for c in m.columns[1:]])
    out = {}
    for tf in ("CTCF", "TBP"):
        row = m[m["TF"] == tf]
        if len(row):
            out[tf] = float(pos[np.argmax(row.iloc[0, 1:].to_numpy(float))])
    return out


def dominant_program(d: dict) -> pd.Series:
    """First entry of program_path = the program of the module nearest the TSS."""
    g = d["genes"]
    if g is None:
        return pd.Series(dtype=object)
    dom = g["program_path"].astype(str).str.split(",").str[0]
    return pd.Series(dom.values, index=g["transcript_id"].values)


def adjusted_rand(a: np.ndarray, b: np.ndarray) -> float:
    """ARI without importing sklearn (keeps this runnable anywhere)."""
    ua, ia = np.unique(a, return_inverse=True)
    ub, ib = np.unique(b, return_inverse=True)
    n = len(a)
    cont = np.zeros((len(ua), len(ub)), dtype=np.int64)
    np.add.at(cont, (ia, ib), 1)
    comb = lambda x: (x * (x - 1) // 2).sum()
    sum_ij = comb(cont)
    sum_i = comb(cont.sum(1))
    sum_j = comb(cont.sum(0))
    tot = n * (n - 1) // 2
    exp = sum_i * sum_j / tot if tot else 0.0
    mx = 0.5 * (sum_i + sum_j)
    return float((sum_ij - exp) / (mx - exp)) if mx != exp else 1.0


def match_programs(ha: pd.DataFrame, hb: pd.DataFrame):
    """Greedy cosine matching of programs across builds on shared TFs."""
    shared = [c for c in ha.columns if c in hb.columns]
    A = ha[shared].to_numpy(float)
    B = hb[shared].to_numpy(float)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    S = A @ B.T
    pairs, used = [], set()
    for i in np.argsort(-S.max(axis=1)):
        j = int(np.argmax(np.where([c not in used for c in range(S.shape[1])],
                                   S[i], -np.inf)))
        if j in used:
            continue
        used.add(j)
        pairs.append((ha.index[i], hb.index[j], float(S[i, j])))
    return pairs, len(shared)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("a"); ap.add_argument("b")
    ap.add_argument("--k", type=int, default=12)
    ap.add_argument("--label-a", default="A"); ap.add_argument("--label-b", default="B")
    args = ap.parse_args()

    A = _load(Path(args.a), args.k); B = _load(Path(args.b), args.k)
    la, lb = args.label_a, args.label_b

    def hdr(t): print(f"\n{t}\n" + "-" * len(t))

    hdr(f"builds (k={args.k})")
    for lab, d in ((la, A), (lb, B)):
        s = d["stamp"] or {}
        print(f"  {lab:<4} {d['root']}")
        print(f"       tier={s.get('tier','?')} tf_set={s.get('tf_set','?')} n_tf={s.get('n_tf','?')}")

    # 1. positional anchors -- pass/fail
    hdr("1. positional anchors (must not move; failure = wrong, not different)")
    aa, ab = anchors(A), anchors(B)
    for tf in sorted(set(aa) | set(ab)):
        va, vb = aa.get(tf), ab.get(tf)
        delta = f"{vb-va:+.0f}" if (va is not None and vb is not None) else "?"
        flag = "" if (va is None or vb is None or abs(vb - va) <= 25) else "   <-- MOVED >25bp"
        print(f"  {tf:<6} {la}={va:+.0f}  {lb}={vb:+.0f}  delta={delta} bp{flag}")

    # 2. TF axis
    hdr("2. TF axis")
    if A["H"] is not None and B["H"] is not None:
        sa, sb = set(A["H"].columns), set(B["H"].columns)
        print(f"  {la}: {len(sa)}   {lb}: {len(sb)}   shared: {len(sa & sb)}")
        only_b = sorted(sb - sa)
        if only_b:
            print(f"  only in {lb} ({len(only_b)}): {', '.join(only_b[:12])}"
                  + (" ..." if len(only_b) > 12 else ""))

    # 3. module-level dilution
    hdr("3. dilution metrics (interpret together; more peaks raises some legitimately)")
    ma, mb = A["modules"], B["modules"]
    rows = [
        ("modules", len(ma), len(mb)),
        ("TSSs with >=1 module", ma["tss_id"].nunique(), mb["tss_id"].nunique()),
        ("median module width bp", ma["width"].median(), mb["width"].median()),
        ("median TFs assigned", ma["n_tfs_assigned"].median(), mb["n_tfs_assigned"].median()),
        ("median peaks in module", ma["n_peaks_in"].median(), mb["n_peaks_in"].median()),
        ("modules per TSS (mean)", len(ma) / ma["tss_id"].nunique(),
                                   len(mb) / mb["tss_id"].nunique()),
    ]
    print(f"  {'metric':<26}{la:>12}{lb:>12}{'ratio':>9}")
    for name, x, y in rows:
        r = (y / x) if x else float("nan")
        print(f"  {name:<26}{x:>12,.1f}{y:>12,.1f}{r:>9.2f}")
    # A whole promoter window collapsing into one module is the clearest
    # single sign that the looser threshold has smeared structure away.
    for lab, m in ((la, ma), (lb, mb)):
        one = (m.groupby("tss_id").size() == 1).mean()
        wide = (m["width"] > 1000).mean()
        print(f"  {lab}: single-module TSSs {one:6.1%}   modules wider than 1kb {wide:6.1%}")

    # 4. program agreement on the stable gene axis
    hdr("4. program agreement (ARI on dominant program per TSS -- gene axis)")
    da, db = dominant_program(A), dominant_program(B)
    common = da.index.intersection(db.index)
    if len(common):
        ari = adjusted_rand(da.loc[common].to_numpy(), db.loc[common].to_numpy())
        print(f"  transcripts compared: {len(common):,}")
        print(f"  ARI = {ari:.3f}   (1.0 identical partition, 0.0 chance)")
    else:
        print("  no shared transcripts")

    # 5. program identity
    hdr("5. program identity (cosine on shared-TF loadings, greedy match)")
    if A["H"] is not None and B["H"] is not None:
        pairs, nshared = match_programs(A["H"], B["H"])
        print(f"  matched on {nshared} shared TFs")
        sims = [s for _, _, s in pairs]
        for pa, pb, s in sorted(pairs, key=lambda t: -t[2]):
            bar = "#" * int(round(s * 20))
            print(f"    {pa:>7} -> {pb:<7} cos={s:.3f} {bar}")
        print(f"  median cosine {np.median(sims):.3f}, min {min(sims):.3f}")
        print("  (near-1.0 across the board = same biology at higher sensitivity;")
        print("   a low tail = programs merged or fragmented)")

    # 6. top-TF overlap per matched program
    if A["top"] is not None and B["top"] is not None and A["H"] is not None:
        hdr("6. top-TF Jaccard per matched program (N=20)")
        ta = A["top"]; tb = B["top"]
        def topset(t, prog, n=20):
            p = int(str(prog).replace("prog", ""))
            return set(t[t["program"] == p].nsmallest(n, "rank")["tf"])
        js = []
        for pa, pb, _ in pairs:
            sa, sb = topset(ta, pa), topset(tb, pb)
            j = len(sa & sb) / len(sa | sb) if (sa | sb) else float("nan")
            js.append(j)
            print(f"    {pa:>7} -> {pb:<7} J={j:.2f}  shared={sorted(sa & sb)[:6]}")
        print(f"  median Jaccard {np.nanmedian(js):.2f}"
              f"   worst {np.nanmin(js):.2f}  <-- watch the worst, not the mean")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
