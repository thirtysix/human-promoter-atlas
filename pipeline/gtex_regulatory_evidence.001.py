#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Tissue-coherence evidence for the regulatory hypothesis at each module.

Produces four outputs that lift the per-transcript view from "TFs are bound
here" to "the bound TFs co-vary with the target gene across tissues":

    1. module_target_correlation.parquet
       For each module: Pearson r between mean TF TPM (across the module's
       assigned TFs) and the target transcript's TPM, computed across the
       66 GTEx tissues.

    2. module_tf_target_correlation.parquet
       For each (module, TF-in-module): per-TF r against the target
       transcript. Reveals the *driver* TFs within a multi-TF module — some
       may have r=0.8 while others are flat.

    3. program_tissue_specificity.tsv
       Per-program tau (Yanai 2005) over module_tissue_activity averaged per
       program, plus top tissue and mean activity in top tissue.

    4. module_supporting_tissues.parquet
       For each (module, tissue): a 'supporting' flag = True iff both the
       module's mean TF TPM AND the target transcript's TPM are in the top
       quartile across the 66 tissues. Plus a per-module summary
       (n_supporting, top_supporting_tissue).
"""

################################################################################
# Libraries ####################################################################
################################################################################
import sys
import time
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd

# Machine-specific paths and build axes -> pipeline/config.py
from config import OUT_DN

sys.stdout.reconfigure(line_buffering=True)


################################################################################
# Paths ########################################################################
################################################################################
ROOT                = OUT_DN   # config.OUT_DN
GTEX_DN = ROOT / "tss_gtex"

TX_MEAN_FN = GTEX_DN / "transcript_tissue_mean.parquet"
TF_EXPR_FN = GTEX_DN / "tf_tissue_expression.parquet"
ACT_FN     = GTEX_DN / "module_tissue_activity.parquet"
MODULES_FN = ROOT / "tss_modules" / "modules.tsv"
MP_FN      = ROOT / "tss_modules" / "nmf.k10.module_program.tsv"
PEAKS_FN   = ROOT / "tss_modules" / "peaks.parquet"
TF_INDEX_FN = ROOT / "tss_modules" / "tf_index.tsv"

DECIMALS    = 2
HIGH_R      = 0.5     # |r| threshold for "high-r" TF count
TOP_QUARTILE = 0.75   # quantile threshold for supporting-tissue logic
MIN_SAMPLES_FOR_TAU = 40  # only include tissues with >= this many samples in
                           # the tau / top-tissue calculation (low-N tissues
                           # have noisy mean TPM that distorts specificity)


################################################################################
# Helpers ######################################################################
################################################################################
def _ts() -> str:
    return dt.datetime.now().strftime("%H:%M:%S")


def _log(msg: str):
    print(f"[{_ts()}] {msg}")


def pearson_rows(M: np.ndarray, V: np.ndarray) -> np.ndarray:
    """Vectorized Pearson r between every row of M (n × c) and 1-D V (c,)."""
    M = M.astype(np.float64, copy=True)
    V = V.astype(np.float64, copy=False) - V.mean()
    M -= M.mean(axis=1, keepdims=True)
    num = (M * V).sum(axis=1)
    den = np.sqrt((M * M).sum(axis=1) * (V * V).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def tau_specificity(x: np.ndarray) -> float:
    """Yanai et al. 2005 tissue specificity index. x = expression vector
    across tissues (non-negative). Returns tau in [0, 1] where 0 = uniform,
    1 = single-tissue."""
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    if x.size == 0 or x.max() <= 0:
        return float("nan")
    n = x.size
    return float(((1.0 - x / x.max())).sum() / (n - 1))


################################################################################
# Execution ####################################################################
################################################################################
def main():
    GTEX_DN.mkdir(parents=True, exist_ok=True)

    _log("loading inputs")
    tx_mean = pd.read_parquet(TX_MEAN_FN).astype(np.float32)
    tf_expr = pd.read_parquet(TF_EXPR_FN).astype(np.float32)
    activity = pd.read_parquet(ACT_FN)  # module_id, tissue, mean_tf_tpm, n_tfs_in_module
    modules = pd.read_csv(MODULES_FN, sep="\t",
                           usecols=["module_id", "tss_id", "transcript_id",
                                    "lo_offset", "hi_offset"])
    mp = pd.read_csv(MP_FN, sep="\t",
                      usecols=["module_id", "dominant_program"])
    tf_idx = pd.read_csv(TF_INDEX_FN, sep="\t")
    tf_idx_to_name = dict(zip(tf_idx["tf_idx"], tf_idx["TF"]))

    _log(f"  tx_mean {tx_mean.shape}, tf_expr {tf_expr.shape}, "
         f"activity {len(activity):,} rows, modules {len(modules):,}")

    # Pivot module_tissue_activity -> wide (module × tissue mean TF TPM) so we
    # can correlate row-by-row against target tissue means.
    _log("pivoting activity to wide module × tissue")
    mt_wide = (activity.pivot_table(index="module_id", columns="tissue",
                                      values="mean_tf_tpm", aggfunc="first"))
    # Align to tx_mean's tissue order
    mt_wide = mt_wide[tx_mean.columns]
    mt_arr = mt_wide.to_numpy(np.float32)
    mt_arr = np.where(np.isnan(mt_arr),
                       np.nanmean(mt_arr, axis=1, keepdims=True), mt_arr)
    _log(f"  wide module × tissue: {mt_wide.shape}")

    # Target tx_mean lookup as numpy
    tx_arr = tx_mean.to_numpy(np.float32)
    tx_index = {tx: i for i, tx in enumerate(tx_mean.index)}

    # ------------------------------------------------------------------------
    # 1) module_target_correlation: r between module's TF-mean profile and
    #    its target transcript's TPM profile, both across the 66 tissues.
    # ------------------------------------------------------------------------
    _log("computing module ↔ target correlation")
    module_to_tx = dict(zip(modules["module_id"], modules["transcript_id"]))
    module_ids = mt_wide.index.values
    rows = []
    t0 = time.time()
    missing = 0
    for i, mid in enumerate(module_ids):
        tx = module_to_tx.get(int(mid))
        ti = tx_index.get(tx) if tx else None
        if ti is None:
            missing += 1
            continue
        target = tx_arr[ti]
        target_filled = np.where(np.isnan(target),
                                  np.nanmean(target), target)
        m_vec = mt_arr[i]
        # 1×c row: vectorize via pearson_rows on a (1, c) matrix
        r = pearson_rows(m_vec.reshape(1, -1), target_filled)[0]
        rows.append({"module_id": int(mid),
                     "r_module_target": float(r) if np.isfinite(r) else float("nan")})
    _log(f"  computed {len(rows):,} module r values; "
         f"{missing} target-not-in-tx_mean ({time.time()-t0:.1f}s)")

    mod_target = pd.DataFrame(rows)
    mod_target["r_module_target"] = (mod_target["r_module_target"]
                                       .astype(np.float32).round(2))

    # ------------------------------------------------------------------------
    # 2) module_tf_target_correlation: per-TF-in-module r against target.
    #    Need to recover (module, TF) assignment from peaks (score≥500 within
    #    [lo, hi]). Reuse the same trick as gtex_correlate.
    # ------------------------------------------------------------------------
    _log("recovering (module, TF) assignments from peaks…")
    peaks = pd.read_parquet(PEAKS_FN)
    peaks["tf"] = peaks["tf_idx"].map(tf_idx_to_name)
    peaks500 = peaks[peaks["score"] >= 500]
    peaks_by_tss = peaks500.groupby("tss_id")

    _log("computing per-(module, TF) correlations")
    pair_rows = []
    summary_rows = []
    tf_expr_arr = tf_expr.to_numpy(np.float32)
    tf_expr_arr = np.where(np.isnan(tf_expr_arr),
                            np.nanmean(tf_expr_arr, axis=1, keepdims=True),
                            tf_expr_arr)
    tf_to_idx = {t: i for i, t in enumerate(tf_expr.index)}

    t0 = time.time()
    for i, m in modules.iterrows():
        try:
            sub = peaks_by_tss.get_group(int(m["tss_id"]))
        except KeyError:
            continue
        in_mod = sub[(sub["local"] >= int(m["lo_offset"])) &
                      (sub["local"] <= int(m["hi_offset"]))]
        tfs = in_mod["tf"].dropna().unique()
        if len(tfs) == 0:
            continue
        tx = m["transcript_id"]
        ti = tx_index.get(tx)
        if ti is None:
            continue
        target = tx_arr[ti]
        target_filled = np.where(np.isnan(target),
                                  np.nanmean(target), target)
        present = [t for t in tfs if t in tf_to_idx]
        if not present:
            continue
        rows_idx = [tf_to_idx[t] for t in present]
        block = tf_expr_arr[rows_idx]
        rs = pearson_rows(block, target_filled)
        # Per-TF rows
        for tf_name, r in zip(present, rs):
            if not np.isfinite(r):
                continue
            pair_rows.append({"module_id": int(m["module_id"]),
                              "tf": tf_name,
                              "r": float(np.round(r, 2))})
        # Per-module driver summary
        finite = np.isfinite(rs)
        if finite.any():
            best = np.nanargmax(np.abs(rs))
            summary_rows.append({
                "module_id": int(m["module_id"]),
                "top_driver_tf": present[best],
                "top_driver_r":  float(np.round(rs[best], 2)),
                "n_tfs_high_r":  int((np.abs(rs[finite]) >= HIGH_R).sum()),
            })
        if (i + 1) % 5000 == 0:
            _log(f"    {i+1:,}/{len(modules):,} modules  "
                 f"({time.time()-t0:.1f}s)")

    mod_tf = pd.DataFrame(pair_rows)
    mod_tf["r"] = mod_tf["r"].astype(np.float16)
    mod_tf.to_parquet(GTEX_DN / "module_tf_target_correlation.parquet",
                      index=False)
    _log(f"  module_tf_target_correlation.parquet: {len(mod_tf):,} pairs  "
         f"({(GTEX_DN / 'module_tf_target_correlation.parquet').stat().st_size/1e6:.1f} MB)")

    drivers = pd.DataFrame(summary_rows)
    mod_target = mod_target.merge(drivers, on="module_id", how="left")

    # ------------------------------------------------------------------------
    # 4) module_supporting_tissues: per-(module, tissue), True if both module
    #    TF mean and target are in top quartile across 66 tissues.
    # ------------------------------------------------------------------------
    _log("computing supporting-tissue flags")
    # Quartile thresholds per row (per module / per target)
    mt_q = np.nanquantile(mt_arr, TOP_QUARTILE, axis=1, keepdims=True)
    # For target, we need the matching tx row per module
    target_q_per_module = np.full(len(module_ids), np.nan, dtype=np.float32)
    target_arr_per_module = np.full((len(module_ids), tx_arr.shape[1]),
                                     np.nan, dtype=np.float32)
    for i, mid in enumerate(module_ids):
        tx = module_to_tx.get(int(mid))
        ti = tx_index.get(tx) if tx else None
        if ti is None:
            continue
        v = tx_arr[ti]
        target_q_per_module[i] = np.nanquantile(v, TOP_QUARTILE)
        target_arr_per_module[i] = v
    target_q_per_module = target_q_per_module.reshape(-1, 1)
    target_high = target_arr_per_module >= target_q_per_module
    module_high = mt_arr >= mt_q
    supporting = module_high & target_high   # bool [n_modules x 66]
    n_supporting = supporting.sum(axis=1)

    # Top supporting tissue per module = tissue with the largest product of
    # (module TF mean rank) × (target rank) within the supporting set.
    tissues = list(tx_mean.columns)
    top_support_tissue = []
    for i in range(len(module_ids)):
        mask = supporting[i]
        if not mask.any():
            top_support_tissue.append("")
            continue
        score = mt_arr[i] * target_arr_per_module[i]
        score = np.where(mask, score, -np.inf)
        top_support_tissue.append(tissues[int(np.nanargmax(score))])

    # Long-format supporting-tissues parquet
    rows = []
    for i, mid in enumerate(module_ids):
        for t_idx, tissue in enumerate(tissues):
            if supporting[i, t_idx]:
                rows.append({"module_id": int(mid), "tissue": tissue,
                             "module_tf_tpm": float(mt_arr[i, t_idx]),
                             "target_tpm":    float(target_arr_per_module[i, t_idx])})
    supp_long = pd.DataFrame(rows)
    if not supp_long.empty:
        supp_long["module_tf_tpm"] = supp_long["module_tf_tpm"].astype(np.float32).round(DECIMALS)
        supp_long["target_tpm"]    = supp_long["target_tpm"].astype(np.float32).round(DECIMALS)
    supp_long.to_parquet(GTEX_DN / "module_supporting_tissues.parquet",
                          index=False)
    _log(f"  module_supporting_tissues.parquet: {len(supp_long):,} rows")

    # Add per-module supporting-tissue summary cols to mod_target
    mod_target["n_supporting_tissues"] = pd.Series(
        n_supporting, index=module_ids).astype(np.int32).reindex(
        mod_target["module_id"]).values
    mod_target["top_supporting_tissue"] = pd.Series(
        top_support_tissue, index=module_ids).reindex(
        mod_target["module_id"]).values

    mod_target.to_parquet(GTEX_DN / "module_target_correlation.parquet",
                          index=False)
    _log(f"  module_target_correlation.parquet: {len(mod_target):,} rows  "
         f"({(GTEX_DN / 'module_target_correlation.parquet').stat().st_size/1e6:.1f} MB)")

    # ------------------------------------------------------------------------
    # 3) program_tissue_specificity: tau over averaged module-tissue activity
    #    per program.
    # ------------------------------------------------------------------------
    _log(f"computing program tissue-specificity (tau) — filtering to "
         f"tissues with >= {MIN_SAMPLES_FOR_TAU} samples")
    activity = activity.merge(mp, on="module_id", how="left")
    prog_tissue_mean = (activity.groupby(["dominant_program", "tissue"])
                                  ["mean_tf_tpm"].mean()
                                  .unstack("tissue"))
    prog_tissue_mean = prog_tissue_mean[tissues]

    # Filter low-N tissues using tissue_index.tsv (mean_samples per tissue)
    tissue_index_fn = GTEX_DN / "tissue_index.tsv"
    well_sampled_tissues = set(tissues)
    if tissue_index_fn.exists():
        ti = pd.read_csv(tissue_index_fn, sep="\t")
        well_sampled_tissues = set(
            ti.loc[ti["mean_samples"] >= MIN_SAMPLES_FOR_TAU, "tissue"]
        )
        dropped = [t for t in tissues if t not in well_sampled_tissues]
        _log(f"  dropping {len(dropped)} low-N tissues from tau: "
             f"{', '.join(dropped[:8])}{'…' if len(dropped) > 8 else ''}")

    keep_cols = [t for t in tissues if t in well_sampled_tissues]
    prog_tissue_mean_filt = prog_tissue_mean[keep_cols]

    prog_rows = []
    for prog, row in prog_tissue_mean_filt.iterrows():
        v = row.to_numpy(np.float64)
        tau = tau_specificity(v)
        if np.all(np.isnan(v)) or np.nanmax(v) <= 0:
            top = ""
            top_v = float("nan")
        else:
            ix = int(np.nanargmax(v))
            top = keep_cols[ix]
            top_v = float(v[ix])
        prog_rows.append({"program": int(prog),
                          "tau": round(tau, 3),
                          "top_tissue": top,
                          "top_tissue_mean_tpm": round(top_v, 2),
                          "median_tpm": round(float(np.nanmedian(v)), 2),
                          "min_samples_for_tau": MIN_SAMPLES_FOR_TAU,
                          "n_tissues_used": len(keep_cols)})
    prog_tau = pd.DataFrame(prog_rows).sort_values("program")
    prog_tau.to_csv(GTEX_DN / "program_tissue_specificity.tsv",
                     sep="\t", index=False)
    _log("  program tissue-specificity:")
    for _, r in prog_tau.iterrows():
        _log(f"    P{int(r['program'])}  tau={r['tau']:.3f}  "
             f"top={r['top_tissue']} ({r['top_tissue_mean_tpm']:.1f} TPM)")

    _log("DONE")


if __name__ == "__main__":
    main()
