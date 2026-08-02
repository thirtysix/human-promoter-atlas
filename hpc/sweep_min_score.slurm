#!/bin/bash
# Sweep MIN_SCORE_ASSIGN at a fixed tier and TF axis.
#
# WHY THIS IS THE CLEANEST EXPERIMENT IN THE PROJECT
# --------------------------------------------------
# ChIP-Atlas score is -10*log10(Q) capped at 1000, so score >= 500 is exactly
# Q < 1E-50. The tier and MIN_SCORE_ASSIGN are therefore two thresholds on the
# SAME quantity: the tier governs module DISCOVERY, MIN_SCORE_ASSIGN governs TF
# ASSIGNMENT. Moving to a looser tier without touching the score left assignment
# pinned at Q<1E-50 -- so the sparse-factor under-counting that motivated the
# move was only half addressed.
#
# MIN_SCORE_ASSIGN touches exactly one line of tss_modules (the sp_mask), never
# the KDE or the module boundaries. So across this sweep the discovered modules
# are IDENTICAL and only attribution changes -- a genuine single-variable
# experiment, unlike the tier comparison where everything moves at once.
#
# The per-TF peak scan is likewise score-independent, so each sweep point
# hardlinks the first point's scan artifacts (peaks.parquet + tss_table.tsv +
# tf_index.tsv) and RESUME_FROM_PEAKS picks them up. That turns a ~90 s scan per
# point into a free link, and guarantees every point sees byte-identical input.
#SBATCH --partition=small
#SBATCH --job-name=hpa-sweep
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /appl/profile/zz-csc-env.sh 2>/dev/null || true
set -uo pipefail

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
TIER="${HPA_TIER:-q1e-5}"
TF_SET="${HPA_TF_SET:-whitelist}"
SCORES="${SCORES:-500 300 200 100 50}"
VENV="${HPA_VENV:-/projappl/${HPA_CSC_PROJECT:?}/hpa_venv}"
PY="${PY:-$VENV/bin/python}"

export HPA_TIER="$TIER" HPA_TF_SET="$TF_SET"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
export HPA_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORK/code/pipeline" || exit 1
echo "=== hpa-sweep ${SLURM_JOB_ID:-local}  tier=$TIER tf_set=$TF_SET scores='$SCORES' ==="

# Locate a completed scan to seed from (any existing build at this tier/axis).
SEED=""
for cand in "$WORK/out/$TIER.$TF_SET" $(for s in $SCORES; do echo "$WORK/out/$TIER.$TF_SET.s$s"; done); do
  if [[ -f "$cand/tss_modules/peaks.parquet" ]]; then SEED="$cand/tss_modules"; break; fi
done
[[ -n "$SEED" ]] && echo "  seed scan: $SEED" || echo "  no seed scan found; first point will scan from BEDs"

rc=0
for S in $SCORES; do
  # Resolve from config, never rebuild the rule here -- it already drifted once
  # when the score entered the directory name.
  OUT=$(HPA_MIN_SCORE_ASSIGN="$S" "$PY" -c "import config; print(config.OUT_DN)")
  MARK="$OUT/.modules.done"
  if [[ -f "$MARK" ]]; then echo "[skip] score=$S (done)"; continue; fi

  mkdir -p "$OUT/tss_modules"
  # Hardlink the score-independent scan so every point sees identical input and
  # skips the ~90 s per-TF pass. Falls back to a copy across filesystems.
  if [[ -n "$SEED" && ! -f "$OUT/tss_modules/peaks.parquet" ]]; then
    for f in peaks.parquet tss_table.tsv tf_index.tsv; do
      ln "$SEED/$f" "$OUT/tss_modules/$f" 2>/dev/null || cp "$SEED/$f" "$OUT/tss_modules/$f"
    done
    echo "  seeded scan into $(basename "$OUT")"
  fi

  echo "[run ] MIN_SCORE_ASSIGN=$S -> $(basename "$OUT")"
  t0=$SECONDS
  if HPA_MIN_SCORE_ASSIGN="$S" "$PY" tss_modules.001.py; then
    echo "ok $(date -Is)" > "$MARK"
    HPA_MIN_SCORE_ASSIGN="$S" "$PY" -c "import sys;sys.path.insert(0,'.');import config;config.write_build_stamp()"
    echo "[done] score=$S in $((SECONDS - t0))s"
    [[ -z "$SEED" ]] && SEED="$OUT/tss_modules"
  else
    echo "[FAIL] score=$S after $((SECONDS - t0))s" >&2; rc=1; break
  fi
done

echo "=== sweep finished rc=$rc ==="
for S in $SCORES; do
  # Resolve from config, never rebuild the rule here -- it already drifted once
  # when the score entered the directory name.
  OUT=$(HPA_MIN_SCORE_ASSIGN="$S" "$PY" -c "import config; print(config.OUT_DN)")
  L=$(ls -t "$OUT"/logs/tss_modules.*.log 2>/dev/null | head -1)
  [[ -n "$L" ]] && printf "  score=%-4s %s\n" "$S" \
    "$(grep -hE 'total modules|modules in matrix' "$L" | tr '\n' ' ' | tr -s ' ')"
done
exit $rc
