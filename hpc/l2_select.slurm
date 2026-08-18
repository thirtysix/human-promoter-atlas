#!/bin/bash
# Choose the ridge penalty for the masked NMF, before sweeping rank with it.
#
# Unregularised, the masked objective leaves W @ H unconstrained wherever
# O == 0, so at high rank the fit is healthy on the entries it is scored on and
# arbitrary everywhere else. Measured on the genome matrix: at k=45-60 every
# seed produced max(W @ H) in the thousands on a 0/1 matrix and held-out RMSE
# of 1.5-8.3, while training error fell smoothly. Reseeding cannot fix that --
# it is capacity escaping, not a bad start -- so the sweep either regularises
# or cannot report the high-rank arm at all. Without a trustworthy high-rank
# arm there is no way to show the optimum is interior, which is the whole point
# of using held-out imputation as the criterion.
#
# lambda is chosen by held-out AUC, the same statistic the rank sweep uses, on
# a grid spanning four orders of magnitude, then held FIXED across the rank
# sweep. Ranks 20/45/60 bracket the healthy, marginal and diverged regimes, so
# this also confirms the penalty actually restores the high end.
#
# LAMBDAS and RANKS are set inside this script, never passed via --export:
# sbatch's export list is comma separated, so a comma-bearing value arrives
# truncated at its first comma and the job silently sweeps one point.
#
#SBATCH --partition=small
#SBATCH --job-name=hpa-l2sel
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
LAMBDAS="${LAMBDAS:-0 0.001 0.01 0.1 1 10}"
RANKS="${RANKS:-20,45,60}"
FOLDS="${FOLDS:-3}"
SUBSAMPLE="${SUBSAMPLE:-50000}"
MATRIX="${MATRIX:?set MATRIX to an occupancy .npz}"
VENV="/projappl/${HPA_CSC_PROJECT:?}/hpa_venv"
PY="${PY:-$VENV/bin/python}"

export HPA_TIER="${HPA_TIER:-q1e-5}" HPA_TF_SET="${HPA_TF_SET:-all}"
export HPA_MIN_SCORE_ASSIGN="${HPA_MIN_SCORE_ASSIGN:-250}"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${HPA_TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
# Solver stage: BLAS gets the whole allocation, set on the invocation because
# OpenBLAS reads this at numpy import.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORK/code/pipeline" || exit 1
OUTDIR="$(dirname "$MATRIX")/k_selection"
mkdir -p "$OUTDIR"
echo "=== hpa-l2sel ${SLURM_JOB_ID:-local} ==="
echo "  matrix=$MATRIX"
echo "  lambdas=$LAMBDAS ranks=$RANKS folds=$FOLDS subsample=$SUBSAMPLE"

rc=0
for L in $LAMBDAS; do
  echo ""
  echo "--- lambda = $L ---"
  # A lambda whose fits all collapse makes the script raise; that is the
  # intended outcome and must not abort the remaining grid points.
  "$PY" nmf_holdout_cv.py --matrix "$MATRIX" --ranks "$RANKS" \
        --folds "$FOLDS" --subsample "$SUBSAMPLE" --l2 "$L" \
        --out "$OUTDIR/holdout_cv.l2_${L}.tsv" \
    || { echo "[collapsed or failed at lambda=$L]" >&2; rc=$((rc + 1)); }
done

echo ""
echo "=== l2sel finished, $rc lambda(s) unusable ==="
exit 0
