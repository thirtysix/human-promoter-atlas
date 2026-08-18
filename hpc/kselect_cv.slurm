#!/bin/bash
# Rank selection by two criteria that are NOT confounded with k.
#
#   1. held-out imputation AUC   (nmf_holdout_cv.py)      -- internal, predictive
#   2. GO-CC complex recovery    (nmf_complex_recovery.py) -- external, biological
#
# The stability statistics already in the repo (reproducible fraction/count,
# ARI, cophenetic) are reported elsewhere as descriptive only: each is
# structurally biased in k, which is why they disagreed about the rank.
#
#SBATCH --partition=small
#SBATCH --job-name=hpa-kselect
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
RANKS="${RANKS:-5,6,7,8,9,10,11,12,13,14,15,16,17,18,20,22,25,30}"
FOLDS="${FOLDS:-3}"
SEEDS="${SEEDS:-5}"
SUBSAMPLE="${SUBSAMPLE:-25000}"
# Empty = the promoter build's occupancy.modules.npz. Set to a genome run's
# occupancy.elements.npz to select a rank for that matrix instead; results
# are then written beside it, not into the promoter build's k_selection.
MATRIX="${MATRIX:-}"
VENV="/projappl/${HPA_CSC_PROJECT:?}/hpa_venv"
PY="${PY:-$VENV/bin/python}"

export HPA_TIER="${HPA_TIER:-q1e-5}" HPA_TF_SET="${HPA_TF_SET:-all}"
export HPA_MIN_SCORE_ASSIGN="${HPA_MIN_SCORE_ASSIGN:-250}"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${HPA_TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
export HPA_MSIGDB="$WORK/data/msigdb/c5.go.bp.v2026.1.Hs.json"
# Solver stage: BLAS gets the whole allocation, set on the invocation because
# OpenBLAS reads this at numpy import.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORK/code/pipeline" || exit 1
echo "=== hpa-kselect ${SLURM_JOB_ID:-local} ==="
echo "  tier=$HPA_TIER tf_set=$HPA_TF_SET score=$HPA_MIN_SCORE_ASSIGN"
echo "  ranks=$RANKS folds=$FOLDS seeds=$SEEDS subsample=$SUBSAMPLE"
echo "  matrix=${MATRIX:-<promoter build default>}"

MAT_ARG=""
[ -n "$MATRIX" ] && MAT_ARG="--matrix $MATRIX"

rc=0
echo ""
echo "--- 1. held-out imputation ---"
"$PY" nmf_holdout_cv.py --ranks "$RANKS" --folds "$FOLDS" \
      --subsample "$SUBSAMPLE" $MAT_ARG \
  || { echo "[FAIL] holdout cv" >&2; rc=1; }

echo ""
echo "--- 2. complex recovery ---"
"$PY" nmf_complex_recovery.py --ranks "$RANKS" --seeds "$SEEDS" \
      --cc "$WORK/data/msigdb/c5.go.cc.v2026.1.Hs.json" $MAT_ARG \
  || { echo "[FAIL] complex recovery" >&2; rc=1; }

echo ""
echo "=== kselect finished rc=$rc ==="
exit $rc
