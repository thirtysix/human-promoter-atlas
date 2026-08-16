#!/bin/bash
# Multi-seed consensus at one rank across a (tier, score) grid.
#
# Scores each cell by REPRODUCIBLE PROGRAM COUNT: fit N seeds, match programs
# by cosine, count those recovered at >= COS_MIN. This is the objective the
# grid is being swept against -- unlike GO it is not monotone in assignment
# volume, and unlike median-TFs-per-module it measures the thing the atlas
# actually ships rather than a proxy for it.
#
# Submit with CELLS="tier:score tier:score ...", e.g.
#   CELLS="q1e-5:250 q1e-5:300 q1e-20:300" sbatch --export=ALL hpc/consensus_sweep.slurm
#
#SBATCH --partition=small
#SBATCH --job-name=hpa-consensus
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
TF_SET="${HPA_TF_SET:-all}"
K="${HPA_K:-10}"
SEEDS="${HPA_SEEDS:-20}"
CELLS="${CELLS:?need CELLS=\"tier:score ...\"}"
VENV="/projappl/${HPA_CSC_PROJECT:?}/hpa_venv"
PY="${PY:-$VENV/bin/python}"

export HPA_TF_SET="$TF_SET"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
# Solver stage: give BLAS the whole allocation. Set on the invocation, because
# OpenBLAS reads this at numpy import -- a pool initialiser is far too late.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORK/code/pipeline" || exit 1
echo "=== hpa-consensus ${SLURM_JOB_ID:-local} k=$K seeds=$SEEDS tf_set=$TF_SET ==="
echo "  cells: $CELLS"

rc=0
for CELL in $CELLS; do
  T="${CELL%%:*}"; S="${CELL##*:}"
  # Ask config for the path -- never rebuild the <tier>.<tf_set>.s<score> rule.
  OUT=$(HPA_TIER="$T" HPA_MIN_SCORE_ASSIGN="$S" "$PY" -c "import config; print(config.OUT_DN)") \
    || { echo "[FAIL] $CELL: cannot resolve OUT_DN" >&2; rc=1; continue; }
  if [[ ! -f "$OUT/tss_modules/occupancy.modules.npz" ]]; then
    echo "[skip] $CELL -- no occupancy matrix in $(basename "$OUT")"; continue
  fi
  if [[ -f "$OUT/tss_modules/nmf.k${K}.consensus.json" ]]; then
    echo "[skip] $CELL (k=$K consensus already present)"; continue
  fi
  echo ""
  echo "[run ] $CELL -> $(basename "$OUT")"
  t0=$SECONDS
  if HPA_TIER="$T" HPA_MIN_SCORE_ASSIGN="$S" \
     "$PY" tss_modules_consensus.py --k "$K" --seeds "$SEEDS"; then
    echo "[done] $CELL in $((SECONDS - t0))s"
  else
    echo "[FAIL] $CELL after $((SECONDS - t0))s" >&2; rc=1
  fi
done

echo ""
echo "=== consensus sweep finished rc=$rc ==="
for CELL in $CELLS; do
  T="${CELL%%:*}"; S="${CELL##*:}"
  OUT=$(HPA_TIER="$T" HPA_MIN_SCORE_ASSIGN="$S" "$PY" -c "import config; print(config.OUT_DN)" 2>/dev/null)
  J="$OUT/tss_modules/nmf.k${K}.consensus.json"
  [[ -f "$J" ]] && printf "  %-16s %s\n" "$CELL" \
    "$("$PY" -c "import json,sys; d=json.load(open('$J')); print('reproducible %d/%d  collapses=%d' % (d['n_reproducible'], d['k'], d['collapsed_seeds_reseeded']))")"
done
exit $rc
