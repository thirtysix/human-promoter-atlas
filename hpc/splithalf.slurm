#!/bin/bash
# Split a tier's peaks by experiment (SRX) and scan each half, producing the two
# peak sets that pipeline/splithalf_threshold.py calibrates against.
#
# The scan is run per half with tss_modules, which also (re)discovers modules
# from that half -- those module sets are DISCARDED. Only peaks.parquet and
# tf_index.tsv are used downstream, and the calibration holds modules fixed from
# the FULL build so that the halves differ in evidence alone.
#SBATCH --partition=small
#SBATCH --job-name=hpa-splithalf
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /appl/profile/zz-csc-env.sh 2>/dev/null || true
set -uo pipefail

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
TIER="${HPA_TIER:-q1e-5}"
TF_SET="${HPA_TF_SET:-whitelist}"
VENV="${HPA_VENV:-/projappl/${HPA_CSC_PROJECT:?}/hpa_venv}"
PY="${PY:-$VENV/bin/python}"
SPLIT="$WORK/data/split"

export HPA_TIER="$TIER" HPA_TF_SET="$TF_SET"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
export HPA_WORKERS="${SLURM_CPUS_PER_TASK:-16}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1
# The split trees hold only TFs with enough experiments to halve (~990 of
# 1,793), which is legitimately below the default 1,000-bed guard.
export HPA_MIN_BEDS="${HPA_MIN_BEDS:-500}"

echo "=== hpa-splithalf ${SLURM_JOB_ID:-local} tier=$TIER cpus=$HPA_WORKERS ==="

# --- 1. split by experiment -------------------------------------------------
# The split TREES are per-tier ($SPLIT/$H/per_TF/$TIER), but the marker used to
# be a single $SPLIT/.split.done shared by every tier. So a second tier found
# the first tier's marker, skipped the split, and then had 0 beds of its own --
# caught here only because the HPA_MIN_BEDS floor is checked below. Key the
# marker on the same axes as the tree it certifies.
SPLIT_MARK="$SPLIT/.split.${TIER}.${TF_SET}.done"
if [[ -f "$SPLIT_MARK" ]]; then
  echo "[skip] split (done for $TIER/$TF_SET)"
else
  echo "[run ] split_by_srx  tier=$TIER tf_set=$TF_SET"
  HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas" HPA_ANALYSIS_ROOT="$WORK/out" \
    "$PY" "$WORK/code/acquire/split_by_srx.py" --tier "$TIER" --out "$SPLIT" \
    || { echo "FATAL: split failed" >&2; exit 1; }
  echo "ok $(date -Is)" > "$SPLIT_MARK"
fi

# The halves must carry the rename table too, or config resolves no renames and
# the two halves would silently use a different TF axis from the full build.
cp "$WORK/data/chip-atlas/chip_atlas_antigen_classification.tsv" "$SPLIT/A/" 2>/dev/null
cp "$WORK/data/chip-atlas/chip_atlas_antigen_classification.tsv" "$SPLIT/B/" 2>/dev/null

# --- 2. scan each half ------------------------------------------------------
cd "$WORK/code/pipeline" || exit 1
rc=0
for H in A B; do
  # The halves MUST carry the axes. This was "splithalf.$H", which collides
  # across tiers: a q1e-20 run found the q1e-5 halves' .scan.done, printed
  # [skip], exited 0, and left splithalf_threshold.py comparing q1e-20 modules
  # against q1e-5 scans -- plausible numbers, silently wrong. The assignment
  # score is deliberately NOT in the name: splithalf_threshold.py reads
  # peaks.parquet and thresholds in memory, so one scan serves every score.
  OUT="$WORK/out/splithalf.${TIER}.${TF_SET}.$H"
  if [[ -f "$OUT/.scan.done" ]]; then echo "[skip] scan $H (done)"; continue; fi
  n=$(ls -1 "$SPLIT/$H/per_TF/$TIER"/*.bed.gz 2>/dev/null | wc -l)
  echo "[run ] scan half $H  ($n beds)"
  [[ "$n" -ge "$HPA_MIN_BEDS" ]] || { echo "FATAL: only $n beds in half $H (floor $HPA_MIN_BEDS)" >&2; rc=1; break; }
  t0=$SECONDS
  # HPA_KS="" -- splithalf_threshold.py reads only peaks.parquet and
  # tf_index.tsv, so the four NMF fits were computed and discarded. On the
  # 1,793-TF axis their dense intermediates are what put this job over 48G.
  if HPA_CHIP_ATLAS_DIR="$SPLIT/$H" HPA_OUT_DIR="$OUT" HPA_KS="" \
     "$PY" tss_modules.001.py; then
    echo "ok" > "$OUT/.scan.done"
    echo "[done] half $H in $((SECONDS-t0))s"
  else
    echo "[FAIL] half $H" >&2; rc=1; break
  fi
done

echo "=== finished rc=$rc ==="
for H in A B; do
  d="$WORK/out/splithalf.${TIER}.${TF_SET}.$H"
  f="$d/tss_modules/peaks.parquet"
  [[ -f "$f" ]] && printf "  half %s %s  peaks.parquet %s\n" \
    "$H" "$(basename "$d")" "$(du -h "$f" | cut -f1)"
done
echo "  compare with:"
echo "    python pipeline/splithalf_threshold.py \\"
echo "      --modules <build>/tss_modules \\"
echo "      --half-a  out/splithalf.${TIER}.${TF_SET}.A/tss_modules \\"
echo "      --half-b  out/splithalf.${TIER}.${TF_SET}.B/tss_modules"
exit $rc
