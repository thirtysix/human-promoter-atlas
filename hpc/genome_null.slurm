#!/bin/bash
# Genome-wide support-floor calibration by circular-shift null.
#
# The pilot calibrated MIN_SUPPORT=11 at 5% FDR on chr20-22 only. Those three
# are all small and gene-dense; the contrast that matters (chr19, densest, vs
# chr13/chr4, large and gene-poor) is absent from it entirely, so the pilot
# cannot say whether the floor is stable genome-wide.
#
# Chromosomes 1-22 and X. Y and MT are excluded on purpose: Y is small and
# repetitive with few genes, and MT is a 16.5 kb circular molecule where a
# +/-1500 bp neighbourhood wraps the genome several times, making both the
# local weighting and the KDE meaningless.
#
#SBATCH --partition=small
#SBATCH --job-name=hpa-gnull
#SBATCH --time=06:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
CHROMS="${CHROMS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X}"
SHUFFLES="${SHUFFLES:-5}"
TARGET_FDR="${TARGET_FDR:-0.05}"
VENV="/projappl/${HPA_CSC_PROJECT:?}/hpa_venv"
PY="${PY:-$VENV/bin/python}"

export HPA_TIER="${HPA_TIER:-q1e-5}" HPA_TF_SET="${HPA_TF_SET:-all}"
export HPA_MIN_SCORE_ASSIGN="${HPA_MIN_SCORE_ASSIGN:-250}"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${HPA_TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
# Pool workers hold per-TF peak slices; 8 is the shape that fit in the
# promoter scan. Raising it raises memory, not throughput.
export HPA_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORK/code/pipeline" || exit 1
echo "=== hpa-gnull ${SLURM_JOB_ID:-local} ==="
echo "  tier=$HPA_TIER tf_set=$HPA_TF_SET score=$HPA_MIN_SCORE_ASSIGN"
echo "  chroms=$CHROMS shuffles=$SHUFFLES target_fdr=$TARGET_FDR"

"$PY" genome_support_null.py \
    --chroms "$CHROMS" --shuffles "$SHUFFLES" \
    --target-fdr "$TARGET_FDR" --workers "$HPA_WORKERS"
rc=$?
echo "=== gnull finished rc=$rc ==="
exit $rc
