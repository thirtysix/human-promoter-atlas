#!/bin/bash
# Genome-wide annotation-free element discovery at the calibrated support floor.
#
# Workstream D2. Discovery is annotation-free; genes enter only afterwards, as
# a distance label (promoter <=1kb / proximal <=10kb / distal). The regression
# gate is that promoter-stratum elements reproduce the promoter atlas's modules
# -- if they do not, this pipeline is wrong and nothing distal may be believed.
#
# MIN_SUPPORT defaults to 11 in genome_modules.py, calibrated against the
# circular-shift null (job 708621). It is an output-path axis, so a re-run at
# another floor lands in its own directory and cannot overwrite this one.
#
# Chromosomes 1-22 and X, matching the null. Y and MT are excluded on purpose:
# Y is small and repetitive, and MT is a 16.5 kb circular molecule where the
# +/-1500 bp local weighting wraps the genome several times.
#
# CHROMS is set INSIDE this script, never passed via --export: sbatch's export
# list is comma-separated, so a comma-bearing value is truncated at its first
# comma and the job runs a silent single-chromosome analysis, exit 0.
#
#SBATCH --partition=small
#SBATCH --job-name=hpa-gmod
#SBATCH --time=04:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
CHROMS="${CHROMS:-1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,X}"
CLUSTER_GAP="${CLUSTER_GAP:-12500}"
VENV="/projappl/${HPA_CSC_PROJECT:?}/hpa_venv"
PY="${PY:-$VENV/bin/python}"

export HPA_TIER="${HPA_TIER:-q1e-5}" HPA_TF_SET="${HPA_TF_SET:-all}"
export HPA_MIN_SCORE_ASSIGN="${HPA_MIN_SCORE_ASSIGN:-250}"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${HPA_TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
# Pool workers hold per-TF peak slices; 8 is the shape that fit the null run
# (16.3 GB peak of 64 GB). Raising it raises memory, not throughput.
export HPA_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export MPLBACKEND=Agg PYTHONUNBUFFERED=1

cd "$WORK/code/pipeline" || exit 1
echo "=== hpa-gmod ${SLURM_JOB_ID:-local} ==="
echo "  tier=$HPA_TIER tf_set=$HPA_TF_SET score=$HPA_MIN_SCORE_ASSIGN"
echo "  chroms=$CHROMS cluster_gap=$CLUSTER_GAP"
echo "  min_support: script default (calibrated), overridable with MIN_SUPPORT"

SUP_ARG=""
[ -n "$MIN_SUPPORT" ] && SUP_ARG="--min-support $MIN_SUPPORT"

"$PY" genome_modules.py \
    --chroms "$CHROMS" --cluster-gap "$CLUSTER_GAP" \
    --workers "$HPA_WORKERS" $SUP_ARG
rc=$?
echo "=== gmod finished rc=$rc ==="
exit $rc
