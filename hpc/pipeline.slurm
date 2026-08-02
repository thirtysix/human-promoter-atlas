#!/bin/bash
# Human Promoter Atlas -- ChIP-Atlas half, on CSC Roihu.
#
# NOTE: --account is deliberately NOT hardcoded. #SBATCH directives cannot expand
# shell variables, so 02_submit.sh exports SBATCH_ACCOUNT from HPA_CSC_PROJECT
# and sbatch honours it natively. Override per-run with `sbatch --account=...`.
#SBATCH --partition=small
#SBATCH --job-name=hpa-pipeline
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
# 16G = exactly 2 GiB/core, which is the Roihu billing break-even: cost is
# max(0.75 BU/coreh * cores, 0.375 BU/GiBh * mem), so anything above this makes
# the memory term dominate and asking for less saves nothing. Measured q1e-50
# peak was 9.9 GiB MaxRSS (which itself over-reads -- the cgroup counts
# reclaimable page cache). Override for looser tiers, which carry ~3-4x the
# peaks: sbatch --mem=32G, or 02_submit.sh --mem 32G.
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

# The CSC env script is not `set -u` clean -- source it BEFORE tightening.
# `|| true` keeps this file runnable locally for rehearsal.
source /appl/profile/zz-csc-env.sh 2>/dev/null || true
set -uo pipefail

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
TIER="${HPA_TIER:-q1e-50}"
TF_SET="${HPA_TF_SET:-whitelist}"
VENV="${HPA_VENV:-/projappl/${HPA_CSC_PROJECT:?}/hpa_venv}"
PY="${PY:-$VENV/bin/python}"
STAGES="${STAGES:-aggregate modules}"

# --- config for pipeline/config.py ------------------------------------------
export HPA_TIER="$TIER" HPA_TF_SET="$TF_SET"
# Explicit, with the same default as config.py. It normally arrives via
# --export, but relying on that alone means a hand-run job silently builds at a
# different score than intended -- and the score is part of the output path.
export HPA_MIN_SCORE_ASSIGN="${HPA_MIN_SCORE_ASSIGN:-250}"
export HPA_CHIP_ATLAS_DIR="$WORK/data/chip-atlas"
export HPA_GTF="$WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz"
export HPA_DNA_BINDING="$WORK/data/genome/DNA_binding_genes.tsv"
export HPA_TF_LIST="$WORK/data/chip-atlas/tf_list.${TF_SET}.txt"
export HPA_ANALYSIS_ROOT="$WORK/out"
# MSigDB / GTEx / DepMap are intentionally unset: those stages run locally, and
# config.py leaves their paths inert rather than requiring dummy values.

# Workers come from the cgroup, not the node. os.cpu_count() would report 384.
export HPA_WORKERS="${SLURM_CPUS_PER_TASK:-8}"
NCPU="${SLURM_CPUS_PER_TASK:-8}"

# BLAS threads are set PER STAGE, not globally, because the two kinds of stage
# want opposite settings and each stage is a fresh process (so the variable is
# read at import, which is the only time OpenBLAS honours it):
#
#   pool stages (aggregate, modules, programs, multibin) -- one thread. They fan
#     out over TFs with multiprocessing, and a thread-per-core inside every
#     worker is catastrophic on a 384-core node.
#   solver stages (selectk, k10, archetypes) -- all cores. These are single
#     process and dominated by NMF; pinning them to one thread would waste the
#     whole allocation. selectk alone fits 8 ranks x 20 seeds.
blas_for () {
  case "$1" in
    aggregate|modules|programs|multibin) echo 1 ;;
    *) echo "$NCPU" ;;
  esac
}
export MPLBACKEND=Agg          # headless plotting
export PYTHONUNBUFFERED=1

# Ask config for the output path rather than rebuilding it here. This used to be
# "$HPA_ANALYSIS_ROOT/${TIER}.${TF_SET}", which silently diverged the moment the
# score entered the directory name: .done markers landed in q1e-5.all/ while the
# data went to q1e-5.all.s250/, so a rerun found a stale marker and skipped the
# stage it was asked to redo. Any path rule duplicated outside config.py will
# drift from it eventually.
OUT=$(cd "$WORK/code/pipeline" && "$PY" -c "import config; print(config.OUT_DN)") \
  || { echo "FATAL: could not resolve OUT_DN from config" >&2; exit 1; }
mkdir -p "$OUT" "$WORK/logs"

echo "=== hpa-pipeline ${SLURM_JOB_ID:-local} ==="
echo "  tier=$TIER tf_set=$TF_SET workers=$HPA_WORKERS"
echo "  work=$WORK  out=$OUT"
echo "  code sha=$(cat "$WORK/code/GITSHA" 2>/dev/null || echo unknown)"

# --- fail fast on configuration, not per-item ------------------------------
# A missing input is a whole-run problem; discovering it 40 minutes in wastes
# the allocation and looks like a data problem.
for f in "$HPA_GTF" "$HPA_DNA_BINDING" "$PY"; do
  [[ -e "$f" ]] || { echo "FATAL: missing $f" >&2; exit 1; }
done
n_bed=$(ls -1 "$HPA_CHIP_ATLAS_DIR/per_TF/$TIER"/*.bed.gz 2>/dev/null | wc -l)
[[ "$n_bed" -ge 1000 ]] || { echo "FATAL: only $n_bed beds for tier $TIER" >&2; exit 1; }
echo "  beds=$n_bed"

"$PY" -c "import pyranges,sklearn,pandas,numpy,scipy" \
  || { echo "FATAL: venv imports broken -- run hpc/00_build_env.sh" >&2; exit 1; }

cd "$WORK/code/pipeline" || exit 1

# --- stages -----------------------------------------------------------------
# .done markers are written LAST and are the only resume key, so a job killed
# mid-stage simply redoes that stage.
run_stage () {
  local name="$1" script="$2"
  local done_marker="$OUT/.${name}.done"
  if [[ -f "$done_marker" ]]; then echo "[skip] $name (done)"; return 0; fi
  local nt; nt=$(blas_for "$name")
  echo "[run ] $name -> $script  (blas threads=$nt)"
  local t0=$SECONDS
  if OMP_NUM_THREADS="$nt" MKL_NUM_THREADS="$nt" OPENBLAS_NUM_THREADS="$nt" \
     NUMEXPR_NUM_THREADS="$nt" VECLIB_MAXIMUM_THREADS="$nt" "$PY" "$script"; then
    echo "ok $(date -Is)" > "$done_marker"
    echo "[done] $name in $((SECONDS - t0))s"
  else
    echo "[FAIL] $name after $((SECONDS - t0))s" >&2
    return 1
  fi
}

rc=0
for stage in $STAGES; do
  case "$stage" in
    aggregate) run_stage aggregate canonical_promoter_aggregate.001.py || rc=1 ;;
    modules)   run_stage modules   tss_modules.001.py                  || rc=1 ;;
    selectk)   run_stage selectk   tss_modules_select_k.001.py         || rc=1 ;;
    k10)       run_stage k10       tss_modules_k10.py                  || rc=1 ;;
    programs)  run_stage programs  tss_programs.001.py                 || rc=1 ;;
    multibin)  run_stage multibin  tss_programs_multibin.001.py        || rc=1 ;;
    archetypes) run_stage archetypes tss_archetypes.001.py             || rc=1 ;;
    *) echo "unknown stage: $stage" >&2; rc=1 ;;
  esac
  [[ $rc -eq 0 ]] || break
done

# Label the build so nothing downstream has to infer it from a directory name.
"$PY" - <<PY
import sys; sys.path.insert(0, "$WORK/code/pipeline")
import config
p = config.write_build_stamp()
print(f"[stamp] {p}: {config.build_stamp()}")
PY

echo "=== finished rc=$rc  $(du -sh "$OUT" 2>/dev/null | cut -f1) ==="
exit $rc
