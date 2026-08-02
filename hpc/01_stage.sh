#!/usr/bin/env bash
# Stage code + inputs for one build onto Roihu scratch.
#
# Only the ChIP-Atlas half runs on the cluster, so only its inputs are staged:
# the per-tier peak BEDs, the GTF, the DNA-binding table, the TF lists and the
# antigen classification. MSigDB / GTEx / DepMap stay local -- config.py leaves
# those paths inert when unset rather than demanding dummy values.
#
# Run from the repo root:   hpc/01_stage.sh [--tier q1e-5] [--dry-run]
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a

TIER="${HPA_TIER:-q1e-50}"; DRY=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --dry-run) DRY="--dry-run"; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PROJ="${HPA_CSC_PROJECT:-}"
[[ -n "$PROJ" ]] || { echo "need HPA_CSC_PROJECT in .env" >&2; exit 2; }
REMOTE="${HPA_REMOTE:-roihu}"
WORK="${HPA_SCRATCH:-/scratch/$PROJ/human-promoter-atlas}"
PER_TF="${HPA_CHIP_ATLAS_DIR:?}/per_TF/$TIER"

[[ -d "$PER_TF" ]] || { echo "no such peak dir: $PER_TF" >&2; exit 2; }
n_bed=$(ls -1 "$PER_TF"/*.bed.gz 2>/dev/null | wc -l)
[[ "$n_bed" -ge 1000 ]] || { echo "only $n_bed beds in $PER_TF -- wrong tier?" >&2; exit 2; }

echo "[stage] remote=$REMOTE work=$WORK tier=$TIER beds=$n_bed ($(du -sh "$PER_TF" | cut -f1))"

ssh "$REMOTE" "mkdir -p $WORK/{code,logs,out,data/chip-atlas/per_TF,data/genome}"

# Code. --delete keeps the staged tree honest; __pycache__ excluded so a local
# 3.12 build never shadows the cluster's 3.9 interpreter.
rsync -az $DRY --delete --exclude='__pycache__' \
  "$REPO/pipeline/" "$REMOTE:$WORK/code/pipeline/"
rsync -az $DRY --exclude='__pycache__' "$REPO/hpc/" "$REMOTE:$WORK/hpc/"
rsync -az $DRY "$REPO/acquire/tf_list."*.txt \
  "$REPO/acquire/chip_atlas_antigen_classification.tsv" \
  "$REMOTE:$WORK/data/chip-atlas/"

# The staged tree has no .git, so capture provenance here and ship it as a file.
GITSHA=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo unknown)
[[ -z "$DRY" ]] && ssh "$REMOTE" "echo $GITSHA > $WORK/code/GITSHA"

# Reference data (small, changes rarely).
rsync -az $DRY "${HPA_GTF:?}" "${HPA_DNA_BINDING:?}" "$REMOTE:$WORK/data/genome/"

# Peaks (the bulk). Resumable and idempotent -- safe to re-run after a drop.
echo "[stage] syncing peaks (resumable)..."
# progress2 emits one line per update, which is fine on a terminal and thousands
# of lines in a captured log -- only ask for it when stdout is a tty.
PROGRESS=""; [[ -t 1 ]] && PROGRESS="--info=progress2"
rsync -az $DRY $PROGRESS --partial \
  "$PER_TF/" "$REMOTE:$WORK/data/chip-atlas/per_TF/$TIER/"

[[ -n "$DRY" ]] && { echo "[stage] dry run, nothing written"; exit 0; }

echo "[stage] verifying remote counts..."
ssh "$REMOTE" "bash -lc '
  n=\$(ls -1 $WORK/data/chip-atlas/per_TF/$TIER/*.bed.gz 2>/dev/null | wc -l)
  echo \"  beds staged : \$n (local $n_bed)\"
  [ \"\$n\" -eq $n_bed ] || { echo \"  COUNT MISMATCH\" >&2; exit 1; }
  echo \"  gtf         : \$(ls -1 $WORK/data/genome/ | tr \"\\n\" \" \")\"
  echo \"  code        : \$(ls -1 $WORK/code/pipeline/*.py | wc -l) py files, sha \$(cat $WORK/code/GITSHA)\"
  du -sh $WORK 2>/dev/null | sed \"s/^/  usage      : /\"
'"
echo "[stage] done"
