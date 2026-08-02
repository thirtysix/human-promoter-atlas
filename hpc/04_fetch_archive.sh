#!/usr/bin/env bash
# Pull the zstd peak archive off Roihu scratch to durable local storage and
# verify it by its own sha256.
#
# The archive is the DURABLE SOURCE OF TRUTH for a tier; the per-TF working
# files are a derived convenience that acquire/split_bulk_to_tfs.py can
# regenerate on any machine. Scratch is billed from the first byte, is not
# backed up, and is periodically cleaned -- so this is a pipeline step.
#
#   hpc/04_fetch_archive.sh --tier q1e-5
#   hpc/04_fetch_archive.sh --tier q1e-5 --clean
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a

TIER="${HPA_TIER:-q1e-5}"; CLEAN=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PROJ="${HPA_CSC_PROJECT:-}"
[[ -n "$PROJ" ]] || { echo "need HPA_CSC_PROJECT in .env" >&2; exit 2; }
REMOTE="${HPA_REMOTE:-roihu}"
WORK="${HPA_SCRATCH:-/scratch/$PROJ/human-promoter-atlas}"
# No default: the destination is machine-specific, so it is required rather than
# guessed. Set HPA_ARCHIVE_DIR in .env (see .env.example).
DST="${HPA_ARCHIVE_DIR:-}"
[[ -n "$DST" ]] || { echo "set HPA_ARCHIVE_DIR (durable storage for the archive)" >&2; exit 2; }

# Fail early rather than half-way through a multi-GB transfer.
mountpoint -q "$(df -P "$(dirname "$DST")" 2>/dev/null | tail -1 | awk '{print $6}')" 2>/dev/null \
  || echo "  note: could not confirm $DST is on a mounted volume -- continuing"
[[ -d "$(dirname "$DST")" ]] || { echo "destination parent missing: $(dirname "$DST")" >&2; exit 2; }
mkdir -p "$DST"

case "$TIER" in q1e-50) I=50 ;; q1e-20) I=20 ;; q1e-10) I=10 ;; q1e-5) I=05 ;;
  *) echo "unknown tier $TIER" >&2; exit 2 ;; esac

# Resolve the exact basename remotely (it encodes the antigen count).
BASE=$(ssh "$REMOTE" "ls -1 $WORK/archive/chip_atlas.hg38.Oth.ALL.$I.tf*.bed.zst 2>/dev/null | head -1 | xargs -r basename | sed 's/\.bed\.zst$//'")
[[ -n "$BASE" ]] || { echo "no archive for tier $TIER in $WORK/archive" >&2; exit 1; }
echo "[archive] $BASE"

avail=$(df -P "$DST" | tail -1 | awk '{print $4}')
need=$(ssh "$REMOTE" "stat -c%s $WORK/archive/$BASE.bed.zst")
echo "  remote size $((need/1024/1024)) MB, local free $((avail/1024)) MB"
[[ $((avail*1024)) -gt $((need*2)) ]] || { echo "  insufficient free space" >&2; exit 1; }

PROGRESS=""; [[ -t 1 ]] && PROGRESS="--info=progress2"
rsync -a $PROGRESS --partial \
  "$REMOTE:$WORK/archive/$BASE.bed.zst" \
  "$REMOTE:$WORK/archive/$BASE.sha256" \
  "$REMOTE:$WORK/archive/$BASE.manifest.json" \
  "$DST/" || { echo "[archive] rsync FAILED" >&2; exit 1; }

echo "[archive] verifying sha256 locally..."
( cd "$DST" && sha256sum -c "$BASE.sha256" ) \
  || { echo "[archive] CHECKSUM FAILED -- not cleaning" >&2; exit 1; }

echo "[archive] verifying it decompresses..."
rows=$(zstd -dc "$DST/$BASE.bed.zst" | wc -l)
want=$(python3 -c "import json;print(json.load(open('$DST/$BASE.manifest.json'))['n_rows'])")
echo "  rows local=$rows manifest=$want"
[[ "$rows" -eq "$want" ]] || { echo "[archive] ROW MISMATCH" >&2; exit 1; }
echo "[archive] verified"

ls -la "$DST"/$BASE.*
if [[ $CLEAN -eq 1 ]]; then
  ssh "$REMOTE" "rm -f $WORK/archive/$BASE.*" && echo "[archive] scratch copy removed"
  echo "[archive] NOTE: $DST is now the only copy."
else
  echo "[archive] scratch copy kept; re-run with --clean once you trust it."
fi
