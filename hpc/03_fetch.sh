#!/usr/bin/env bash
# Pull a finished build down from Roihu scratch, verify it, and optionally free
# the scratch copy.
#
# This is a PIPELINE STEP, not cleanup. Roihu bills scratch at 6 BU/TiB-hour
# from the first byte with no free tier, so left alone the storage cost of a
# result set overtakes the entire compute cost of producing it within days.
#
#   hpc/03_fetch.sh --tier q1e-50 --tf-set whitelist
#   hpc/03_fetch.sh --tier q1e-5 --tf-set all --clean
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a

TIER="${HPA_TIER:-q1e-50}"; TF_SET="${HPA_TF_SET:-whitelist}"; CLEAN=0; SCORE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --tf-set) TF_SET="$2"; shift 2 ;;
    --min-score) SCORE="$2"; shift 2 ;;
    --clean) CLEAN=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PROJ="${HPA_CSC_PROJECT:-}"
[[ -n "$PROJ" ]] || { echo "need HPA_CSC_PROJECT in .env" >&2; exit 2; }
REMOTE="${HPA_REMOTE:-roihu}"
WORK="${HPA_SCRATCH:-/scratch/$PROJ/human-promoter-atlas}"

# Ask config for the build directory, exactly as pipeline.slurm does. This was
# "the .s<score> suffix appears only off the default 500", copied from config
# and then left behind when config changed to ALWAYS include the score: the two
# had already diverged, so a fetch would rsync q1e-5.all/ over a build that
# lives in q1e-5.all.s250/. A duplicated path rule drifts; asking cannot.
# `env` and not a bare assignment prefix: a prefix has to be literal at parse
# time, so ${SCORE:+HPA_MIN_SCORE_ASSIGN=...} would expand into a command name.
DST=$(cd "$REPO/pipeline" && env HPA_TIER="$TIER" HPA_TF_SET="$TF_SET" \
      ${SCORE:+HPA_MIN_SCORE_ASSIGN="$SCORE"} \
      python3 -c "import config; print(config.OUT_DN)") \
  || { echo "[fetch] could not resolve the build path from config" >&2; exit 1; }
BUILD="$(basename "$DST")"
SRC="$WORK/out/$BUILD"

echo "[fetch] $REMOTE:$SRC -> $DST"

# Reachability first, and separately from the stamp check. An expired cert makes
# the ssh below print nothing, which the stamp test then reports as "job did not
# finish" -- sending you to look at SLURM when the real problem is 24 h of cert
# validity having run out. Roihu certs are short-lived; say so by name.
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" true 2>/dev/null; then
  echo "[fetch] cannot reach $REMOTE -- this is a connection/auth failure, not a" >&2
  echo "        missing build. Roihu certs last 24 h; re-sign with:" >&2
  echo "        python3 ~/.ssh/csc_cert.py -u <user> ~/.ssh/id_ed25519.pub" >&2
  exit 1
fi

# Refuse to fetch a partial build: a silently incomplete result set is the worst
# outcome, because nothing downstream flags it.
stamp=$(ssh "$REMOTE" "cat $SRC/_BUILD.json 2>/dev/null")
[[ -n "$stamp" ]] || { echo "[fetch] no _BUILD.json in $SRC -- job did not finish" >&2; exit 1; }
echo "$stamp" | grep -q "\"tier\": \"$TIER\"" \
  || { echo "[fetch] stamp tier mismatch:" >&2; echo "$stamp" >&2; exit 1; }
echo "$stamp" | grep -q "\"tf_set\": \"$TF_SET\"" \
  || { echo "[fetch] stamp tf_set mismatch:" >&2; echo "$stamp" >&2; exit 1; }
echo "[fetch] remote stamp ok"

ssh "$REMOTE" "du -sh $SRC 2>/dev/null" | sed 's/^/  remote size: /'

mkdir -p "$DST"
PROGRESS=""; [[ -t 1 ]] && PROGRESS="--info=progress2"
rsync -az $PROGRESS --partial "$REMOTE:$SRC/" "$DST/" \
  || { echo "[fetch] rsync FAILED -- not cleaning" >&2; exit 1; }

# Verify the invariant that actually matters: every remote file arrived with the
# same size. Compare per-file (by size, not du -- block accounting differs across
# filesystems). Deliberately NOT "the two trees are identical": the destination
# legitimately accumulates local run logs from laptop-side runs, and failing on
# those would block a fetch that is in fact complete.
ssh "$REMOTE" "cd $SRC && find . -type f -printf '%p\t%s\n' | sort" > /tmp/hpa_remote.$$
(cd "$DST" && find . -type f -printf '%p\t%s\n' | sort) > /tmp/hpa_local.$$
trap 'rm -f /tmp/hpa_remote.$$ /tmp/hpa_local.$$' EXIT

missing=$(comm -23 /tmp/hpa_remote.$$ /tmp/hpa_local.$$)
extra=$(comm -13 /tmp/hpa_remote.$$ /tmp/hpa_local.$$ | cut -f1)
r_files=$(wc -l < /tmp/hpa_remote.$$)
r_bytes=$(awk -F'\t' '{s+=$2}END{print s+0}' /tmp/hpa_remote.$$)
printf "  remote files=%s bytes=%s\n" "$r_files" "$r_bytes"

if [[ -n "$missing" ]]; then
  echo "[fetch] INCOMPLETE -- these remote files did not arrive intact:" >&2
  echo "$missing" | head -20 >&2
  echo "[fetch] refusing to clean scratch" >&2
  exit 1
fi
echo "[fetch] verified: all $r_files remote files present at matching size"
if [[ -n "$extra" ]]; then
  echo "  note: $(echo "$extra" | wc -l) local-only file(s) kept (e.g. laptop run logs):"
  echo "$extra" | head -5 | sed 's/^/    /'
fi

if [[ $CLEAN -eq 1 ]]; then
  ssh "$REMOTE" "rm -rf $SRC" && echo "[fetch] scratch build removed"
  echo "[fetch] NOTE: $DST is now the only copy of this build."
else
  echo "[fetch] scratch copy kept. Re-run with --clean once you trust it;"
  echo "        scratch is billed from the first byte and is not backed up."
fi
