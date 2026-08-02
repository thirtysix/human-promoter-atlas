#!/usr/bin/env bash
# Preflight (read-only, costs no BU) then submit the pipeline job.
#
#   hpc/02_submit.sh --tier q1e-5 --tf-set all
#   hpc/02_submit.sh --tier q1e-50 --check-only
#   hpc/02_submit.sh --tier q1e-5 --stages "aggregate modules selectk"
set -uo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
[[ -f "$REPO/.env" ]] && set -a && . "$REPO/.env" && set +a

TIER="${HPA_TIER:-q1e-50}"; TF_SET="${HPA_TF_SET:-whitelist}"
STAGES="aggregate modules"; CHECK_ONLY=0; MEM=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --tf-set) TF_SET="$2"; shift 2 ;;
    --stages) STAGES="$2"; shift 2 ;;
    --mem) MEM="$2"; shift 2 ;;
    --check-only) CHECK_ONLY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

PROJ="${HPA_CSC_PROJECT:-}"
[[ -n "$PROJ" ]] || { echo "need HPA_CSC_PROJECT in .env" >&2; exit 2; }
REMOTE="${HPA_REMOTE:-roihu}"
WORK="${HPA_SCRATCH:-/scratch/$PROJ/human-promoter-atlas}"
VENV="/projappl/$PROJ/hpa_venv"

fail=0
say () { printf "  %-34s %s\n" "$1" "$2"; }

echo "[preflight] $REMOTE  tier=$TIER tf_set=$TF_SET stages='$STAGES'"

# 1. Reachability. A stale SSH certificate is the single most common blocker --
#    Roihu certs last 24 h -- so say exactly how to fix it.
if ! ssh -o BatchMode=yes -o ConnectTimeout=20 "$REMOTE" true 2>/dev/null; then
  say "ssh" "UNREACHABLE"
  echo "      re-sign the 24h cert:  python3 ~/.ssh/csc_cert.py -u <user> ~/.ssh/id_ed25519.pub" >&2
  exit 1
fi
say "ssh" "ok"

# Everything else in ONE remote shell -- each round trip is ~1 s.
out=$(ssh "$REMOTE" "bash -lc '
  set -uo pipefail
  echo \"venv=\$([ -x $VENV/bin/python ] && $VENV/bin/python -c \"import pyranges,sklearn,pandas,numpy,scipy;print(\\\"ok\\\")\" 2>/dev/null || echo MISSING)\"
  echo \"beds=\$(ls -1 $WORK/data/chip-atlas/per_TF/$TIER/*.bed.gz 2>/dev/null | wc -l)\"
  echo \"code=\$(ls -1 $WORK/code/pipeline/*.py 2>/dev/null | wc -l)\"
  echo \"tflist=\$([ -f $WORK/data/chip-atlas/tf_list.$TF_SET.txt ] && grep -vc \"^#\" $WORK/data/chip-atlas/tf_list.$TF_SET.txt || echo 0)\"
  echo \"gtf=\$([ -f $WORK/data/genome/Homo_sapiens.GRCh38.114.gtf.gz ] && echo ok || echo MISSING)\"
  echo \"sha=\$(cat $WORK/code/GITSHA 2>/dev/null || echo unknown)\"
  echo \"parts=\$(sinfo -h -o %P 2>/dev/null | tr \"\\n\" \",\")\"
'" 2>/dev/null)

get () { echo "$out" | grep "^$1=" | cut -d= -f2-; }

[[ "$(get venv)" == "ok" ]] && say "venv imports" "ok" \
  || { say "venv imports" "FAIL -- run hpc/00_build_env.sh"; fail=1; }
n=$(get beds); [[ "${n:-0}" -ge 1000 ]] && say "peak beds ($TIER)" "$n" \
  || { say "peak beds ($TIER)" "${n:-0} -- run hpc/01_stage.sh --tier $TIER"; fail=1; }
n=$(get code); [[ "${n:-0}" -ge 15 ]] && say "pipeline scripts" "$n" \
  || { say "pipeline scripts" "${n:-0} -- run hpc/01_stage.sh"; fail=1; }
n=$(get tflist); [[ "${n:-0}" -ge 1000 ]] && say "tf_list.$TF_SET" "$n TFs" \
  || { say "tf_list.$TF_SET" "${n:-0} -- run hpc/01_stage.sh"; fail=1; }
[[ "$(get gtf)" == "ok" ]] && say "GTF" "ok" || { say "GTF" "MISSING"; fail=1; }
say "code sha" "$(get sha)"
say "partitions" "$(get parts)"

[[ $fail -eq 0 ]] || { echo "[preflight] BLOCKED"; exit 1; }
echo "[preflight] all clear"
[[ $CHECK_ONLY -eq 1 ]] && exit 0

# Billing is max(0.75 BU/coreh * cores, 0.375 BU/GiBh * GiB) per hour. At the
# default 8 cores / 16 GiB both terms are 6 BU/h, which is the cheapest shape
# for this many cores; raising --mem past 2 GiB/core only inflates the bill.
MEM_ARG=""; [[ -n "$MEM" ]] && MEM_ARG="--mem=$MEM"
echo "[submit] 8 cores${MEM:+ / $MEM} -- est. ~3-6 BU for a 0.5 h run"
JOBID=$(ssh "$REMOTE" "bash -lc '
  cd $WORK &&
  export SBATCH_ACCOUNT=$PROJ &&
  sbatch --parsable $MEM_ARG \
    --export=ALL,WORK=$WORK,HPA_TIER=$TIER,HPA_TF_SET=$TF_SET,HPA_CSC_PROJECT=$PROJ,STAGES=\"$STAGES\" \
    $WORK/hpc/pipeline.slurm
'")
[[ -n "$JOBID" ]] || { echo "[submit] sbatch returned no job id" >&2; exit 1; }
echo "[submit] job $JOBID"
echo "  watch : ssh $REMOTE 'squeue -j $JOBID'"
echo "  log   : ssh $REMOTE 'tail -f $WORK/logs/hpa-pipeline-$JOBID.out'"
echo "  usage : ssh $REMOTE 'seff $JOBID'   # after it finishes, to right-size --mem"
