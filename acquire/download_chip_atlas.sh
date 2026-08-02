#!/usr/bin/env bash
# Fetch ChIP-Atlas per-antigen peak BEDs at a given significance tier, strip them
# to 7 columns, and store them per-TF as <symbol>.bed.gz.
#
# TIER NUMBERING TRAP -- read this before changing anything:
#   ChIP-Atlas uses TWO numberings for the SAME thresholds, differing by 10x.
#     bulk filename infix : 05  10  20  50     (= Q < 1E-0N; 05 is LOOSEST)
#     website picker      : 50 100 500 1000    (= -10*log10(Q))
#   This script takes the FILENAME infix. --tier 05 means Q<1E-05.
#
# The TF list is a TSV of  antigen[TAB]output_symbol.  The second column exists
# because many ChIP-Atlas antigens are legacy gene symbols: the file is served as
# Oth.ALL.05.MKL1.AllCell.bed but the current symbol is MRTFA. We fetch under the
# antigen name and store -- and label column 4 -- under the current symbol, so
# nothing downstream has to know about the rename.
# Generate such a list from chip-atlas/00.data/TF/chip_atlas_antigen_classification.tsv.
#
# Usage:
#   ./download_chip_atlas.sh --tier 05 --list download_order.txt --out /path/per_TF/q1e-5
#   ./download_chip_atlas.sh --tier 05 --list recover.tsv --out ... --jobs 4
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BASE="https://chip-atlas.dbcls.jp/data/hg38/assembled"
TIER=""; LIST=""; OUT=""; JOBS=6; DRY=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --tier) TIER="$2"; shift 2 ;;
    --list) LIST="$2"; shift 2 ;;
    --out)  OUT="$2";  shift 2 ;;
    --jobs) JOBS="$2"; shift 2 ;;
    --dry-run) DRY=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

# Plain tests, not ${VAR:?msg} -- a '}' inside the message closes the expansion
# early and bash does not flag it (SE-CaCTS gotcha 49).
[[ -n "$TIER" ]] || { echo "need --tier (05|10|20|50)" >&2; exit 2; }
[[ -n "$LIST" ]] || { echo "need --list <tsv>" >&2; exit 2; }
[[ -n "$OUT"  ]] || { echo "need --out <dir>" >&2; exit 2; }
case "$TIER" in 05|10|20|50) ;; *) echo "bad --tier '$TIER' (want 05|10|20|50)" >&2; exit 2 ;; esac
[[ -f "$LIST" ]] || { echo "no such list: $LIST" >&2; exit 2; }
[[ -f "$HERE/strip7.pl" ]] || { echo "missing $HERE/strip7.pl" >&2; exit 2; }

mkdir -p "$OUT"
n=$(grep -cvE '^\s*(#|$)' "$LIST")
echo "[chip-atlas] tier=$TIER (Q<1E-${TIER#0}) list=$LIST n=$n out=$OUT jobs=$JOBS"
[[ "$DRY" == "1" ]] && { echo "(dry run)"; exit 0; }

export BASE TIER OUT HERE

grep -vE '^\s*(#|$)' "$LIST" | xargs -P "$JOBS" -I{} bash -c '
  line="$1"
  antigen="${line%%$'"'"'\t'"'"'*}"
  symbol="${line#*$'"'"'\t'"'"'}"
  [[ "$symbol" == "$line" ]] && symbol="$antigen"   # single-column list
  [[ -n "$antigen" ]] || exit 0
  out="$OUT/${symbol}.bed.gz"; tmp="${out}.tmp.$$"

  # Resume: a non-empty, readable gzip is done.
  if [[ -s "$out" ]] && zcat "$out" 2>/dev/null | head -c1 | grep -q .; then
    echo "skip $symbol"; exit 0
  fi

  url="$BASE/Oth.ALL.${TIER}.${antigen}.AllCell.bed"
  for attempt in 1 2 3; do
    if curl -sf --retry 2 --max-time 10800 "$url" 2>/dev/null \
         | perl "$HERE/strip7.pl" "$symbol" | gzip > "$tmp"; then
      if zcat "$tmp" 2>/dev/null | head -c1 | grep -q .; then
        mv "$tmp" "$out"
        note=""; [[ "$symbol" != "$antigen" ]] && note=" (antigen $antigen)"
        echo "OK   $symbol$note $(zcat "$out" | wc -l) peaks, $(du -h "$out" | cut -f1)"
        exit 0
      fi
    fi
    rm -f "$tmp"; [[ "$attempt" -lt 3 ]] && sleep 4
  done
  # A 404 is legitimate: the antigen may have no peaks at this tier.
  echo "FAIL $symbol$([[ "$symbol" != "$antigen" ]] && echo " (antigen $antigen)")"
' _ {}

echo "[chip-atlas] done. files in $OUT: $(ls -1 "$OUT"/*.bed.gz 2>/dev/null | wc -l)"
