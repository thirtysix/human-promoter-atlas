#!/bin/bash
# Build a single durable, checksummed zstd archive of one ChIP-Atlas tier.
#
# WHY ONE FILE: 1,793 per-TF BEDs are awkward to move, verify and cite. One
# stream is a single checksum, a single transfer, and a citable artifact
# for a data deposit. Nothing is lost -- the TF is column 4 of every row, and the
# q1e-5 files also carry SRX (col 6) and cell group (col 7).
#
# WHY zstd -19 RATHER THAN gzip OR A FASTER LEVEL: measured on 60 real per-TF
# files (1.05 GiB uncompressed), 8 threads:
#
#     level   ratio    time    smaller than the per-TF gzip
#      -9     5.06x    9.2s              4%
#     -12     5.05x   17.6s              4%
#     -15     5.65x   56.0s             14%
#     -19     6.69x  255.5s             27%
#
# -19 costs ~28x the time of -9 for ~24% better compression. That is the right
# trade ONLY because this is a write-once, read-many durable artifact: the hour
# is paid once and saves ~0.9 GiB on every copy and transfer thereafter. Drop to
# -12 with ZSTD_LEVEL if you are iterating on the archive itself.
#
# NOTE this level, not the sort, is what makes the job take ~an hour. Removing
# the global sort (below) was still correct -- it was redundant work and it
# spilled to Lustre -- but it saves ~5-8 min of ~55, not the order of magnitude
# first assumed. Measure before attributing cost here.
#
# (ZSTD_LEVEL is set below the #SBATCH block, not here: sbatch stops reading
# directives at the first non-comment line, so an assignment above them silently
# drops every #SBATCH that follows and the job reverts to partition/time/mem
# defaults.)
#
# WHY IN A JOB, NOT ON THE LOGIN NODE: this is heavy sustained CPU and IO, which
# CSC etiquette keeps off shared login nodes.
#
# DETERMINISM WITHOUT SORTING: the archive used to pipe everything through a
# global `sort`, which cost 3609s of a 1h job and spilled to Lustre. It was
# unnecessary. Each per-TF BED is already position-sorted, and concatenating the
# files in LC_ALL=C name order gives TF-major, position-ordered output directly.
#
# Verified: against a globally sorted build the row multiset is IDENTICAL (same
# md5 once both are fully sorted) and the concatenation is byte-stable across
# repeats. The two differ only in how they break ties -- rows sharing
# (chrom,start,end) but from different experiments -- because GNU sort falls back
# to whole-line comparison while the files keep their native experiment order.
# Determinism was the only thing the sort provided, and concatenation provides it
# for free, so archives built this way are NOT byte-identical to the earlier one
# but contain exactly the same data.
#SBATCH --partition=small
#SBATCH --job-name=hpa-archive
#SBATCH --time=03:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /appl/profile/zz-csc-env.sh 2>/dev/null || true
set -uo pipefail

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
TIER="${HPA_TIER:-q1e-5}"
NCPU="${SLURM_CPUS_PER_TASK:-8}"
ZSTD_LEVEL="${ZSTD_LEVEL:-19}"   # see the table at the top before changing
SRC="$WORK/data/chip-atlas/per_TF/$TIER"
OUT="$WORK/archive"
INFIX=""
case "$TIER" in
  q1e-50) INFIX=50 ;; q1e-20) INFIX=20 ;; q1e-10) INFIX=10 ;; q1e-5) INFIX=05 ;;
  *) echo "FATAL: unknown tier $TIER" >&2; exit 1 ;;
esac

[[ -d "$SRC" ]] || { echo "FATAL: no $SRC" >&2; exit 1; }
n_bed=$(ls -1 "$SRC"/*.bed.gz 2>/dev/null | wc -l)
[[ "$n_bed" -ge 1000 ]] || { echo "FATAL: only $n_bed beds in $SRC" >&2; exit 1; }
command -v zstd >/dev/null || { echo "FATAL: zstd not on PATH" >&2; exit 1; }

mkdir -p "$OUT"
BASE="chip_atlas.hg38.Oth.ALL.${INFIX}.tf${n_bed}"
BED="$OUT/$BASE.bed.zst"
TMP="$BED.tmp.$$"

echo "=== hpa-archive ${SLURM_JOB_ID:-local} ==="
echo "  tier=$TIER infix=$INFIX beds=$n_bed cpus=$NCPU"
echo "  src=$SRC"
echo "  out=$BED"

t0=$SECONDS

# Guard the assumption this design rests on: if the inputs ever stop being
# position-sorted, the archive silently stops being ordered. Checking a sample is
# cheap; checking all 1,793 is not.
echo "[check] input files position-sorted?"
bad=0
for f in $(ls -1 "$SRC"/*.bed.gz | head -20); do
  # -s is REQUIRED, not cosmetic. Without it GNU sort applies a last-resort
  # whole-line comparison to rows whose keys tie, so every file with two peaks
  # sharing (chrom,start) from different experiments is reported as unsorted --
  # which is most of them, and is exactly the tie behaviour we already decided
  # is harmless. -s disables that and checks the key alone.
  zcat "$f" | LC_ALL=C sort -s -k1,1 -k2,2n -c 2>/dev/null \
    || { echo "  NOT SORTED: $(basename "$f")" >&2; bad=1; }
done
[[ $bad -eq 0 ]] || { echo "FATAL: inputs are not position-sorted; restore the sort" >&2; exit 1; }
echo "  sample of 20 OK"

# LC_ALL=C fixes the collation so file order -- and therefore the archive and its
# checksum -- does not depend on the machine's locale.
export LC_ALL=C
# find|sort|xargs rather than a shell glob: 1,793 paths is fine today but a
# glob is one tier away from ARG_MAX, and it would fail loudly mid-build.
find "$SRC" -name '*.bed.gz' | sort | xargs zcat \
  | zstd -"$ZSTD_LEVEL" -T"$NCPU" --long=27 -q -o "$TMP"
rc=$?
[[ $rc -eq 0 ]] || { echo "FATAL: build failed rc=$rc" >&2; rm -f "$TMP"; exit 1; }

# Verify the archive decompresses and round-trips to the same row count before
# publishing it under its real name.
echo "[verify] decompressing..."
n_arch=$(zstd -dc "$TMP" | wc -l)
n_src=$(zcat "$SRC"/*.bed.gz | wc -l)
echo "  rows: archive=$n_arch source=$n_src"
[[ "$n_arch" -eq "$n_src" ]] || { echo "FATAL: row count mismatch" >&2; rm -f "$TMP"; exit 1; }

mv "$TMP" "$BED"
( cd "$OUT" && sha256sum "$(basename "$BED")" > "$BASE.sha256" )

raw_bytes=$(zstd -dc "$BED" | wc -c)
arc_bytes=$(stat -c%s "$BED")
gz_bytes=$(du -cb "$SRC"/*.bed.gz | tail -1 | cut -f1)

cat > "$OUT/$BASE.manifest.json" <<JSON
{
  "dataset": "ChIP-Atlas hg38 TF-class peaks, one file per antigen, concatenated",
  "tier": "$TIER",
  "chip_atlas_filename_infix": "$INFIX",
  "qvalue_threshold": "Q < 1E-${INFIX#0}",
  "note": "ChIP-Atlas uses two numberings 10x apart: bulk filenames 05/10/20/50 vs website picker 50/100/500/1000. This records the FILENAME infix.",
  "n_antigen_files": $n_bed,
  "n_rows": $n_arch,
  "columns": ["chrom", "start", "end", "tf_symbol", "score", "srx", "cell_group"],
  "sort_order": "TF-major (files concatenated in LC_ALL=C name order), each file in its native position-sorted order; ties among rows sharing chrom/start/end keep experiment order",
  "score_semantics": "-10*log10(Q), capped at 1000",
  "source_url_template": "https://chip-atlas.dbcls.jp/data/hg38/assembled/Oth.ALL.<infix>.<antigen>.AllCell.bed",
  "symbols": "current GRCh38 symbols; legacy antigen names remapped per chip_atlas_antigen_classification.tsv",
  "bytes_uncompressed": $raw_bytes,
  "zstd_level": $ZSTD_LEVEL,
  "bytes_compressed": $arc_bytes,
  "bytes_as_per_tf_gzip": $gz_bytes,
  "built_by": "hpc/archive.slurm",
  "code_sha": "$(cat "$WORK/code/GITSHA" 2>/dev/null || echo unknown)",
  "slurm_job": "${SLURM_JOB_ID:-local}"
}
JSON

echo "=== done in $((SECONDS - t0))s ==="
python3 - <<PY 2>/dev/null || true
raw, arc, gz = $raw_bytes, $arc_bytes, $gz_bytes
print(f"  uncompressed   {raw/2**30:8.2f} GiB")
print(f"  per-TF gzip    {gz/2**30:8.2f} GiB  ({raw/gz:.2f}x)")
print(f"  archive zstd$ZSTD_LEVEL {arc/2**30:8.2f} GiB  ({raw/arc:.2f}x)   {100*(gz-arc)/gz:.0f}% smaller than gzip")
PY
ls -la "$OUT"
