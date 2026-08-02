#!/usr/bin/env bash
# Build the Python environment for the Human Promoter Atlas pipeline on CSC Roihu.
#
# WHY A PLAIN VENV, not tykky / not a module:
#   Roihu has NO python-data module and NO tykky (verified 2026-07-31: `module
#   -t spider` lists no python at all, and `module` is not even on PATH until
#   /appl/profile/zz-csc-env.sh is sourced). The system interpreter is Python
#   3.9.25 and Roihu CPU nodes are x86_64, so manylinux wheels apply and a plain
#   venv in /projappl installs in a couple of minutes with no compilation.
#
# PYTHON 3.9 PINS -- do not "upgrade" these casually:
#   pyranges==0.1.4  the pipeline uses the 0.x API (pr.PyRanges(df), .merge(),
#                    .join()). pyranges 1.x rewrote that API AND requires >=3.10.
#                    Its compiled deps (ncls, sorted_nearest) must resolve to
#                    cp39 manylinux wheels -- this script fails loudly if pip
#                    falls back to building from source.
#   numpy<2          numpy 2.1+ requires >=3.10, and the local build used 2.2.
#                    Cross-version float drift is why the reference baseline is
#                    re-established ON Roihu rather than compared to the laptop.
#
# Run it as (module/env needs a LOGIN shell -- `ssh host 'cmd'` has no module
# function, and `bash -l < script` does not get one either; only `bash -lc`):
#   ssh roihu 'bash -lc "$(cat)"' < hpc/00_build_env.sh
set -uo pipefail

# The CSC env script is not `set -u` clean, so source it BEFORE tightening.
source /appl/profile/zz-csc-env.sh 2>/dev/null || true
set -u

PROJ="${HPA_CSC_PROJECT:-}"
[[ -n "$PROJ" ]] || { echo "need HPA_CSC_PROJECT (e.g. project_0000000)" >&2; exit 2; }
VENV="${HPA_VENV:-/projappl/$PROJ/hpa_venv}"

PKGS=(
  "numpy<2"
  "pandas"
  "scipy"
  "scikit-learn"
  "pyranges==0.1.4"
  "pyarrow"
  "matplotlib"
  "seaborn"
)
# Deliberately NOT installed: umap-learn (drags in numba/llvmlite, the worst
# compile risk on 3.9). cluster_tfs*.py are the only users and they stay local.

echo "[hpa-env] project=$PROJ venv=$VENV"
echo "[hpa-env] python: $(python3 --version 2>&1)"

if [[ -x "$VENV/bin/python" ]] && \
   "$VENV/bin/python" -c "import pyranges,sklearn,pandas,numpy,scipy,pyarrow" 2>/dev/null; then
  echo "[hpa-env] venv already good"
  "$VENV/bin/python" - <<'PY'
import numpy, pandas, scipy, sklearn, pyranges, pyarrow, sys
print(f"  python       {sys.version.split()[0]}")
for m in (numpy, pandas, scipy, sklearn, pyranges, pyarrow):
    print(f"  {m.__name__:<12} {getattr(m, '__version__', '?')}")
PY
  exit 0
fi

mkdir -p "$(dirname "$VENV")"
python3 -m venv "$VENV" || { echo "[hpa-env] venv creation FAILED" >&2; exit 1; }
"$VENV/bin/pip" install --quiet --upgrade pip wheel

# --only-binary on the compiled deps: if no cp39 wheel exists we want a loud
# failure here, not a silent 10-minute source build on a login node.
echo "[hpa-env] installing (wheels only for compiled deps)..."
"$VENV/bin/pip" install --quiet --only-binary=ncls,sorted_nearest,numpy,scipy,scikit-learn,pyarrow \
    "${PKGS[@]}" || { echo "[hpa-env] pip install FAILED" >&2; exit 1; }

echo "[hpa-env] verifying imports..."
"$VENV/bin/python" - <<'PY'
import sys
import numpy, pandas, scipy, sklearn, pyranges, pyarrow, matplotlib, seaborn
print(f"  python       {sys.version.split()[0]}")
for m in (numpy, pandas, scipy, sklearn, pyranges, pyarrow, matplotlib, seaborn):
    print(f"  {m.__name__:<12} {getattr(m, '__version__', '?')}")

# Exercise the exact pyranges 0.x API the pipeline depends on, so an API break
# surfaces here rather than 20 minutes into a job.
import pandas as pd
a = pyranges.PyRanges(pd.DataFrame({
    "Chromosome": ["1", "1"], "Start": [100, 400], "End": [200, 500]}))
b = pyranges.PyRanges(pd.DataFrame({
    "Chromosome": ["1"], "Start": [150], "End": [450]}))
assert len(a.merge()) == 2, "pyranges .merge() misbehaving"
j = a.join(b, suffix="_w")
assert len(j) == 2, f"pyranges .join() returned {len(j)}, expected 2"
print("  pyranges 0.x API (PyRanges/.merge/.join): OK")
PY
echo "[hpa-env] done -> $VENV"
