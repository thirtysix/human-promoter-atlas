#!/bin/bash
# Decide how select_k should initialise and solve NMF.
#
# The 20-seed ARI is meant to answer "does rank k give a stable partition when
# fitted from independent starting points". Two candidate settings each break it
# in an opposite way, both measured on the 1,793-TF matrix:
#
#   init=random,   solver=mu : genuine seed diversity, but the multiplicative
#                              solver collapses a component on some seeds
#                              (1/20 at k=18 fell to a SINGLE component), giving
#                              reconstruction_err_=nan and ARI floors of exactly
#                              0.000. Stability scores partly measure solver
#                              failure. -> ARI k=5 0.384, k=8 0.509
#   init=nndsvdar, solver=mu : no collapse, but every seed starts from nearly the
#                              same NNDSVD point and converges to nearly the same
#                              answer. -> ARI 0.981/0.984, i.e. the metric now
#                              measures init determinism, not the data.
#
# This probes the remaining options at the single worst rank (k=18):
#   A random+mu   (baseline, expect collapses)
#   B random+cd   (coordinate descent is less prone to zeroing a component)
#   C nndsvdar+mu (baseline, expect ~0.98 ARI)
#   D random+mu with collapsed fits REJECTED and retried on a fresh seed
#
# What we want: no collapses AND ARI well below 1.0 with real spread.
#SBATCH --partition=small
#SBATCH --job-name=hpa-nmfprobe
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

source /appl/profile/zz-csc-env.sh 2>/dev/null || true
set -uo pipefail

WORK="${WORK:-$SLURM_SUBMIT_DIR}"
VENV="${HPA_VENV:-/projappl/${HPA_CSC_PROJECT:?}/hpa_venv}"
PY="${PY:-$VENV/bin/python}"
NPZ="${NPZ:-$WORK/out/q1e-5.all.s250/tss_modules/occupancy.modules.npz}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}" PYTHONUNBUFFERED=1

"$PY" - "$NPZ" <<'PY'
import sys, itertools, warnings
import numpy as np, scipy.sparse as sp
from sklearn.decomposition import NMF
from sklearn.metrics import adjusted_rand_score
warnings.filterwarnings("ignore")

M = sp.load_npz(sys.argv[1]).tocsr()
K = 18
N = 20
print(f"matrix {M.shape} nnz={M.nnz:,}  k={K} seeds={N}\n", flush=True)


def fit(k, seed, init, solver):
    m = NMF(n_components=k, init=init, solver=solver, beta_loss="frobenius",
            max_iter=250, random_state=seed, tol=1e-4)
    W = m.fit_transform(M)
    dom = W.argmax(axis=1)
    collapsed = len(np.unique(dom)) < k or not np.isfinite(m.reconstruction_err_)
    return dom, m.reconstruction_err_, collapsed


def summarise(name, labels, n_collapse, n_retry=0):
    aris = [adjusted_rand_score(a, b) for a, b in itertools.combinations(labels, 2)]
    q = np.percentile(aris, [25, 50, 75])
    print(f"  {name:<26} collapses={n_collapse:<3} retries={n_retry:<3} "
          f"ARI med={q[1]:.3f} IQR=[{q[0]:.3f}, {q[2]:.3f}] min={min(aris):.3f}",
          flush=True)


for name, init, solver in (("A random+mu", "random", "mu"),
                           ("B random+cd", "random", "cd"),
                           ("C nndsvdar+mu", "nndsvdar", "mu")):
    labels, nc = [], 0
    for s in range(N):
        dom, err, col = fit(K, s, init, solver)
        nc += col
        labels.append(dom)
    summarise(name, labels, nc)

# D: random+mu, but a collapsed fit is discarded and retried on a fresh seed.
# Keeps genuine seed diversity while excluding solver failures from the metric.
labels, nc, retries, seed = [], 0, 0, 0
while len(labels) < N and seed < 200:
    dom, err, col = fit(K, seed, "random", "mu")
    seed += 1
    if col:
        nc += 1; retries += 1; continue
    labels.append(dom)
summarise("D random+mu, reject+retry", labels, nc, retries)
PY
