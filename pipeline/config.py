"""
Single source of truth for every machine-specific path and build parameter the
upstream pipeline needs.

Extends the ``HPA_*`` environment-variable convention already used by
``data/build_app_db.py`` and ``app/lib/db.py``, and adds the two axes that
define a build:

    HPA_TIER    which ChIP-Atlas significance tier the peaks come from
    HPA_TF_SET  which transcription-factor axis to build on

Values come from the process environment, falling back to a ``.env`` file at
the repo root (gitignored -- see ``.env.example``). No machine-specific path
has a silent default: :func:`require` raises with the variable name and a
pointer to ``.env.example`` rather than letting an unset value quietly become
``Path(".")``.

Usage from a pipeline script (they run with ``pipeline/`` on ``sys.path``,
so a plain import works)::

    from config import GTF_FN, PER_TF_DN, OUT_DN, WORKERS, tf_name_set
"""

from __future__ import annotations

import json
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
_DOTENV = REPO_DIR / ".env"


################################################################################
# .env loading #################################################################
################################################################################
def _load_dotenv(path: Path = _DOTENV) -> dict:
    """Minimal KEY=VALUE reader. No dependency, no interpolation, no export.

    Real environment variables always win, so a one-off run can override the
    file without editing it.
    """
    vals: dict[str, str] = {}
    if not path.is_file():
        return vals
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        val = val.strip().strip('"').strip("'")
        vals[key.strip()] = val
    return vals


_FILE_ENV = _load_dotenv()


def optional(name: str, default: str | None = None) -> str | None:
    """Environment first, then .env, then the supplied default."""
    val = os.environ.get(name) or _FILE_ENV.get(name)
    return val if val else default


def require(name: str) -> str:
    """Like :func:`optional` but fails loudly, naming the fix."""
    val = optional(name)
    if not val:
        raise SystemExit(
            f"config: {name} is not set.\n"
            f"  Set it in {_DOTENV} (copy {REPO_DIR / '.env.example'}) "
            f"or export it for this run."
        )
    return val


################################################################################
# Build axes ###################################################################
################################################################################
# ChIP-Atlas ships two different numberings for the SAME thresholds and they
# differ by 10x. This mapping is the only place that fact should ever live:
#
#   directory / tier name   bulk filename infix   website picker   meaning
#   q1e-50                  50                    1000            Q < 1E-50  (strictest)
#   q1e-20                  20                    500             Q < 1E-20
#   q1e-10                  10                    100             Q < 1E-10
#   q1e-5                   05                    50              Q < 1E-05  (loosest)
TIER_TO_FILE_INFIX = {"q1e-50": "50", "q1e-20": "20", "q1e-10": "10", "q1e-5": "05"}
TIER_TO_QVALUE = {"q1e-50": 1e-50, "q1e-20": 1e-20, "q1e-10": 1e-10, "q1e-5": 1e-5}

TIER = optional("HPA_TIER", "q1e-50")
if TIER not in TIER_TO_FILE_INFIX:
    raise SystemExit(
        f"config: HPA_TIER={TIER!r} is not one of {sorted(TIER_TO_FILE_INFIX)}"
    )

# whitelist -> the curated DNA_binding_genes.tsv axis (1,304 TFs, the original
#              build). all -> every downloaded antigen that is a real GRCh38
#              gene symbol (1,767 TFs); see acquire/build_tf_list.py.
TF_SETS = ("whitelist", "all")
TF_SET = optional("HPA_TF_SET", "whitelist")
if TF_SET not in TF_SETS:
    raise SystemExit(f"config: HPA_TF_SET={TF_SET!r} is not one of {list(TF_SETS)}")

# Score required for a TF to be ASSIGNED to a module. This is the same quantity
# as the tier -- ChIP-Atlas score is -10*log10(Q) capped at 1000, so score >= 500
# is exactly Q < 1E-50 -- which makes the pipeline a TWO-threshold design:
#
#   discovery  : every peak in the tier file (HPA_TIER)
#   assignment : only peaks at or above this score
#
# Historically both were pinned to Q<1E-50 by accident: the input was the q1e-50
# tier, so score>=500 excluded nothing and the filter was inert. On a looser tier
# it bites hard -- at q1e-5 it discards 81% of in-window peaks and drops ~15% of
# discovered modules entirely for having no TF above the bar.
#
# The default of 250 (Q < 1E-25) was CHOSEN, not inherited, on three independent
# criteria measured on the q1e-5 build (see docs/threshold-calibration.md):
#
#   reproducibility  Split-half replication across disjoint experiment sets is
#                    flat in the threshold (mean 0.676-0.680 over scores 50-500),
#                    so 500 buys no precision -- it only discards evidence.
#   specificity      GO-BP programs are more specific at 250 than 500 (median
#                    odds ratio 1.81 -> 1.99 at k=20; reliance on generic
#                    >=1000-gene GO sets 73% -> 57%).
#   plausibility     Removing the filter entirely scores BEST on GO but assigns a
#                    median 38 TFs per ~177 bp module, with 16.6% of modules
#                    above 200 TFs and one credited to 1,194 of 1,311 factors.
#                    GO cannot see that failure mode -- it scores program gene
#                    sets, which only improve as assignments grow. At 250 the
#                    median is 13, matching the published q1e-50 atlas's 12.
MIN_SCORE_ASSIGN = int(optional("HPA_MIN_SCORE_ASSIGN", "250"))
if not 0 <= MIN_SCORE_ASSIGN <= 1000:
    raise SystemExit(
        f"config: HPA_MIN_SCORE_ASSIGN={MIN_SCORE_ASSIGN} out of range 0-1000 "
        f"(score is -10*log10(Q), capped at 1000)")


################################################################################
# Paths ########################################################################
################################################################################
CHIP_ATLAS_DN = Path(require("HPA_CHIP_ATLAS_DIR"))

# Peaks live in a per-tier subdirectory. Deriving this from TIER (rather than
# letting it be set independently) is what makes it structurally impossible to
# label a q1e-5 build as q1e-50.
PER_TF_DN = Path(optional("HPA_PER_TF_DIR") or CHIP_ATLAS_DN / "per_TF" / TIER)
if PER_TF_DN.name != TIER:
    raise SystemExit(
        f"config: HPA_PER_TF_DIR={PER_TF_DN} does not end in the configured "
        f"tier {TIER!r}. Refusing to run -- this is exactly how a build gets "
        f"mislabelled. Either fix the path or set HPA_TIER to match."
    )

GTF_FN = Path(require("HPA_GTF"))
DNA_BINDING_FN = Path(require("HPA_DNA_BINDING"))


class _MissingPath:
    """Placeholder for a path only some scripts need.

    Importing config must not demand every input: the HPC job runs the
    ChIP-Atlas half and has no MSigDB/GTEx/DepMap staged, while the enrichment
    scripts need MSigDB but no peaks. An unset value therefore stays inert until
    something actually uses it, then fails naming the variable -- rather than
    silently becoming Path(".") or forcing dummy values into the job script.
    """

    def __init__(self, var: str):
        self._var = var

    def _die(self, *_a, **_k):
        raise SystemExit(
            f"config: {self._var} is not set, but this script needs it.\n"
            f"  Set it in {_DOTENV} (see {REPO_DIR / '.env.example'}) or export it."
        )

    __truediv__ = __fspath__ = __enter__ = _die

    def exists(self, *_a, **_k):
        return False

    def __repr__(self):
        return f"<unset {self._var}>"

    __str__ = __repr__


def _path_or_missing(name: str):
    val = optional(name)
    return Path(val) if val else _MissingPath(name)


# Needed only by the enrichment / GTEx / DepMap stages, which never run on the
# HPC side. Left inert when unset so the peak pipeline imports cleanly there.
MSIGDB_FN = _path_or_missing("HPA_MSIGDB")
GTEX_DN = _path_or_missing("HPA_GTEX_DIR")
DEPMAP_DN = _path_or_missing("HPA_DEPMAP_DIR")

# Generated TF lists (acquire/build_tf_list.py). Only consulted for TF_SET="all".
TF_LIST_FN = Path(optional("HPA_TF_LIST") or REPO_DIR / "acquire" / f"tf_list.{TF_SET}.txt")

# Output root. Defaults to a per-build directory so two tiers/axes can coexist.
# The original flat layout is reproduced by pointing HPA_OUT_DIR straight at it.
# The score is ALWAYS in the build directory name. It used to be omitted at the
# then-default 500, which meant a directory's name silently depended on what the
# default happened to be -- so changing the default would have made existing
# q1e-5.whitelist (a score-500 build) read as the new default. Explicit is worth
# the one-off rename; _BUILD.json remains the authority either way.
_SCORE_SFX = f".s{MIN_SCORE_ASSIGN}"

# HPA_OUT_DIR fully determines the output path, so HPA_ANALYSIS_ROOT is only
# required when it is absent. Demanding both made one-off runs that write
# somewhere specific -- regression diffs, the split-half scans -- fail on a
# variable whose value they would never use.
_out_override = optional("HPA_OUT_DIR")
if _out_override:
    OUT_DN = Path(_out_override)
    ANALYSIS_ROOT = OUT_DN.parent
else:
    ANALYSIS_ROOT = Path(require("HPA_ANALYSIS_ROOT"))
    OUT_DN = ANALYSIS_ROOT / f"{TIER}.{TF_SET}{_SCORE_SFX}"

# Kept for scripts that referenced it directly.
TF_CLUSTER_FN = OUT_DN / "clustering" / "tf_cluster_table.tsv"


################################################################################
# Compute ######################################################################
################################################################################
def _default_workers() -> int:
    """SLURM allocation first, then a laptop-friendly cap.

    On an HPC node ``os.cpu_count()`` reports the whole machine (384 cores on a
    Roihu CPU node) while the cgroup owns only ``--cpus-per-task``; using
    cpu_count there would oversubscribe by ~50x. Locally we cap at 12 to leave
    headroom and limit sustained thermal load.
    """
    slurm = os.environ.get("SLURM_CPUS_PER_TASK")
    if slurm:
        return max(1, int(slurm))
    return max(1, min(12, (os.cpu_count() or 2) - 2))


WORKERS = max(1, int(optional("HPA_WORKERS") or _default_workers()))


def pin_blas_threads(n: int = 1) -> None:
    """Stop BLAS from oversubscribing inside multiprocessing workers.

    scikit-learn's NMF is BLAS-heavy. With N worker processes each spawning a
    thread per core, a 384-core node produces N*384 threads fighting over the
    cgroup's real allocation. Call this from the pool initializer; leave the
    single-process NMF stage unpinned so it can use the whole allocation.
    """
    for var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[var] = str(n)


################################################################################
# TF axis ######################################################################
################################################################################
def tf_name_set() -> set[str]:
    """The set of TF names this build is allowed to use.

    ``whitelist`` reads the curated DNA-binding gene table (the original 1,304
    axis). ``all`` reads the generated list of every downloaded antigen that is
    a real GRCh38 gene symbol (1,767).
    """
    if TF_SET == "whitelist":
        import pandas as pd  # local import: keeps config importable without pandas
        return set(pd.read_csv(DNA_BINDING_FN, sep="\t")["gene_name"].astype(str))
    if not TF_LIST_FN.is_file():
        raise SystemExit(
            f"config: HPA_TF_SET={TF_SET!r} needs {TF_LIST_FN}, which does not "
            f"exist. Generate it with:  python acquire/build_tf_list.py --tf-set {TF_SET}"
        )
    return {ln.strip() for ln in TF_LIST_FN.read_text().splitlines() if ln.strip()}


################################################################################
# Peak-file discovery ##########################################################
################################################################################
_RENAMES: dict[str, str] | None = None


def rename_map() -> dict[str, str]:
    """Legacy ChIP-Atlas antigen -> current GRCh38 symbol (cached).

    Read from chip_atlas_antigen_classification.tsv next to the peaks. Without
    it, a symbol-only match silently drops ~33 real factors whose antigen name
    is a legacy symbol (MKL1/MRTFA, WHSC1/NSD2, ARNTL/BMAL1, ...), because
    Ensembl's GTF carries no gene_synonym attribute.
    """
    global _RENAMES
    if _RENAMES is None:
        _RENAMES = {}
        # Prefer the copy beside the peaks (authoritative, regenerated whenever
        # the antigen set changes); fall back to the vendored copy so a fresh
        # clone works without the data tree present.
        name = "chip_atlas_antigen_classification.tsv"
        path = next((p for p in (CHIP_ATLAS_DN / name, REPO_DIR / "acquire" / name)
                     if p.is_file()), None)
        if path is not None:
            for ln in path.read_text().splitlines():
                if ln.startswith("#") or not ln.strip():
                    continue
                f = ln.split("\t")
                if len(f) >= 3 and f[1] == "renamed_gene" and f[2]:
                    _RENAMES[f[0]] = f[2]
    return _RENAMES


def discover_tf_files(per_tf_dn=None, tf_names: set[str] | None = None) -> list:
    """Map the peak directory to [(symbol, [paths]), ...].

    Two subtleties this exists to handle, both of which silently corrupt a build
    if ignored:

    1. A peak file may be named after a LEGACY antigen (ARNTL.bed.gz) while the
       axis uses the current symbol (BMAL1). Resolved via :func:`rename_map`.
    2. A symbol may be backed by MORE THAN ONE file, because ChIP-Atlas serves
       some factors under both the legacy and the current name as separate
       antigens holding different experiments (verified: zero shared SRX). Both
       must be read; taking either alone drops real data.

    Returns paths as a list per symbol, sorted for determinism.
    """
    per_tf_dn = Path(per_tf_dn or PER_TF_DN)
    files = sorted(per_tf_dn.glob("*.bed.gz"))
    # Fail loud on an empty/thin glob: the 2026-07-19 chip-atlas reorg moved
    # every BED into per_TF/<tier>/, so a stale path globs to ZERO files and the
    # run completes with an empty-but-plausible build instead of erroring.
    #
    # The floor is settable because some trees are legitimately smaller: the
    # split-half calibration only contains TFs with enough experiments to split
    # (~990 of 1,793). Lower it deliberately for those; never to 0, which would
    # reinstate the silent-empty-build bug this guard exists to prevent.
    min_beds = int(optional("HPA_MIN_BEDS", "1000"))
    if len(files) < min_beds:
        raise SystemExit(
            f"discover_tf_files: only {len(files)} *.bed.gz under {per_tf_dn} "
            f"(floor {min_beds}). Wrong tier directory, or set HPA_MIN_BEDS if "
            f"this tree is intentionally smaller."
        )
    if tf_names is None:
        tf_names = tf_name_set()
    ren = rename_map()

    by_symbol: dict[str, list[str]] = {}
    for fp in files:
        stem = fp.name[: -len(".bed.gz")]
        symbol = ren.get(stem, stem)
        if symbol in tf_names:
            by_symbol.setdefault(symbol, []).append(str(fp))
    return [(sym, sorted(paths)) for sym, paths in sorted(by_symbol.items())]


# Column layout is identical across tiers: BED9 (q1e-50) and the 7-column
# stripped form (q1e-5) agree on 0,1,2,4 = chrom, start, end, score.
PEAK_USECOLS = [0, 1, 2, 4]
PEAK_NAMES = ["Chromosome", "Start", "End", "score"]
# category + int16 rather than string + float32: measured 230.4 -> 42.1 MiB on
# MYC (5.5x) and slightly faster to parse. Exact, not lossy -- ChIP-Atlas score
# is -10*log10(Q) rounded to an integer and capped at 1000, verified integral
# and within 50..1000 on both tiers. A chromosome column of ~25 distinct values
# has no business being stored as one Python string per row when a 65.6M-row TF
# like CTCF is read in a single worker.
PEAK_DTYPE = {"Chromosome": "category", "Start": "int32",
              "End": "int32", "score": "int16"}


def normalize_chrom(s):
    """Strip a leading 'chr' and map mitochondrial M -> MT, to match GTF seqids.

    Category-preserving. The obvious implementation converts to string, which
    silently undoes the categorical read: measured 25.5 -> 128.6 MiB on GATA1,
    and CTCF is 30x larger again. Renaming the ~25 categories touches 25 values
    instead of 65 million rows.
    """
    import pandas as pd

    def fix(c: str) -> str:
        c = c[3:] if c.startswith("chr") else c
        return "MT" if c == "M" else c

    if isinstance(s.dtype, pd.CategoricalDtype):
        mapping = {c: fix(c) for c in s.cat.categories}
        if len(set(mapping.values())) == len(mapping):
            return s.cat.rename_categories(mapping)
        # Two categories collapsing to one (e.g. both 'chrM' and 'M' present)
        # is a real remap, which rename_categories refuses. Rare; take the slow
        # path rather than guess.
        return pd.Categorical([mapping[c] for c in s.astype(str)])
    s = s.astype("string").str.removeprefix("chr")
    return s.where(s != "M", "MT")


def read_peak_beds(paths):
    """Read one or many peak BEDs for a single TF into one DataFrame."""
    import pandas as pd
    if isinstance(paths, (str, Path)):
        paths = [paths]
    frames = [
        pd.read_csv(p, sep="\t", header=None, usecols=PEAK_USECOLS,
                    names=PEAK_NAMES, dtype=PEAK_DTYPE, compression="gzip")
        for p in paths
    ]
    if len(frames) == 1:
        return frames[0]
    # Categories can differ between a TF's files; union them so the concat stays
    # categorical instead of falling back to object.
    return pd.concat(frames, ignore_index=True)


################################################################################
# Build stamp ##################################################################
################################################################################
# Written into every output root so a downstream consumer can prove which build
# it is reading, instead of inferring it from a directory name someone renamed.
STAMP_NAME = "_BUILD.json"


def build_stamp(n_tf: int | None = None) -> dict:
    if n_tf is None:
        # Derive it rather than record null -- the TF count is the single most
        # useful field for telling two builds apart at a glance.
        try:
            n_tf = len(discover_tf_files())
        except SystemExit:
            n_tf = None
    return {
        "tier": TIER,
        "tier_file_infix": TIER_TO_FILE_INFIX[TIER],
        "qvalue": TIER_TO_QVALUE[TIER],
        "tf_set": TF_SET,
        "min_score_assign": MIN_SCORE_ASSIGN,
        "n_tf": n_tf,
        "per_tf_dn": str(PER_TF_DN),
    }


def write_build_stamp(out_dn: Path | None = None, n_tf: int | None = None) -> Path:
    out_dn = Path(out_dn or OUT_DN)
    out_dn.mkdir(parents=True, exist_ok=True)
    path = out_dn / STAMP_NAME
    path.write_text(json.dumps(build_stamp(n_tf), indent=2) + "\n")
    return path


def read_build_stamp(out_dn: Path | None = None) -> dict | None:
    path = Path(out_dn or OUT_DN) / STAMP_NAME
    return json.loads(path.read_text()) if path.is_file() else None


def assert_build_stamp_matches(out_dn: Path | None = None) -> dict:
    """Refuse to proceed when an output tree disagrees with the environment.

    This is the guard against the worst silent failure mode: packaging a q1e-5
    build while the manifest, the app and the Methods page all say q1e-50.
    """
    stamp = read_build_stamp(out_dn)
    if stamp is None:
        raise SystemExit(
            f"config: no {STAMP_NAME} in {out_dn or OUT_DN}. Re-run the pipeline "
            f"so the build is labelled, or write one deliberately."
        )
    # Every axis that defines a build must be checked. min_score_assign was
    # added late and was initially omitted here, which let a score-500 tree be
    # read as the current default -- the exact mislabelling this guard exists to
    # stop. Builds predating the field are rejected rather than assumed.
    mismatch = [
        f"{k}: stamp={stamp.get(k, '<absent>')!r} env={v!r}"
        for k, v in (("tier", TIER), ("tf_set", TF_SET),
                     ("min_score_assign", MIN_SCORE_ASSIGN))
        if stamp.get(k) != v
    ]
    if mismatch:
        raise SystemExit(
            "config: build stamp disagrees with the environment -- refusing to "
            "build a mislabelled artifact.\n  " + "\n  ".join(mismatch)
        )
    return stamp


def describe() -> str:
    return (f"tier={TIER} (Q<{TIER_TO_QVALUE[TIER]:g}, file infix "
            f"{TIER_TO_FILE_INFIX[TIER]}) tf_set={TF_SET} workers={WORKERS}")


if __name__ == "__main__":
    print(describe())
    for name in ("CHIP_ATLAS_DN", "PER_TF_DN", "OUT_DN", "GTF_FN",
                 "DNA_BINDING_FN", "TF_LIST_FN", "MSIGDB_FN", "GTEX_DN", "DEPMAP_DN"):
        val = globals()[name]
        print(f"  {name:<16} {val}  {'' if Path(val).exists() else '  [MISSING]'}")
