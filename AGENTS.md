# AGENTS.md

Orientation for AI coding agents (and humans) arriving at this repo: what it is, how to run it, and the conventions worth knowing before making changes.

## What this is
Human Promoter Atlas: the Streamlit viewer for a canonical-promoter analysis of TF ChIP-seq binding at the TSSs of all canonical protein-coding transcripts in the human genome (Ensembl GRCh38.114), including the per-gene regulatory modules and NMF programs derived from chip-atlas data. Nine top-nav tabs explore the same underlying peak calls at different scales (genome-wide aggregate, programs/modules, gene archetypes, GO reverse-lookup, per-transcript, compare, per-TF, TF co-binding network, methods). This repo is the viewer only; the upstream pipeline that produces its inputs lives elsewhere.

## Stack & layout
- **App**: Streamlit + Plotly, single-language Python. Entry: `app/streamlit_app.py` (nav, query-param seeding, footer). One module per tab under `app/tabs/`; shared helpers in `app/lib/` (`db.py` cached DuckDB connection + query helpers, `plotting.py` figure builders + palette, `nav.py`, `ui.py`).
- **Data**: a DuckDB single-file analytical database plus parquet shards, all built offline by the scripts in `data/` (`build_app_db.py`, `build_depmap_*.py`, `build_tf_pair_table.py`). Built artifacts are gitignored.
- **Deploy**: a single Docker container (`app/Dockerfile`, `deploy/docker-compose.yml`) that binds to localhost only (`127.0.0.1:8501`) behind a reverse proxy.
- Requires Python >= 3.11 (`pyproject.toml`); runtime deps pinned in `app/requirements.txt`.

## Run, test, lint
```bash
pip install -e .          # or: pip install -e ".[dev]"  (adds ruff + pytest)
make db                   # build data/canonical_promoter.duckdb from the upstream analyses (one-time)
make dev                  # streamlit on :8501
ruff check app/           # lint (line-length 100)
```
The data build reads raw inputs from `HPA_ANALYSIS_DIR` (chip-atlas analysis outputs) and `HPA_DEPMAP_RAW` (raw DepMap CSVs); both default under `data/raw/` and are overridable via env var. Docker: `make up` / `make down` (recipe in `DEPLOY.md`).

## Conventions
- URL deep-linking is a first-class feature: query params like `?gene=GAPDH&tf=CTCF&program=7` seed the state on load, and selectbox changes write back to the URL so every view is shareable. Preserve this round-trip when adding controls.
- All DuckDB access goes through the cached helpers in `app/lib/db.py`; per-TSS / per-TF queries rely on a single indexed `WHERE` hit. Add queries there rather than opening ad-hoc connections.
- Runtime deps are pinned in `app/requirements.txt` (baked into the image); `pyproject.toml` carries looser floors for local dev. Keep the two in sync when changing a dependency.

## The pipeline (`pipeline/`, `acquire/`, `hpc/`)
Added 2026-08-01. The upstream analysis used to live outside the repo with hardcoded `/home/...` paths; it is now here, with every machine-specific path resolved by `pipeline/config.py` from a gitignored `.env` (copy `.env.example`). Print the resolved config with `python pipeline/config.py`.

A build is identified by three axes — `HPA_TIER` (ChIP-Atlas significance tier), `HPA_TF_SET` (`whitelist` 1,311 / `all` 1,793) and `HPA_MIN_SCORE_ASSIGN` (default 250) — and lands in `<root>/<tier>.<tf_set>.s<score>/` with a `_BUILD.json` stamp. Heavy stages can run on CSC Roihu via `hpc/` (`01_stage.sh` → `02_submit.sh` → `03_fetch.sh`).

## Gotchas
- The viewer does not regenerate the data layer at runtime: it reads a prebuilt `data/` mounted read-only. Re-run the `data/build_*.py` scripts (or `make db`) after any upstream change, or the app serves stale or empty data.
- **Never rebuild a build path outside `config.py`.** Two SLURM scripts duplicated the `<tier>.<tf_set>` rule and silently diverged when the score joined the name: `.done` markers went to one directory and data to another, so a rerun found a stale marker and **skipped the stage it was asked to redo, exiting 0 in nine seconds**. Ask config: `python -c "import config; print(config.OUT_DN)"`.
- **The tier and `MIN_SCORE_ASSIGN` are the same quantity.** ChIP-Atlas score is −10·log₁₀(Q) capped at 1000, so `score ≥ 500` *is* `Q < 1E-50`. On the old Q<1E-50 input the ≥500 filter was therefore inert, while the Methods page described it as a working two-tier design. See `docs/threshold-calibration.md`.
- **NMF here can collapse.** `init="random"` with `solver="mu"` drives a component to zero on some seeds, giving `reconstruction_err_=nan` and an ARI of exactly 0.000. `select_k` detects and re-seeds such fits; `tss_modules`, `tss_modules_k10` and `tss_archetypes` still use `init="random"` and are safe only because they pin seed 0.
- **BLAS threads are set per stage, not globally.** Pool stages (aggregate, modules, programs, multibin) need one thread; solver stages (selectk, k10, archetypes) need all of them. Setting it in a pool initialiser does *not* work — OpenBLAS reads the variable at import, which has already happened in the parent.
- `os.cpu_count()` reports the whole node on HPC (384 on a Roihu CPU node), not the cgroup. Worker counts read `SLURM_CPUS_PER_TASK` first and cap at 12 locally.
- Behind a reverse proxy, Streamlit needs its WebSocket route (`/_stcore/stream`) proxied as well; without it the page loads but the spinner never resolves (see `DEPLOY.md`).
- The production container runs read-only with a small `/tmp` tmpfs and drops all Linux capabilities, so anything that tries to write to disk at runtime will fail by design.
