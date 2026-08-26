# Human Promoter Atlas

**Live at https://tfbss.org**

Interactive web companion to a canonical-promoter analysis: TF ChIP-seq
binding patterns at the TSSs of all canonical protein-coding transcripts
in the human genome (Ensembl GRCh38.114), plus the per-gene regulatory
**modules** and NMF **programs** discovered from chip-atlas data over
1,793 TFs and 19,745 TSSs.

This repo contains the **viewer**. The upstream analysis pipeline that
produces the inputs lives outside this repo; pointers to its expected
layout are in `data/build_app_db.py`.

## Tour

The viewer has nine top-nav tabs. Each addresses a different scale of
question — population-level, program-level, gene-level, TF-level — over
the same underlying ~12 million peak calls.

URL deep-linking is supported throughout: `?gene=GAPDH&tf=CTCF&program=7`
seeds the search state on load; selectbox changes write back to the URL
so every view is shareable.

---

### 1 · Aggregate — the genome-wide baseline

![Aggregate tab](docs/screenshots/01-aggregate.png)

> Mean binding profile of each TF across 19,745 canonical protein-coding
> promoters, transcription-oriented around the TSS at 0 bp. Establishes
> the reference everything else is interpreted against — for example,
> TBP peaking just upstream of the TSS confirms the canonical TATA-box
> position. Highlight individual TFs to compare any one factor to the
> crowd; toggle between binary occupancy and summed score.

---

### 2 · Programs & modules — recurring TF co-binding

![Programs tab](docs/screenshots/02-programs.png)

> A **module** is a local cluster of TF binding within a single promoter
> (±1.5 kb of its TSS). A **program** is one of 10 archetypal modules —
> discovered by NMF on the ~77,000-module × ~1,300-TF occupancy matrix —
> each with a recognizable biological signature (e.g. P5 cohesin, P7
> PIC, P1 chromatin downstream). For each program: top TFs by NMF H
> loading, position-density across the window, module-driver-class
> breakdown, and a full-atlas TF × tissue expression heatmap from GTEx.

---

### 3 · Program families — the vocabulary over the programs

![Program families tab](docs/screenshots/03-archetypes.png)

> The layer above programs: the 140 genome-wide programs grouped into
> **28 families** by how they CO-OCCUR across elements — do two programs
> mark the same places — with each family named by MSigDB enrichment and
> an FDR on every label. Grouping by shared TFs does not work here: NMF
> drives components onto disjoint factor sets, so PRC2 and PRC1.1 have a
> TF-loading cosine of 0.009 while marking the same locations.
>
> This replaced a gene-level archetype layer, which was tested and
> rejected: once genomic distance is controlled, gene identity carries no
> information about an element's program (same-gene element pairs score
> 0.2006 mean cosine against 0.2243 for distance-matched different-gene
> pairs).

---

### 4 · GO search — reverse lookup by biology

![GO search tab](docs/screenshots/04-go-search.png)

> Type a biological process (or pick from autocomplete over 981 terms
> that clear FDR in at least one family) and see every program family
> enriched for it, plus the family TFs that carried each hit. The
> inverse of the per-program GO view: instead of *"what does this
> program do?"*, it answers *"which programs implement this biology?"*

---

### 5 · Per-transcript — what's happening at one promoter

![Per-transcript tab](docs/screenshots/05-transcript.png)

> The full module decomposition of a single canonical promoter. The
> KDE curve is the smoothed density of TF binding within ±1.5 kb of the
> TSS; the vertical color bands behind it are the detected modules,
> each tinted by the program family it inherits from the genome layer.
> The header row reports how many programs are present and how many
> modules were detected. Below the promoter map (off-screen in this
> capture) sit per-TF rugs, GTEx
> tissue expression, DepMap CRISPR essentiality, and a TF–target
> essentiality coupling card.

---

### 6 · Compare — two transcripts side-by-side

![Compare tab](docs/screenshots/06-compare.png)

> Compare any two genes' promoter architecture on aligned coordinates:
> promoter maps, program presence diff (A-only / shared / B-only),
> paired GTEx tissue expression, and paired DepMap essentiality. Three
> curated pairs are prefilled as quick-starts (GAPDH vs IL6, TBP vs MYC,
> CDK4 vs RB1) to illustrate the comparisons the viewer is best at:
> housekeeper vs cytokine, general factor vs proliferation amplifier,
> kinase vs its substrate.

---

### 7 · Per-TF — what does this transcription factor do?

![Per-TF tab](docs/screenshots/07-per-tf.png)

> Everything the atlas knows about a single TF: aggregate binding
> profile (with optional cluster-mean overlay), loading on each genome
> program it reaches, TF-cluster membership at K=8 / K=12, DepMap CRISPR
> essentiality across lineages, GTEx tissue expression, top co-binding
> partners ranked by shared modules, and top bound TSSs. CTCF — shown
> here — appears in both P5 (cohesin near TSS) and P1 (chromatin
> downstream): same factor, two roles, separated cleanly by NMF.

---

### 8 · TF network — atlas-wide co-binding pairs

![TF network tab](docs/screenshots/08-tf-network.png)

> One row per unordered (TF A, TF B) pair, counting how many of the
> ~77,000 modules they share. 327,242 pairs over 1,064 unique TFs.
> Filter by search / minimum n_shared / minimum Jaccard / sort by lift
> (co-occurrence over independence). Surfaces TF cliques and obligate
> partnerships — MAX × MYC, KAT6A × KAT6B paralogs, CTCF × RAD21
> cohesin (87% of RAD21's modules co-bind CTCF), NFYA × SP1, ZFX × ZFY.

---

### 9 · Methods — provenance and reproducibility

![Methods tab](docs/screenshots/09-methods.png)

> Glossary, pipeline summary, parameters with rationale, verification
> anchors, limitations, citations. The exact dataset versions, parameter
> values, and scripts that produced every number in the rest of the app.

---

## Stack

- **Streamlit** + **Plotly** — single-language Python app.
- **DuckDB** — single-file analytical database; per-TSS / per-TF
  queries are sub-millisecond after a single `WHERE` index hit.
- **Docker** — single-container deploy behind your reverse proxy of
  choice.

## Repo layout

```
canonical-promoters/
├── README.md                       # this file
├── DEPLOY.md                       # minimal deploy recipe (nginx vhost)
├── Makefile                        # `make db | dev | image | up | logs ...`
├── pyproject.toml                  # python deps
├── data/
│   ├── build_app_db.py             # analyses → duckdb + aggregate parquets
│   ├── build_depmap_*.py           # precompute DepMap pair matrices + corr
│   ├── build_tf_pair_table.py      # atlas-wide TF×TF co-occurrence
│   ├── canonical_promoter.duckdb   # built artifact (gitignored)
│   ├── aggregate/                  # parquet matrices (gitignored)
│   ├── gtex/                       # gtex parquet shards (gitignored)
│   ├── depmap/                     # depmap parquet shards (gitignored)
│   ├── network/                    # tf-pair atlas parquet (gitignored)
│   └── manifest.json               # versions + parameters captured at build
├── app/
│   ├── streamlit_app.py            # entry: nav, query-param seed, footer
│   ├── lib/
│   │   ├── db.py                   # cached DuckDB conn + query helpers
│   │   ├── plotting.py             # plotly figure builders + brand palette
│   │   ├── nav.py                  # tab-to-tab navigation helpers
│   │   └── ui.py                   # shared intro_card helper
│   ├── tabs/                       # one module per top-nav tab
│   ├── requirements.txt
│   └── Dockerfile
├── pipeline/                       # the upstream analysis, previously unversioned
│   ├── config.py                   # single resolver for paths + build axes
│   ├── canonical_promoter_aggregate.001.py
│   ├── tss_modules.001.py          # KDE → modules → module×TF NMF
│   ├── tss_modules_select_k.001.py # ARI + Brunet cophenetic rank selection
│   ├── compare_builds.py           # A/B two builds (ARI, program identity)
│   ├── compare_enrichment.py       # GO-specificity arbiter between builds
│   ├── splithalf_threshold.py      # per-TF threshold calibration
│   └── ... (18 analysis scripts in total)
├── acquire/                        # ChIP-Atlas acquisition, also previously loose
│   ├── download_chip_atlas.sh      # tier-parameterised fetch (05/10/20/50)
│   ├── build_tf_list.py            # generates the TF axis
│   ├── split_by_srx.py             # split peaks by experiment (calibration input)
│   ├── chip_atlas_antigen_classification.tsv   # 79 non-gene antigens classified
│   └── tf_list.{all,whitelist}.txt # 1,793 / 1,311 TF axes
├── hpc/                            # run the heavy half on CSC Roihu
│   ├── 00_build_env.sh             # venv (no tykky/python-data on Roihu)
│   ├── 01_stage.sh 02_submit.sh 03_fetch.sh 04_fetch_archive.sh
│   ├── pipeline.slurm archive.slurm sweep_min_score.slurm splithalf.slurm
│   └── nmf_init_probe.slurm
├── docs/
│   └── threshold-calibration.md    # why MIN_SCORE_ASSIGN=250 and k=140
├── .streamlit/
│   └── config.toml                 # theme (teal primary)
└── deploy/
    ├── docker-compose.yml          # 127.0.0.1:8501 backend, hardened
    └── nginx-tfbss.conf            # sample nginx vhost
```

## The analysis pipeline

`pipeline/` holds the upstream analysis that produces the viewer's inputs. Every
machine-specific path resolves through `pipeline/config.py`, which reads a
gitignored `.env` (see `.env.example`) and refuses to run rather than guess:

| env var | what it selects |
|---|---|
| `HPA_TIER` | ChIP-Atlas significance tier — `q1e-50` / `q1e-20` / `q1e-10` / `q1e-5` |
| `HPA_TF_SET` | TF axis — `whitelist` (1,311) or `all` (1,793) |
| `HPA_MIN_SCORE_ASSIGN` | score a peak needs to *assign* a TF to a module (default 250) |
| `HPA_CHIP_ATLAS_DIR`, `HPA_GTF`, `HPA_DNA_BINDING`, `HPA_MSIGDB`, `HPA_GTEX_DIR`, `HPA_DEPMAP_DIR` | input locations |
| `HPA_ANALYSIS_ROOT` / `HPA_OUT_DIR` | where a build lands |

Builds are written to `<root>/<tier>.<tf_set>.s<score>/` so tiers, axes and
thresholds coexist rather than overwrite, and each carries a `_BUILD.json` stamp
that downstream steps check before consuming it.

> **ChIP-Atlas numbering trap.** Bulk filenames use `05/10/20/50` (= Q < 1E-0N)
> while the ChIP-Atlas *website* picker uses `50/100/500/1000` (= −10·log₁₀ Q).
> They differ by 10×. `pipeline/config.py` holds the only mapping between them.

Peak-heavy stages can run on CSC Roihu via `hpc/` (stage → submit → fetch);
GTEx, DepMap and the MSigDB enrichments stay local because their raw inputs are
large and already here.

## Build the data layer

The viewer reads from a pre-built `data/canonical_promoter.duckdb` plus
several parquet shards. Build them with:

```bash
make db                                       # writes canonical_promoter.duckdb
python data/build_depmap_pair_matrices.py     # slim DepMap matrices
python data/build_depmap_tf_target_correlations.py
python data/build_tf_pair_table.py            # TF × TF co-occurrence
```

The build scripts read raw inputs from two directories whose locations
are configurable via env vars:

| env var              | what it holds                                  | default                |
|----------------------|------------------------------------------------|------------------------|
| `HPA_ANALYSIS_DIR`   | upstream chip-atlas analysis outputs           | `data/raw/analyses`    |
| `HPA_DEPMAP_RAW`     | raw DepMap CSVs (Chronos, expression, Model)   | `data/raw/depmap`      |

## Run locally

```bash
pip install -e .
make db          # one-time (or when upstream analyses change)
make dev         # streamlit on :8501
```

Open http://localhost:8501 .

## Deploy

See `DEPLOY.md` for the minimal nginx + Docker recipe.

## Cite

Forthcoming. The data underlying this viewer will be deposited at Zenodo
with a DOI on first public release.

Source code: MIT. Generated figures + tables: CC-BY 4.0.

Underlying data:
- chip-atlas TF ChIP-seq peaks (https://chip-atlas.org)
- Ensembl GRCh38.114 (https://www.ensembl.org)
- MSigDB c5.go.bp.v2026.1.Hs (https://www.gsea-msigdb.org/gsea/msigdb)
- GTEx v8 (https://gtexportal.org)
- DepMap CRISPR (Chronos) gene effects (https://depmap.org)
