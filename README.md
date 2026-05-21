# Human Promoter Atlas

Interactive web companion to a canonical-promoter analysis: TF ChIP-seq
binding patterns at the TSSs of all canonical protein-coding transcripts
in the human genome (Ensembl GRCh38.114), plus the per-gene regulatory
**modules** and NMF **programs** discovered from chip-atlas data over
~1,300 TFs and ~19,700 TSSs.

This repo contains the **viewer**. The upstream analysis pipeline that
produces the inputs lives outside this repo; pointers to its expected
layout are in `data/build_app_db.py`.

## What you can do here

- **Aggregate**: how TFs distribute around canonical TSSs across the
  whole genome — heatmaps with toggleable row ordering (total signal /
  argmax position / hierarchical / K=8 cluster), metaplots, quick-start
  links.
- **Programs & modules**: browse the 10 algorithmically-selected NMF
  programs. Top TFs, top GO BP terms, position density, K=8-cluster
  cross-tab, plus a 10×10 program-co-occurrence heatmap (lift = obs/exp).
- **Archetypes**: gene-level NMF on per-gene program-presence vectors
  (A=8). Each canonical gene gets a single archetype label with
  biology-clean GO BP enrichment (e.g., A6 cohesin → homophilic
  cell-cell adhesion at OR=7.5).
- **GO search**: reverse search by GO term — find programs / archetypes
  that enrich for any of 147 indexed biological processes.
- **Per-transcript**: search by gene/transcript → modules + TF rugs +
  per-program rows + clickable program-detail popovers + archetype label
  + GTEx tissue expression + DepMap CRISPR essentiality.
- **Compare**: two transcripts side-by-side — promoter maps, program
  presence diff, paired GTEx, paired DepMap.
- **Per-TF**: aggregate profile (with optional cluster-mean overlay),
  program loadings across the 10 programs, GTEx expression, DepMap
  essentiality, top co-binding partners, top bound TSSs.
- **TF network**: atlas-wide TF × TF co-binding pairs — 327k pairs over
  1,064 TFs, ranked by n_shared / Jaccard / lift.
- **Methods**: glossary, pipeline, parameters with rationale,
  verification anchors, limitations, citations.

URL deep-linking is supported: `?gene=GAPDH&tf=CTCF&program=7` seeds the
search state on load, and selectbox changes write back to the URL so
every view is shareable.

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
├── .streamlit/
│   └── config.toml                 # theme (teal primary)
└── deploy/
    ├── docker-compose.yml          # 127.0.0.1:8501 backend, hardened
    └── nginx-tfbss.conf            # sample nginx vhost
```

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
