"""Methods tab: provenance, glossary, pipeline, parameter rationale,
verification anchors, limitations, citations."""
from __future__ import annotations

import pandas as pd
import streamlit as st

from app.lib import db, ui


# ---------------------------------------------------------------------------
# Static content
# ---------------------------------------------------------------------------
GLOSSARY = [
    ("TSS", "Transcription start site. Each canonical protein-coding "
            "transcript (Ensembl_canonical) contributes one TSS."),
    ("ChIP-atlas peak", "A genomic interval where a TF was found bound in "
                        "at least one ChIP-seq experiment. Peak score "
                        "(0–1000) reflects how many experiments support it; "
                        "score = 1000 means the peak is essentially "
                        "universal across the experiments aggregated."),
    ("25-nt recentering",
     "Each chip-atlas peak is collapsed to a 25-bp block around its midpoint "
     "before any analysis, so a single TF call contributes a fixed-width "
     "footprint to its TSS regardless of how broadly the peak was originally "
     "called."),
    ("Module",
     "A local concentration of distinct-TF binding within ±1.5 kb of a TSS, "
     "found as a peak in the per-TSS KDE density (σ = 25 bp). A module is "
     "supported by ≥2 distinct TFs (any score) and is assigned the TFs that "
     "have ≥1 peak with score ≥ 500 inside its boundaries."),
    ("Program",
     "An NMF component over the [n_module × n_TF] occupancy matrix. We chose "
     "k=10 by ARI stability + Brunet cophenetic correlation (both peaked at "
     "k=10). Each program has a TF signature (its top loadings in H) and a "
     "characteristic position relative to the TSS."),
    ("program_path",
     "An ordered list of dominant programs across a gene's modules, "
     "transcription-direction. e.g. `9,7,5,1` reads as 'NFY/Sp upstream → PIC "
     "at core → cohesin → downstream chromatin'."),
    ("KDE",
     "Kernel Density Estimation — a smoothed 1-D density built from "
     "individual peak midpoints. We use a Gaussian kernel with σ = 25 bp, "
     "matched to ChIP-peak resolution."),
    ("Lift",
     "Observed / expected under independence. Lift > 1 means two programs "
     "co-occur at the same gene more often than chance; < 1 means they avoid "
     "each other."),
    ("genome_bg",
     "Background gene set used in GO BP hypergeometric tests = the entire "
     "MSigDB c5.go.bp gene universe (~18,000 genes). The standard "
     "biological-interpretation background."),
]

PIPELINE = [
    ("1. Aggregate",
     "canonical_promoter_aggregate.001.py",
     "ChIP-atlas BEDs → mean TF×position matrices over ±1 kb (binary, score, "
     "raw, score==1000). Peaks recentered to 25 bp before counting."),
    ("2. TF clustering",
     "cluster_tfs.001.py / cluster_tfs.no_filter.001.py",
     "Ward + Euclidean on each TF's peak-normalized aggregate profile shape. "
     "Produces 8 clusters (filtered, 184 TFs) and 12 (no_filter, 1,207 TFs)."),
    ("3. TF-cluster GO BP",
     "enrich_clusters_msigdb.001.py",
     "Hypergeometric vs MSigDB c5.go.bp universe, BH-FDR, both `tf_bg` and "
     "`genome_bg` backgrounds. The genome_bg version is used in the app."),
    ("4. Per-gene modules",
     "tss_modules.001.py",
     "Per-TSS KDE on peak midpoints (σ = 25 bp, weight 1 per TF per TSS), "
     "find_peaks → module centers, walk-out boundary detection, ≥2-TF "
     "support filter, score ≥ 500 for binary occupancy assignment."),
    ("5. NMF on modules",
     "tss_modules_select_k.001.py + tss_modules_k10.py",
     "20-seed ARI + 5,000-module Brunet cophenetic to pick k=10. NMF (MU "
     "solver, random init) on the 76,999 × 1,304 sparse binary matrix → "
     "W (modules × programs), H (programs × TFs)."),
    ("6. Per-program GO BP",
     "enrich_tss_modules_msigdb.001.py",
     "Hypergeometric vs MSigDB universe; gene set per program = unique "
     "gene_names with ≥1 module dominantly assigned."),
    ("7. App data layer",
     "data/build_app_db.py",
     "Packs every output above into a single DuckDB plus the aggregate "
     "parquets used by this viewer."),
]

PARAMS_WITH_RATIONALE = [
    ("Window (aggregate)", "±1,000 bp",
     "Centered around the TSS. Wide enough to see core + flanks; narrow "
     "enough to keep memory tractable."),
    ("Window (modules)", "±1,500 bp",
     "Wider than aggregate so we can also catch downstream insulator / "
     "elongation modules; KDE peaks beyond this window would not represent "
     "promoter-proximal regulation."),
    ("Peak recentering", "25 bp",
     "ChIP-peak resolution is in the 25–100 bp range; collapsing to 25 bp "
     "around the midpoint sharpens the aggregate metaplot without losing "
     "signal."),
    ("KDE bandwidth (σ)", "25 bp",
     "Matches the recentered peak width; smaller would over-fragment; larger "
     "would merge adjacent functional sites."),
    ("Min support per module", "≥2 distinct TFs",
     "≥1 invites isolated-peak noise; ≥3 occasionally drops single-TF + "
     "cofactor lineage modules. 2 is the empirical compromise."),
    ("Min peak score for assignment", "≥500",
     "ChIP-atlas scores cap at 1,000. ≥500 keeps cell-type-restricted "
     "strong peaks while filtering weak / one-off calls. Module *discovery* "
     "uses all peaks; only TF *assignment* applies the filter."),
    ("Boundary fraction", "20% of peak height",
     "How far we walk outward from a module center before declaring its "
     "edge. Combined with valley-detection between adjacent peaks."),
    ("NMF rank k", "10",
     "Chosen by 20-seed ARI stability + Brunet cophenetic correlation. Both "
     "diagnostics peak at k=10. Higher k explores finer biology but is "
     "less reproducible across initializations."),
    ("NMF solver", "Multiplicative-update (MU), random init",
     "Default `init=nndsvd` stalled 25 min on a futex with default OpenBLAS "
     "threading on the sparse matrix. MU + random init runs in 1–8 s/fit."),
    ("MIN_SAMPLES_FOR_TAU", "40 donors / tissue",
     "GTEx tissues with fewer than 40 average donor samples are excluded "
     "from the Yanai 2005 tau (tissue-specificity index) computation for "
     "programs and archetypes. Below ~40 donors the per-tissue mean TPM is "
     "noisy enough to inflate apparent tissue specificity. With the cutoff, "
     "~50 of 66 GTEx v8 tissues are retained — every major organ system "
     "stays, only thinly-sampled sub-tissues (e.g. specific brain "
     "subregions, Cells - cultured fibroblasts) are dropped. Same threshold "
     "applies to program and archetype tau."),
]

VERIFICATION = [
    ("TBP argmax", "≈ −30 bp from TSS",
     "Canonical TATA-box position. If recentering or strand orientation is "
     "broken, TBP centers at 0 instead of −30."),
    ("CTCF argmax", "within ±200 bp",
     "CTCF's promoter-proximal binding peaks at the TSS flank. Far-from-TSS "
     "argmax = strand-flip suspected."),
    ("Aggregate metaplot peak", "within ±100 bp",
     "Mean across all TFs across all canonical TSSs has a clear central "
     "peak — the promoter is a real signal-density spike, not a flat plateau."),
    ("CTCF in two programs",
     "P5 (median −43 bp, w/ RAD21/SMC1A/SMC3) and P1 (median +217 bp, w/ "
     "BRD4/EP300/ETV6)",
     "Same TF in two roles at two positions, separated by NMF — biological "
     "validation that the per-gene module framing adds resolution beyond "
     "single-window analysis."),
    ("P10 → ribosome biogenesis",
     "GABPA/THAP11/ETS1; OR=5.4, q = 7e-70",
     "GABPA is the canonical regulator of nuclear-encoded ribosomal/"
     "mitochondrial genes. The sharpest enrichment in the GO BP set."),
    ("P3 → immune system process",
     "RUNX1/FLI1/SPI1; OR=2.3, q = 1e-54",
     "Hematopoietic ETS factors regulating immune-cell genes — textbook."),
    ("Modules per TSS", "median 4, mean 4.2",
     "Most canonical promoters are *multipartite*. ~7% have 0 modules "
     "(no chip-atlas evidence); ~12% are mono-modular focused promoters."),
]

LIMITATIONS = [
    ("Cell-type pooling",
     "ChIP-atlas aggregates experiments across tissues and cell lines for "
     "each TF. A peak at a TSS reflects 'observed in *any* assayed context', "
     "not 'always co-bound'. Cell-type-specific co-binding is therefore "
     "blurred."),
    ("Score saturation",
     "ChIP-atlas peak scores cap at 1,000. The score = 1,000 subset is "
     "biased toward universally bound TFs (CTCF/MYC/SP1) — we use score ≥ "
     "500 instead so cell-type-restricted regulators survive."),
    ("Annotation dependence",
     "We restrict to Ensembl_canonical protein-coding transcripts on "
     "chromosomes 1–22, X, Y, MT. lncRNAs, miRNAs, and non-canonical "
     "transcripts are out of scope."),
    ("TF coverage",
     "1,304 TFs after intersecting chip-atlas filename stems with a curated "
     "DNA-binding gene list. ~250 known human TFs are not represented because "
     "no chip-atlas data exists for them."),
    ("KDE bandwidth choice",
     "σ = 25 bp is matched to ChIP-peak resolution. Smaller σ over-segments; "
     "larger σ merges adjacent sites. Sweeping σ ∈ {15, 25, 50} as a "
     "robustness check is on the to-do list."),
    ("k = 10 is one rank",
     "ARI / cophenetic both peaked at 10, but the data is genuinely high-"
     "rank — k = 25 also has high stability and reveals finer biology. "
     "Higher-k summary tables are kept for browsing."),
    ("No archetype layer (yet)",
     "Each gene currently has a `program_path` (ordered list of program ids) "
     "but no learned 'gene archetype'. NMF on the [n_gene × 10] program-"
     "presence matrix is the natural next layer and is on the to-do list."),
]


# ---------------------------------------------------------------------------
# Render
# ---------------------------------------------------------------------------
def render() -> None:
    ui.intro_card(
        title="Methods & data — provenance and reproducibility",
        what="The exact input dataset versions, parameter values, and "
             "scripts that produced everything in this app — plus a "
             "glossary, the verification anchors that gate releases, and "
             "the known limitations.",
        objective="Make every number in the atlas reproducible: which "
                   "TSSs entered, what window, what score filter, what "
                   "KDE bandwidth, what NMF rank, and how to re-run if "
                   "you want to swap any of these.",
        significance="A viewer without provenance is anecdote. The "
                      "Methods tab is what lets a reader audit, replicate, "
                      "or rebuild the analysis.",
    )

    m = db.load_manifest()

    # ---- Jump-to table of contents ---------------------------------------
    st.markdown(
        "**Jump to:** "
        "[Glossary](#glossary) · "
        "[Pipeline](#pipeline) · "
        "[Parameters](#parameters) · "
        "[Verification](#verification) · "
        "[Limitations](#limitations) · "
        + ("[Manifest](#manifest) · " if m else "")
        + "[Cite](#cite)"
    )

    # ---- Glossary ---------------------------------------------------------
    with st.container(border=True):
        st.markdown('<a id="glossary"></a>', unsafe_allow_html=True)
        st.markdown("### Glossary",
                     help="Short definitions of the key terms used throughout "
                          "the atlas.")
        st.dataframe(
            pd.DataFrame(GLOSSARY, columns=["Term", "Definition"]),
            hide_index=True, width="stretch",
            column_config={
                "Term": st.column_config.TextColumn(width="small"),
                "Definition": st.column_config.TextColumn(width="large"),
            },
        )

    # ---- Pipeline ---------------------------------------------------------
    with st.container(border=True):
        st.markdown('<a id="pipeline"></a>', unsafe_allow_html=True)
        st.markdown("### Pipeline",
                     help="Stage-by-stage breakdown of the upstream analysis "
                          "scripts that fed this viewer.")
        st.dataframe(
            pd.DataFrame(PIPELINE, columns=["Stage", "Script", "What it does"]),
            hide_index=True, width="stretch",
            column_config={
                "Stage": st.column_config.TextColumn(width="small"),
                "Script": st.column_config.TextColumn(width="medium"),
                "What it does": st.column_config.TextColumn(width="large"),
            },
        )

    # ---- Parameters with rationale ---------------------------------------
    with st.container(border=True):
        st.markdown('<a id="parameters"></a>', unsafe_allow_html=True)
        st.markdown("### Parameters (with rationale)",
                     help="Why each numerical parameter was chosen at the "
                          "value it is. Re-running with different values "
                          "changes module / program output.")
        st.dataframe(
            pd.DataFrame(PARAMS_WITH_RATIONALE,
                          columns=["Parameter", "Value", "Rationale"]),
            hide_index=True, width="stretch",
            column_config={
                "Parameter": st.column_config.TextColumn(width="small"),
                "Value": st.column_config.TextColumn(width="small"),
                "Rationale": st.column_config.TextColumn(width="large"),
            },
        )

    # ---- Verification anchors --------------------------------------------
    with st.container(border=True):
        st.markdown('<a id="verification"></a>', unsafe_allow_html=True)
        st.markdown("### Verification anchors",
                     help="Biology-known checkpoints that gate every "
                          "release. If any of these break, the upstream "
                          "pipeline is suspect.")
        st.dataframe(
            pd.DataFrame(VERIFICATION,
                          columns=["Anchor", "Expected", "Why it matters"]),
            hide_index=True, width="stretch",
            column_config={
                "Anchor": st.column_config.TextColumn(width="small"),
                "Expected": st.column_config.TextColumn(width="medium"),
                "Why it matters": st.column_config.TextColumn(width="large"),
            },
        )

    # ---- Limitations ------------------------------------------------------
    with st.container(border=True):
        st.markdown('<a id="limitations"></a>', unsafe_allow_html=True)
        st.markdown("### Limitations & caveats",
                     help="Known scope and assumption issues — read these "
                          "before drawing strong claims from any specific "
                          "promoter or program.")
        st.dataframe(
            pd.DataFrame(LIMITATIONS, columns=["Limitation", "Detail"]),
            hide_index=True, width="stretch",
            column_config={
                "Limitation": st.column_config.TextColumn(width="small"),
                "Detail": st.column_config.TextColumn(width="large"),
            },
        )

    # ---- Datasets + parameters from manifest -----------------------------
    if m:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown('<a id="manifest"></a>', unsafe_allow_html=True)
                st.markdown("### Build manifest — datasets",
                             help="Source data versions captured at build "
                                  "time.")
                for k, v in m.get("datasets", {}).items():
                    st.markdown(f"- **{k}**: {v}")
                st.markdown(f"*Built at: {m.get('built_at', '?')}*")
        with col2:
            with st.container(border=True):
                st.markdown("### Build manifest — parameters",
                             help="Numerical parameters captured at build "
                                  "time.")
                for k, v in m.get("parameters", {}).items():
                    st.markdown(f"- `{k}` = `{v}`")

    # ---- Citation + License ----------------------------------------------
    with st.container(border=True):
        st.markdown('<a id="cite"></a>', unsafe_allow_html=True)
        st.markdown("### Cite & data sources")
        st.markdown(
            """
            **Cite this app**: forthcoming. The data + viewer will be
            deposited at Zenodo with a DOI on first public release.

            Please cite the underlying data sources alongside this app:

            - **chip-atlas**: Oki, S. *et al.* (2018). EMBO Reports, e46255.
              DOI:10.15252/embr.201846255
            - **Ensembl**: Martin, F.J. *et al.* (2023). Ensembl 2023.
              Nucleic Acids Research 51(D1).
            - **MSigDB**: Liberzon, A. *et al.* (2011). Bioinformatics 27(12).

            **License**: app source code MIT; data tables and figures CC-BY 4.0.
            """
        )
