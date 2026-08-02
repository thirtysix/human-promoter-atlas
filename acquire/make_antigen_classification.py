#!/usr/bin/env python3
"""
Classify every ChIP-Atlas TF-class antigen that is NOT a current GRCh38 gene
symbol, and record which ones are recoverable real genes hiding behind a
deprecated symbol.

Why this exists
---------------
The per-TF peak files are named after the ChIP-Atlas *antigen*, which for many
factors is a legacy symbol. Matching those names against a current Ensembl GTF
silently drops real, well-studied chromatin factors -- MKL1/MKL2 are MRTFA/MRTFB,
WHSC1/WHSC1L1 are NSD2/NSD3, FAM208A is TASOR, and so on. Ensembl's GTF carries
no `gene_synonym` attribute, so a symbol-only match cannot recover them and the
loss is invisible.

This script emits `chip_atlas_antigen_classification.tsv`, which is the
authoritative answer to "what are the non-gene-symbol antigens, and which of
them are actually genes?". Downstream, `acquire/build_tf_list.py` in the
human-promoter-atlas repo consumes the `renamed_gene` rows to build its TF axis.

Every rename is VERIFIED: the current symbol must exist in the GTF, otherwise
the row is demoted to `retired_unmapped` rather than asserted.

Usage:
    python make_antigen_classification.py [--gtf PATH] [--per-tf PATH]
"""

from __future__ import annotations

import argparse
import os
import sys
import gzip
import re
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "pipeline"))

# Resolved through the repo's config so no machine-specific path is hardcoded.
# Falls back to plain env vars when run standalone next to the peak data, where
# pipeline/config.py may not be importable.
try:
    import config as _cfg
    DEFAULT_GTF = Path(_cfg.GTF_FN)
    DEFAULT_PER_TF = _cfg.CHIP_ATLAS_DN / "per_TF" / "q1e-50"
except Exception:
    DEFAULT_GTF = Path(os.environ.get("HPA_GTF", "Homo_sapiens.GRCh38.114.gtf.gz"))
    DEFAULT_PER_TF = Path(os.environ.get("HPA_CHIP_ATLAS_DIR", ".")) / "per_TF" / "q1e-50"
OUT_TSV = HERE / "chip_atlas_antigen_classification.tsv"

# --- Proposed deprecated -> current symbol map -------------------------------
# Each target is checked against the GTF before being written.
RENAMES = {
    "ARNTL": "BMAL1", "ARNTL2": "BMAL2", "ASUN": "INTS13", "BMI": "BMI1",
    "C10orf12": "LCOR", "C11orf53": "POU2AF2", "C17orf96": "EPOP",
    "C7orf26": "INTS15", "CCDC101": "SGF29", "CPSF3L": "INTS11",
    "DEC1": "DELEC1", "FAM208A": "TASOR", "FAM208B": "TASOR2",
    "GLTSCR1": "BICRA", "GUCY1B3": "GUCY1B1", "HDGF2": "HDGFL2",
    "MGEA5": "OGA", "MINA": "RIOX2", "MKL1": "MRTFA", "MKL2": "MRTFB",
    "MRE11A": "MRE11", "PAD2": "PADI2", "PHB": "PHB1", "PTRF": "CAVIN1",
    "STRA13": "CENPX", "T": "TBXT", "TAZ": "WWTR1", "TCEB3": "ELOA",
    "TRF2": "TERF2", "WHSC1": "NSD2", "WHSC1L1": "NSD3", "ZCCHC11": "TUT4",
    "ZUFSP": "ZUP1",
}

# Symbols retired to pseudogene status in GRCh38.114. Real loci, but not
# protein-coding genes any more -- deliberately NOT added to the TF axis.
PSEUDOGENES = {"ZNF788": "ZNF788P", "ZSCAN5D": "ZSCAN5DP"}

# Legacy names whose modern reading is genuinely ambiguous. Recorded, and
# resolved the way the ChIP-seq context requires, but flagged so a reader can
# disagree rather than inherit a silent guess:
#   TAZ  -> WWTR1 (Hippo coactivator) or TAFAZZIN (mitochondrial acyltransferase)
#   TRF2 -> TERF2 (telomere binding) or TBPL1 (TBP-related factor 2)
# Both alternatives are implausible ChIP targets, but the names collide, and
# these two are ALSO dual-listed, so merging them would fold in a second set of
# experiments on the strength of a guess. Left unmerged.
AMBIGUOUS = {"TAZ", "TRF2"}

# --- Non-gene antigen classes ------------------------------------------------
CHEMICAL_MARK = {
    "5-hmC", "5-mC", "8-Hydroxydeoxyguanosine", "Crotonyllysine", "Cyclobutane",
    "O-GlcNAc", "Pan-acetyllysine", "G-quadruplex", "RDme1", "DNA-RNA", "RNA-DNA",
}
VIRAL_PROTEIN = {
    "AAV", "EBNA1", "EBNA2", "EBNA3", "EBV-ZEBRA", "HBcAg", "HHV", "HIV",
    "HIV1", "Hepatitis", "KSHV", "MCPV", "RTA", "TRP47", "VSV-G",
}
FUSION_PROTEIN = {"AML1-ETO", "MLL-AF10", "MLL-AF4", "MLL-AF6"}
REAGENT_OR_TAG = {
    "Biotin", "BrdU", "Cas9", "Cpf1", "Epitope", "GFP", "MethylCap",
    "pFM2", "SVS-1",
}

NOTES = {
    "renamed_gene": "real gene under a deprecated symbol; RECOVERABLE -- fetch "
                    "from ChIP-Atlas under the antigen name, store under current_symbol",
    "pseudogene_now": "locus retired to pseudogene status in GRCh38.114; excluded from the TF axis",
    "retired_unmapped": "absent from GRCh38.114 under this or any obvious current "
                        "symbol; needs manual review before use",
    "chemical_mark": "chemical/structural DNA or histone mark, not a protein antigen",
    "viral_protein": "virus-encoded protein, not a human gene",
    "fusion_protein": "oncogenic fusion, not a single reference gene",
    "reagent_or_tag": "affinity tag, nuclease, or capture reagent, not a biological antigen",
}


def gtf_gene_symbols(gtf_fn: Path) -> set[str]:
    out: set[str] = set()
    rx = re.compile(r'gene_name "([^"]+)"')
    with gzip.open(gtf_fn, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            f = line.split("\t", 3)
            if len(f) < 3 or f[2] != "gene":
                continue
            m = rx.search(line)
            if m:
                out.add(m.group(1))
    return out


def peak_count(path: Path) -> int:
    if not path.exists():
        return -1
    r = subprocess.run(f"zcat {path!s} | wc -l", shell=True,
                       capture_output=True, text=True)
    return int(r.stdout.strip() or 0)


def classify(name: str, genes: set[str]) -> tuple[str, str]:
    if name in RENAMES:
        tgt = RENAMES[name]
        return ("renamed_gene", tgt) if tgt in genes else ("retired_unmapped", "")
    if name in PSEUDOGENES:
        return "pseudogene_now", PSEUDOGENES[name]
    if name in CHEMICAL_MARK:
        return "chemical_mark", ""
    if name in VIRAL_PROTEIN:
        return "viral_protein", ""
    if name in FUSION_PROTEIN:
        return "fusion_protein", ""
    if name in REAGENT_OR_TAG:
        return "reagent_or_tag", ""
    return "retired_unmapped", ""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gtf", type=Path, default=DEFAULT_GTF)
    ap.add_argument("--per-tf", type=Path, default=DEFAULT_PER_TF)
    ap.add_argument("--out", type=Path, default=OUT_TSV)
    args = ap.parse_args()

    genes = gtf_gene_symbols(args.gtf)
    stems = sorted(p.name[: -len(".bed.gz")] for p in args.per_tf.glob("*.bed.gz"))
    non_gene = [s for s in stems if s not in genes]

    rows = []
    for name in non_gene:
        cls, cur = classify(name, genes)
        # Is the current symbol ALSO served as its own antigen? If so the two
        # files hold DIFFERENT experiments (verified: zero shared SRX), so both
        # must be read -- taking only one silently drops data.
        dual = bool(cur) and (args.per_tf / f"{cur}.bed.gz").exists()
        note = NOTES[cls]
        if dual:
            note += "; DUAL-LISTED -- current symbol is also its own antigen file, " \
                    "read BOTH (they contain different SRX experiments)"
        if name in AMBIGUOUS:
            note += "; AMBIGUOUS legacy name -- not merged, review before use"
        rows.append({
            "antigen": name,
            "class": cls,
            "current_symbol": cur,
            "in_grch38_114": "yes" if cur and cur in genes else "no",
            "dual_listed": "yes" if dual else "no",
            "ambiguous": "yes" if name in AMBIGUOUS else "no",
            "peaks_q1e50": peak_count(args.per_tf / f"{name}.bed.gz"),
            "note": note,
        })

    cols = ["antigen", "class", "current_symbol", "in_grch38_114", "dual_listed",
            "ambiguous", "peaks_q1e50", "note"]
    header = (
        "# ChIP-Atlas TF-class antigens that are NOT current GRCh38.114 gene symbols.\n"
        f"# Source antigen set: <HPA_CHIP_ATLAS_DIR>/per_TF/{args.per_tf.name}\n"
        f"# Reference: {args.gtf.name}\n"
        "#\n"
        "# class=renamed_gene rows are REAL factors recoverable under current_symbol.\n"
        "# Ensembl's GTF has no gene_synonym attribute, so a symbol-only match drops\n"
        "# them silently -- that is the bug this table exists to prevent.\n"
        "# Regenerate: python make_antigen_classification.py\n"
    )
    with open(args.out, "w") as fh:
        fh.write(header)
        fh.write("\t".join(cols) + "\n")
        for r in sorted(rows, key=lambda r: (r["class"], r["antigen"])):
            fh.write("\t".join(str(r[c]) for c in cols) + "\n")

    from collections import Counter
    counts = Counter(r["class"] for r in rows)
    print(f"antigens on disk        {len(stems):>5}")
    print(f"not a GRCh38 symbol     {len(non_gene):>5}")
    for cls, n in sorted(counts.items(), key=lambda kv: -kv[1]):
        print(f"  {cls:<20} {n:>5}")
    rec = [r for r in rows if r["class"] == "renamed_gene"]
    print(f"\nRECOVERABLE: {len(rec)} factors, "
          f"{sum(r['peaks_q1e50'] for r in rec):,} peaks at q1e-50")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
