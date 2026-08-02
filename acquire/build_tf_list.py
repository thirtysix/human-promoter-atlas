#!/usr/bin/env python3
"""
Generate the transcription-factor axis for a build, and write it as a tracked,
reviewable list instead of leaving it implicit in whichever TSV happened to be
on disk.

Two axes are produced from the same downloaded peak set:

  whitelist  peak-file stems that appear in the curated DNA_binding_genes.tsv.
             This reproduces the original atlas (1,304 TFs).

  all        peak-file stems that are real GRCh38 gene symbols. This is the
             broader axis (1,767 TFs). The extra factors are not junk: they are
             cohesin (STAG1/2/3), Mediator (CDK8, MED30), bromodomains
             (BRD2/BRD9), PRC1 (RNF2), SAGA (ATXN7L3), LSD1 (KDM1A) and
             similar chromatin proteins that the curated whitelist happens to
             exclude even though the atlas's own NMF programs already feature
             the same class (EP300, BRD4, MED1, SUZ12, KMT2A, RAD21, SMC1A/3).

Non-gene antigens (5-hmC, 5-mC, BrdU, Cas9, Biotin, Epitope, EBNA*, AAV, ...)
are excluded from BOTH axes by the GTF gene-symbol test, which is the point of
using the GTF rather than a hand-maintained deny-list.

Gene-symbol drift
-----------------
A symbol-only match against the GTF silently drops real factors whose ChIP-Atlas
antigen name is a legacy symbol -- MKL1/MKL2 are MRTFA/MRTFB, WHSC1/WHSC1L1 are
NSD2/NSD3, FAM208A is TASOR, ARNTL is BMAL1. Ensembl's GTF has no gene_synonym
attribute, so the loss is invisible. We therefore consult the verified rename
table maintained alongside the peaks:

    chip-atlas/00.data/TF/chip_atlas_antigen_classification.tsv

Any peak file whose stem appears there as class=renamed_gene is admitted under
its current_symbol. Peak files downloaded by acquire/download_chip_atlas.sh are
already stored under the current symbol, so this mapping is a no-op for them and
matters only for older per-TF trees named after the raw antigen.

Usage:
    python acquire/build_tf_list.py                 # both axes, tier from .env
    python acquire/build_tf_list.py --tf-set all
    python acquire/build_tf_list.py --tier q1e-5 --check
"""

from __future__ import annotations

import argparse
import gzip
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "pipeline"))
import config  # noqa: E402

GENE_NAME_RE = re.compile(r'gene_name "([^"]+)"')


def gtf_gene_symbols(gtf_fn: Path) -> set[str]:
    """Every gene_name on a `gene` feature line. ~2 s on the GRCh38 GTF."""
    out: set[str] = set()
    with gzip.open(gtf_fn, "rt") as fh:
        for line in fh:
            if line.startswith("#"):
                continue
            # Cheap field check before the regex — most lines are not genes.
            f = line.split("\t", 3)
            if len(f) < 3 or f[2] != "gene":
                continue
            m = GENE_NAME_RE.search(line)
            if m:
                out.add(m.group(1))
    return out


def peak_file_stems(per_tf_dn: Path) -> set[str]:
    stems = {p.name[: -len(".bed.gz")] for p in per_tf_dn.glob("*.bed.gz")}
    if len(stems) < 1000:
        raise SystemExit(
            f"build_tf_list: only {len(stems)} *.bed.gz under {per_tf_dn} "
            f"(expected ~1,800). Wrong tier directory?"
        )
    return stems


def whitelist_names(dna_binding_fn: Path) -> set[str]:
    lines = dna_binding_fn.read_text().splitlines()
    return {ln.split("\t")[0].strip() for ln in lines[1:] if ln.strip()}


def rename_map(chip_atlas_dn: Path) -> dict[str, str]:
    """Legacy ChIP-Atlas antigen -> current GRCh38 symbol.

    Read from chip_atlas_antigen_classification.tsv, which only records renames
    whose target was verified present in the GTF. Missing file is not fatal:
    the axis simply loses the recoverable factors, and we say so.
    """
    path = chip_atlas_dn / "chip_atlas_antigen_classification.tsv"
    if not path.is_file():
        print(f"note: {path.name} not found — {path.parent}\n"
              f"      gene-symbol drift will NOT be corrected; regenerate with\n"
              f"      python {path.parent}/make_antigen_classification.py")
        return {}
    out: dict[str, str] = {}
    for ln in path.read_text().splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        f = ln.split("\t")
        if len(f) >= 3 and f[1] == "renamed_gene" and f[2]:
            out[f[0]] = f[2]
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default=config.TIER,
                    choices=sorted(config.TIER_TO_FILE_INFIX))
    ap.add_argument("--tf-set", default=None, choices=list(config.TF_SETS),
                    help="default: write both")
    ap.add_argument("--check", action="store_true",
                    help="report only; do not write")
    ap.add_argument("--label", default=None,
                    help="write to tf_list.<label>.txt instead of tf_list.<tf-set>.txt. "
                         "Use to freeze a comparison baseline, e.g. "
                         "--tier q1e-50 --tf-set whitelist --label baseline1304")
    args = ap.parse_args()
    if args.label and not args.tf_set:
        ap.error("--label requires --tf-set (a label names exactly one axis)")

    per_tf_dn = config.CHIP_ATLAS_DN / "per_TF" / args.tier
    out_dir = Path(__file__).resolve().parent

    stems = peak_file_stems(per_tf_dn)
    genes = gtf_gene_symbols(config.GTF_FN)
    wl = whitelist_names(config.DNA_BINDING_FN)
    renames = rename_map(config.CHIP_ATLAS_DN)

    # Resolve each peak-file stem to a current symbol where one is known. Files
    # already stored under the current symbol map to themselves.
    resolved = {s: renames.get(s, s) for s in stems}
    recovered = sorted({v for s, v in resolved.items() if s != v and v in genes})
    symbols = {v for v in resolved.values() if v in genes}

    axes = {
        "all": sorted(symbols),
        "whitelist": sorted(symbols & wl),
    }
    non_genes = sorted(s for s, v in resolved.items() if v not in genes)

    print(f"tier                    {args.tier}  ({per_tf_dn})")
    print(f"peak files              {len(stems):>6}")
    print(f"GRCh38 gene symbols     {len(genes):>6}")
    print(f"curated whitelist       {len(wl):>6}")
    print(f"rename table entries    {len(renames):>6}")
    print(f"recovered via rename    {len(recovered):>6}"
          + (f"  e.g. {', '.join(recovered[:8])}" if recovered else "  (none needed)"))
    print(f"-> tf_list.all          {len(axes['all']):>6}")
    print(f"-> tf_list.whitelist    {len(axes['whitelist']):>6}")
    print(f"non-gene antigens       {len(non_genes):>6}"
          + (f"  {non_genes[:8]}" if non_genes else "  (none)"))
    extra = sorted(set(axes["all"]) - set(axes["whitelist"]))
    print(f"in all, not whitelist   {len(extra):>6}"
          + (f"  e.g. {', '.join(extra[:10])}" if extra else ""))

    if args.check:
        return 0

    for name, names in axes.items():
        if args.tf_set and name != args.tf_set:
            continue
        path = out_dir / f"tf_list.{args.label or name}.txt"
        header = (
            f"# Human Promoter Atlas TF axis: {name}\n"
            f"# {len(names)} factors — peak-file stems from ChIP-Atlas tier "
            f"{args.tier}, intersected with GRCh38 gene symbols"
            + (" and the curated DNA_binding_genes.tsv" if name == "whitelist" else "")
            + "\n# Regenerate: python acquire/build_tf_list.py\n"
        )
        path.write_text(header + "\n".join(names) + "\n")
        print(f"wrote {path}  ({len(names)} TFs)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
