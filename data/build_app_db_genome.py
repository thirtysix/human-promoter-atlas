#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Python vers. 3.12 ###########################################################
"""
Add the genome-wide element layer to the app database.

ADDITIVE, not a replacement. build_app_db.py's promoter tables (tss, modules,
module_program) keep serving the gene-centric front door -- a user searches a
gene and gets its promoter, which is what the site is for. This adds the
annotation-free layer behind it, so the same gene page can also show proximal
and distal elements that a +/-1.5 kb window could never display, with programs
from the genome-wide factorization.

The regression gate is what licenses running both off one page: promoter-stratum
elements reproduce 98.2% of the comparable promoter modules at a median peak
offset of 12 bp (half the 25 bp recentering bin), so the two layers agree
wherever they overlap and disagreement would be a bug, not a finding.

Why element_program_top is long and not wide
--------------------------------------------
module_program stores one REAL column per program (prog1_w .. prog10_w). At
k=140 over 467,223 elements that is a 140-column table of 65 million floats,
and a UI showing 140 weights tells a reader nothing. Only the dominant
assignment is stored wide; the next-best few go in long format, which is both
smaller and the shape the UI actually queries.

NEAREST GENE IS A LOCATOR, NOT AN ASSIGNMENT
--------------------------------------------
n_tss_comparably_close travels with every element on purpose. 56.6% of distal
elements have a rival TSS within twice the nearest distance, so for most of
them "nearest gene" is close to a coin flip. Any view that lists distal
elements under a gene must surface that column; presenting the link without it
asserts a regulatory relationship the data does not support.

Usage:
    python data/build_app_db_genome.py --genome-dir <dir> --k 140
"""

################################################################################
# Libraries ####################################################################
################################################################################
import argparse
import datetime as dt
from pathlib import Path

import duckdb
import numpy as np
import pandas as pd

TOP_PROGRAMS_PER_ELEMENT = 5     # long-format rows kept per element
MIN_STORED_WEIGHT = 0.01         # below this a program contributes nothing


def _log(m):
    print(f"[{dt.datetime.now().strftime('%H:%M:%S')}] {m}", flush=True)


def _exec(con, sql, *params):
    con.execute(sql, params) if params else con.execute(sql)


def load_elements(con, root: Path):
    """The element table, gene-annotated so a gene page can query it."""
    src = root / "elements.genes.tsv"
    if not src.exists():
        raise SystemExit(
            f"{src} not found. Run pipeline/genome_annotate_genes.py first -- "
            f"elements.tsv alone has dist_to_tss but no gene identity, so a "
            f"gene-centric view cannot query it.")
    el = pd.read_csv(src, sep="\t", dtype={"chrom": str})
    keep = ["element_id", "chrom", "start", "end", "peak", "width",
            "n_peaks_in", "n_tfs_supporting", "n_tfs_assigned",
            "dist_to_tss", "stratum", "cluster_id", "cluster_size",
            "nearest_tss_id", "nearest_gene_name", "nearest_gene_id",
            "nearest_transcript_id", "n_tss_within_10kb",
            "n_tss_comparably_close"]
    missing = [c for c in keep if c not in el.columns]
    if missing:
        raise SystemExit(f"{src} is missing {missing}")
    el = el[keep].copy()
    for c in ("element_id", "start", "end", "peak", "width", "n_peaks_in",
              "n_tfs_supporting", "n_tfs_assigned", "dist_to_tss",
              "cluster_id", "cluster_size", "n_tss_within_10kb",
              "n_tss_comparably_close"):
        el[c] = el[c].astype(np.int64)

    _exec(con, "DROP TABLE IF EXISTS elements;")
    _exec(con, """
        CREATE TABLE elements (
            element_id             INTEGER PRIMARY KEY,
            chrom                  VARCHAR,
            start                  INTEGER,
            "end"                  INTEGER,
            peak                   INTEGER,
            width                  INTEGER,
            n_peaks_in             INTEGER,
            n_tfs_supporting       INTEGER,
            n_tfs_assigned         INTEGER,
            dist_to_tss            INTEGER,
            stratum                VARCHAR,
            cluster_id             INTEGER,
            cluster_size           INTEGER,
            nearest_tss_id         VARCHAR,
            nearest_gene_name      VARCHAR,
            nearest_gene_id        VARCHAR,
            nearest_transcript_id  VARCHAR,
            n_tss_within_10kb      INTEGER,
            -- how many TSSs sit within 2x the nearest distance. 56.6% of
            -- distal elements have >=2; any gene-page listing must show it.
            n_tss_comparably_close INTEGER
        );
    """)
    con.register("el_df", el)
    _exec(con, "INSERT INTO elements SELECT * FROM el_df;")
    con.unregister("el_df")
    for col in ("nearest_gene_name", "nearest_tss_id", "stratum",
                "cluster_id", "chrom"):
        _exec(con, f"CREATE INDEX idx_el_{col} ON elements({col});")
    _log(f"  elements: {len(el):,} rows")
    return el


def load_element_programs(con, root: Path, k: int, n_elements: int):
    """Dominant assignment wide, next-best few long."""
    src = root / f"nmf.k{k}.element_program.tsv.gz"
    ep = pd.read_csv(src, sep="\t", dtype={"chrom": str})
    if len(ep) != n_elements:
        raise SystemExit(
            f"{src} has {len(ep):,} rows but elements has {n_elements:,}. The "
            f"factorization and the element table are not the same build.")
    wide = ep[["element_id", "dominant_program", "dominant_weight"]].copy()
    wide["element_id"] = wide.element_id.astype(np.int64)
    wide["dominant_program"] = wide.dominant_program.astype(np.int32)
    wide["dominant_weight"] = wide.dominant_weight.astype(np.float32)

    _exec(con, "DROP TABLE IF EXISTS element_program;")
    _exec(con, """
        CREATE TABLE element_program (
            element_id       INTEGER PRIMARY KEY,
            dominant_program INTEGER,
            dominant_weight  REAL
        );
    """)
    con.register("ep_df", wide)
    _exec(con, "INSERT INTO element_program SELECT * FROM ep_df;")
    con.unregister("ep_df")
    _exec(con, "CREATE INDEX idx_ep_program ON element_program(dominant_program);")
    _log(f"  element_program (k={k}): {len(wide):,} rows")

    # long format for the runners-up, straight off the stored W
    wnpz = root / f"nmf.k{k}.W.npz"
    z = np.load(wnpz)
    W = z["W"]
    ids = z["element_id"]
    if W.shape[0] != n_elements:
        raise SystemExit(f"{wnpz} has {W.shape[0]:,} rows, expected {n_elements:,}")
    rs = W.sum(axis=1, keepdims=True)
    Wn = W / np.where(rs > 0, rs, 1.0)
    top = np.argsort(Wn, axis=1)[:, ::-1][:, :TOP_PROGRAMS_PER_ELEMENT]
    rows_e, rows_p, rows_w = [], [], []
    for j in range(TOP_PROGRAMS_PER_ELEMENT):
        p = top[:, j]
        w = Wn[np.arange(len(p)), p]
        m = w >= MIN_STORED_WEIGHT
        rows_e.append(ids[m]); rows_p.append(p[m] + 1); rows_w.append(w[m])
    lng = pd.DataFrame({
        "element_id": np.concatenate(rows_e).astype(np.int64),
        "program": np.concatenate(rows_p).astype(np.int32),
        "weight": np.concatenate(rows_w).astype(np.float32)})
    _exec(con, "DROP TABLE IF EXISTS element_program_top;")
    _exec(con, """
        CREATE TABLE element_program_top (
            element_id INTEGER,
            program    INTEGER,
            weight     REAL
        );
    """)
    con.register("lng_df", lng)
    _exec(con, "INSERT INTO element_program_top SELECT * FROM lng_df;")
    con.unregister("lng_df")
    _exec(con, "CREATE INDEX idx_ept_el ON element_program_top(element_id);")
    _exec(con, "CREATE INDEX idx_ept_pr ON element_program_top(program);")
    _log(f"  element_program_top: {len(lng):,} rows "
         f"(<= {TOP_PROGRAMS_PER_ELEMENT}/element, weight >= {MIN_STORED_WEIGHT})")


def load_genome_programs(con, root: Path, k: int):
    """Program summary, carrying BOTH enrichment columns and reproducibility."""
    s = pd.read_csv(root / f"nmf.k{k}.summary.tsv", sep="\t")
    t = pd.read_csv(root / f"nmf.k{k}.top_tfs.tsv", sep="\t")
    s = s.rename(columns={"median_cosine": "seed_stability"})
    # A program pinned to a handful of elements reconverges perfectly, so seed
    # stability alone marks it reproducible. 22 of 140 have <100 elements and
    # cover 0.1% of the data between them; flag rather than silently include.
    s["substantive"] = (s.n_elements >= 100) & (s.seed_stability >= 0.90)
    _exec(con, "DROP TABLE IF EXISTS genome_programs;")
    con.register("gp_df", s)
    _exec(con, "CREATE TABLE genome_programs AS SELECT * FROM gp_df;")
    con.unregister("gp_df")
    _exec(con, "CREATE INDEX idx_gp_prog ON genome_programs(program);")
    _log(f"  genome_programs: {len(s):,} rows "
         f"({int(s.substantive.sum())} substantive)")

    _exec(con, "DROP TABLE IF EXISTS genome_program_tf_top;")
    con.register("gt_df", t)
    _exec(con, "CREATE TABLE genome_program_tf_top AS SELECT * FROM gt_df;")
    con.unregister("gt_df")
    _exec(con, "CREATE INDEX idx_gtt_prog ON genome_program_tf_top(program);")
    _exec(con, "CREATE INDEX idx_gtt_tf   ON genome_program_tf_top(tf);")
    _log(f"  genome_program_tf_top: {len(t):,} rows")


def load_program_families(con, root: Path, k: int):
    """Named families over the programs -- the app's browsable vocabulary.

    Families group programs by CO-OCCURRENCE across elements, not by shared
    TFs. That distinction matters for how the UI should describe them: PRC2 and
    PRC1.1 are one family here because they occupy the same domains, not
    because they share subunits (they share almost none). A family is "these
    programs mark the same places", which is a claim about chromatin, not about
    protein complexes.
    """
    fdir = root / f"program_families.k{k}"
    if not fdir.exists():
        _log(f"  no {fdir.name}; skipping families "
             f"(run pipeline/genome_program_families.py)")
        return
    pf = pd.read_csv(fdir / "program_family.tsv", sep="\t")
    fs = pd.read_csv(fdir / "family_summary.tsv", sep="\t")
    lab_p, term_p = fdir / "family_labels.tsv", fdir / "family_terms.tsv"
    if lab_p.exists():
        fs = fs.merge(pd.read_csv(lab_p, sep="\t")[
            ["family", "label", "term", "lib", "q", "overlap", "set_size",
             "named"]], on="family", how="left")

    _exec(con, "DROP TABLE IF EXISTS program_families;")
    con.register("fs_df", fs)
    _exec(con, "CREATE TABLE program_families AS SELECT * FROM fs_df;")
    con.unregister("fs_df")
    _exec(con, "CREATE INDEX idx_pf_family ON program_families(family);")
    _log(f"  program_families: {len(fs):,} rows"
         + (f", {int(fs.named.sum())} named by enrichment"
            if "named" in fs.columns else ""))

    # Every significant term per family, not just the headline. A family is
    # usually enriched for several related terms, and overlap_tfs names which
    # members drove each call -- that is the evidence, and it lets a reader
    # judge a 3/7 hit instead of taking the label on trust.
    if term_p.exists():
        ft = pd.read_csv(term_p, sep="\t")
        _exec(con, "DROP TABLE IF EXISTS family_terms;")
        con.register("ft_df", ft)
        _exec(con, "CREATE TABLE family_terms AS SELECT * FROM ft_df;")
        con.unregister("ft_df")
        _exec(con, "CREATE INDEX idx_ft_family ON family_terms(family);")
        _exec(con, "CREATE INDEX idx_ft_lib ON family_terms(lib);")
        _log(f"  family_terms: {len(ft):,} rows "
             f"(median {int(ft.groupby('family').size().median())} per family)")

    # attach the family to each program so a program page can name its family
    # without a second query
    con.register("pfm_df", pf[["program", "family"]])
    _exec(con, "ALTER TABLE genome_programs ADD COLUMN family INTEGER;")
    _exec(con, """
        UPDATE genome_programs SET family = (
            SELECT m.family FROM pfm_df m WHERE m.program = genome_programs.program
        );""")
    con.unregister("pfm_df")
    n = con.execute(
        "SELECT COUNT(*) FROM genome_programs WHERE family IS NULL").fetchone()[0]
    if n:
        raise SystemExit(f"{n} programs have no family -- program_family.tsv "
                         f"and the factorization disagree")
    _exec(con, "CREATE INDEX idx_gp_family ON genome_programs(family);")
    _log(f"  genome_programs.family populated for all "
         f"{len(pf):,} programs")


################################################################################
# Execution ####################################################################
################################################################################
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--genome-dir", required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--db", default="data/canonical_promoter.duckdb")
    args = ap.parse_args()
    root = Path(args.genome_dir)

    con = duckdb.connect(args.db)
    _log(f"db {args.db}")
    el = load_elements(con, root)
    load_element_programs(con, root, args.k, len(el))
    load_genome_programs(con, root, args.k)
    load_program_families(con, root, args.k)

    # what a gene page will actually run
    q = con.execute("""
        SELECT e.stratum, COUNT(*) n, COUNT(DISTINCT e.nearest_gene_name) genes
        FROM elements e GROUP BY e.stratum ORDER BY n DESC""").fetchdf()
    print()
    _log("=== elements by stratum ===")
    print(q.to_string(index=False))
    ex = con.execute("""
        SELECT e.stratum, e.dist_to_tss, e.n_tfs_assigned,
               e.n_tss_comparably_close AS rivals, p.dominant_program AS prog
        FROM elements e JOIN element_program p USING (element_id)
        WHERE e.nearest_gene_name = 'SOX2'
        ORDER BY ABS(e.dist_to_tss) LIMIT 8""").fetchdf()
    total = con.execute(
        "SELECT COUNT(*) FROM elements WHERE nearest_gene_name = 'SOX2'"
    ).fetchone()[0]
    print(f"\n  sample gene-page query (SOX2), {len(ex)} of {total} elements:")
    print(ex.to_string(index=False))
    _write_genome_manifest(con, root, args.k, Path(args.db))
    con.close()
    return 0


def _write_genome_manifest(con, root: Path, k: int, db_path: Path):
    """Merge the genome layer into manifest.json.

    The Methods tab reads the manifest. Without this it described a build with
    no programs at all -- build_app_db.py counts only its own tables, so it
    recorded n_programs None while the app served 140 of them, and the promoter
    valid_chroms still listed Y and MT which genome discovery excludes.
    """
    import json
    mf = db_path.parent / "manifest.json"
    if not mf.exists():
        _log(f"  no {mf.name}; skipping manifest merge")
        return
    m = json.loads(mf.read_text())
    q = lambda s: con.execute(s).fetchone()[0]
    m["genome"] = {
        "k": k,
        "n_elements": q("SELECT COUNT(*) FROM elements"),
        "n_programs": q("SELECT COUNT(*) FROM genome_programs"),
        "n_programs_substantive":
            q("SELECT COUNT(*) FROM genome_programs WHERE substantive"),
        "n_families": q("SELECT COUNT(*) FROM program_families"),
        "n_families_named":
            q("SELECT COUNT(*) FROM program_families WHERE named")
            if con.execute("SELECT COUNT(*) FROM information_schema.columns "
                           "WHERE table_name='program_families' AND "
                           "column_name='named'").fetchone()[0] else None,
        "strata": {s: n for s, n in con.execute(
            "SELECT stratum, COUNT(*) FROM elements GROUP BY 1").fetchall()},
        "min_support": 11,
        "min_support_basis": "circular-shift null, 5% FDR genome-wide",
        "valid_chroms": ["1-22", "X"],
        "excluded_chroms": ["Y", "MT"],
        "source": str(root),
    }
    m.setdefault("counts", {})["n_programs_served"] = m["genome"]["n_programs"]
    mf.write_text(json.dumps(m, indent=2))
    g = m["genome"]
    _log(f"  manifest: genome layer recorded (k={g['k']}, "
         f"{g['n_elements']:,} elements, {g['n_programs']} programs, "
         f"{g['n_families']} families)")


if __name__ == "__main__":
    raise SystemExit(main())
