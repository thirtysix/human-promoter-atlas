# Choosing the assignment threshold

`MIN_SCORE_ASSIGN = 250`, chosen 2026-08-01. This records why, including the
approaches that turned out not to work, so the number is reproducible rather
than folklore.

## The problem

ChIP-Atlas peak score is −10·log₁₀(Q), capped at 1000. So `score ≥ 500` **is**
`Q < 1E-50`: the significance tier and the assignment score are two thresholds
on the same quantity, applied at different stages.

- **discovery** — which peaks define where modules are (the tier)
- **assignment** — which peaks are strong enough to name a TF in a module (the score)

The original atlas read the Q<1E-50 tier *and* filtered at score ≥ 500. Every
peak in that input already scored ≥ 500 by construction, so **the filter was
inert** — while the Methods page described a working two-tier design. Moving to
the Q<1E-5 tier made it bite for the first time, and hard: it discards 81% of
in-window peaks and drops 16,065 of 109,670 discovered modules (14.6%) for
having no TF above the bar — precisely the weak, cell-type-restricted modules
the looser tier was adopted to capture.

## Two criteria that could not decide it

**Agreement with the previous atlas is circular.** The premise of loosening the
tier is that Q<1E-50 under-counts sparse factors. Optimising for agreement with
it guarantees no improvement is possible. It measures how much changed, not
what is right.

**GO enrichment gives a direction but no optimum, and is confounded.** Lower
thresholds produce larger program gene sets (median `fg_total` 2,934 at score
500 vs 3,206 at 250) and therefore more statistical power regardless of truth.
Odds ratio and term distinctness are more robust than q-values, but there is
still no maximum to hit.

## The criterion that worked, and its own trap

Split each TF's ChIP experiments into two disjoint halves (by SRX hash, per TF),
hold the modules fixed from the full build, and ask what fraction of half-A
assignments reappear in half-B. Real binding replicates; a stray peak does not.
This needs no external data and makes no assumption about which atlas is right.

**Raw replication is not a precision measure.** As the score falls, both halves
assign a TF to nearly every module and overlap almost perfectly by construction
— so raw replication rises toward 1, rewarding exactly the saturation it should
detect. The first run duly reported that 56 of 90 TFs "preferred" the loosest
setting. Corrected against the overlap two independent draws of the same sizes
would give, the picture inverts.

## Results (q1e-5, whitelist axis, 589 TFs with enough assignments to judge)

| score | Q | median adj. replication | assignments | median TFs/module | modules >200 TFs |
|------:|---|------------------------:|------------:|------------------:|-----------------:|
| 500 | 1E-50 | 0.754 | 1,322,162 | 7 | 0.4% |
| 400 | 1E-40 | **0.755** | 1,716,618 | – | – |
| 300 | 1E-30 | 0.745 | 2,353,066 | – | – |
| **250** | **1E-25** | 0.740 | 2,842,213 | **13** | 5.1% |
| 100 | 1E-10 | 0.728 | 6,193,949 | 33 | – |
| 50 | 1E-5 | 0.725 | 6,884,591 | 38 | 16.6% |

Reference: the published Q<1E-50 atlas has median 12 TFs/module and 0.5% of
modules above 200 TFs.

**Mean corrected replication is flat to three decimals across the whole range
(0.676–0.680).** Assignments admitted at low scores replicate across disjoint
experiments about as well as the strictest ones, so score 500 was not buying
precision — it was discarding evidence.

## Why not no threshold at all

Score 50 is equivalent to no filter (the Q<1E-5 input floor is exactly 50, with
zero peaks below it). It scores **best** on every GO metric — median odds ratio
2.12 vs 2.04 at 250 (k=12), reliance on generic GO sets 63% vs 70%, and higher
significance — and retains 100% of modules.

It was rejected on a failure mode GO cannot see. GO scores program *gene* sets,
which only improve as assignments grow; nothing in it penalises an implausibly
promiscuous module. Without the filter, **one module in six is assigned more
than 200 TFs**, and the worst is credited with 1,194 of 1,311 assayed factors
inside a ~177 bp window. Corroborating this, at k=20 one program
(PIAS1/MAU2/YAP1/MTOR/PRMT1/KMT2C) has **zero** significant terms out of 5,709
tested across 1,865 modules — a noise sink. It would also degrade two shipped
features: a per-transcript view listing 1,194 factors is unreadable, and the
TF×TF network approaches complete, since one 200-TF module alone contributes
19,900 pairs.

This argument is a plausibility judgement, not a measurement. It is the step
doing the real work in ruling out no-threshold, and should be read as such.

## Why not per-TF thresholds

Per-TF optima are real and heterogeneous (bimodal, spanning 50–500), but:

- **they buy little** — median corrected replication 0.788 vs 0.757 for a good
  global threshold, with 204 of 589 TFs gaining more than 0.05;
- **they cannot be extended** — threshold tracks experiment count at Spearman
  −0.19, with interquartile ranges spanning nearly the full range in every
  stratum, so the 722 TFs too sparse to split have nothing to extrapolate from;
- **they cost interpretability** — the module×TF matrix feeds NMF and the TF×TF
  co-occurrence network, both of which treat columns as comparable. Columns
  built at different evidence bars are not.

The machinery is retained (`pipeline/splithalf_threshold.py`), and the full
per-TF curve is in `docs/splithalf_threshold.q1e-5.tsv`, so this is revisitable.

## Reproducing

```bash
hpc/01_stage.sh --tier q1e-5                       # stage peaks
sbatch hpc/sweep_min_score.slurm                   # SCORES="500 300 250 200 100 50"
sbatch hpc/splithalf.slurm                         # split by SRX, scan both halves
python pipeline/splithalf_threshold.py --modules <build>/tss_modules \
    --half-a <A>/tss_modules --half-b <B>/tss_modules
python pipeline/compare_enrichment.py s500=<...> s250=<...> s50=<...> --k 20
```

---

# Choosing k on the rebuilt atlas

Re-run of `tss_modules_select_k` on the q1e-5 / all(1,793 TF) / score-250 build,
20 seeds per rank, collapsed fits re-seeded (see below).

| k | ARI med | ARI min | ARI p25 | cophenetic | recon. err |
|---:|---:|---:|---:|---:|---:|
| 5 | 0.384 | 0.141 | 0.318 | 0.8677 | 1761.0 |
| 8 | **0.509** | 0.232 | 0.440 | 0.9028 | 1727.6 |
| **10** | 0.492 | 0.267 | 0.436 | 0.9003 | 1710.2 |
| 12 | 0.442 | 0.267 | 0.387 | 0.9067 | 1695.8 |
| 15 | 0.461 | 0.309 | 0.413 | 0.9039 | 1677.3 |
| 18 | 0.427 | 0.308 | 0.387 | **0.9152** | 1661.2 |
| 20 | 0.446 | 0.332 | 0.410 | 0.9137 | 1651.8 |
| 25 | 0.420 | 0.298 | 0.384 | 0.8747 | 1630.3 |

**The published k=10 survives the rebuild.** ARI peaks at k=8 (0.509) with k=10
second at 0.492 -- a 3% difference, well inside the spread of either. The
cophenetic curve is nearly flat from k=8 to k=20 (0.900-0.915, a 0.015 range)
and so does not discriminate; its apparent peak at k=18 is not a meaningful
margin. Reconstruction error falls monotonically with k, as it must, and carries
no information about the right rank.

So: **k=8-10 by stability, and k=10 is defensible.** Keeping it also avoids
renumbering every program in a published atlas for a change that is within noise.

**Honest limitation.** Median ARI is 0.38-0.51 at *every* rank. The NMF partition
is only moderately reproducible across random restarts on this data, whatever k
is chosen. That is a property of the data, not of the selection, and it should
temper how firmly any single program assignment is read.

## The metric had to be fixed first

The first run of this table was not trustworthy. With `init="random"` and the
multiplicative-update solver, some seeds drive a component to zero; those fits
return `reconstruction_err_ = nan` and an ARI of exactly 0.000 against any
healthy partition. `err=nan` and `min_ari=0.000` coincided **exactly** at
k=12/15/18/25 -- solver failures being scored as instability.

Switching the initialiser to `nndsvdar` removed the collapses but broke the
metric in the other direction: every seed then starts from nearly the same
NNDSVD point, and median ARI rose to 0.981/0.984 at k=5/8. A curve flat near 1.0
at every rank cannot choose between ranks.

Measured at k=18, 20 seeds:

| option | collapses | ARI med | min |
|---|---:|---:|---:|
| random + mu | 1 | 0.420 | 0.000 |
| random + cd | 0 | 0.233 | 0.120 |
| nndsvdar + mu | 0 | 0.819 | 0.581 |
| **random + mu, reject & re-seed** | 1 (re-seeded) | **0.425** | **0.308** |

Re-seeding is the minimal intervention: median unchanged, minimum off the floor.
The rebuilt run needed 5 re-seeds across 160 fits (k=12, 15, 18 one each; k=25
two) and has **zero** ARI floors at 0.000, against four before.

The cophenetic stage needed the same fix for a subtler reason: a collapsed fit
puts every subsampled module in one cluster, which *inflates* the consensus
matrix -- the worse the fit, the better the cophenetic correlation would look.
The dispersion column moved from 0.08-0.23 to 0.51-0.86 once collapsed fits were
excluded, confirming those numbers were contaminated too.

**Still using `init="random"` elsewhere:** `tss_modules.001.py`,
`tss_modules_k10.py` and `tss_archetypes.001.py`. They are safe only because they
pin seed 0, which happens not to collapse. Switching them would change the
published factorisation (programs renumbered, membership shifted), so it belongs
with a deliberate rebuild rather than a bug fix.
