# Methods

This is a clean reimplementation of `5FU-resistance-ML`, reproducing that
project's FINAL, validated methodology only — not the exploratory history
that got there. This document is the method spec: what was done and why,
one paragraph per decision. For the numbers that came out of it, see
[FINDINGS.md](FINDINGS.md). For what the numbers can't be stretched to
claim, see [LIMITATIONS.md](LIMITATIONS.md).

## Cohort

GDSC1 and GDSC2 are two independent dose-response screens run on
overlapping sets of cell lines, at different times, sometimes with
different assay protocols. Both are used: GDSC1 as the training screen
(twice the dynamic range), GDSC2 as an external validation screen for the
same drug on largely the same lines. Haematological lineages (leukaemias,
lymphomas, myelomas — `config.HAEM`) are excluded from every modelling
step: the DTP/regenerative programmes this project studies are defined in
solid epithelial tumours, and blood cancers are both biologically
different and easy for a model to identify by tissue-of-origin alone,
which would inflate apparent performance for the wrong reason.

## Target

**AUC**, not LN_IC50. GDSC's IC50 is right-censored — many solid lines are
never killed at screenable concentrations — which makes IC50 both
noisier and harder to interpret. AUC degrades more gracefully under
censoring. `config.CEILING_R = {"AUC": 0.604, "LN_IC50": 0.560}` records
how well GDSC1 and GDSC2 agree with each other on the same lines for each
target — the ceiling on what any model predicting one screen from
baseline expression could achieve, since that agreement bounds the
achievable correlation regardless of modelling quality.

## Features

Baseline (pre-treatment) bulk RNA-seq, `rsem_tpm` from Cell Model
Passports. That value is **already log2(TPM+1)**, confirmed empirically
(values top out around 18, per-sample sums around 6×10⁴ rather than the
~10⁶ raw TPM would give) — it is not re-logged. Genes missing in more than
10% of lines are dropped (`config.MAX_GENE_MISSING`); the small remainder
is median-filled, an unsupervised step applied identically to every fold
so it cannot leak test information (`lib/io.py::load_screen`).

## Model

`SelectKBest(f_regression, k=1500) → StandardScaler → ElasticNet(alpha=0.02,
l1_ratio=0.5)` (`lib/modeling.py::elasticnet_pipeline`), evaluated by 5×5
repeated K-fold CV (`repeated_cv`): 5 folds, 5 repeats with different
random splits, reporting the mean and 95% CI (2.5th/97.5th percentile)
across the 5 per-repeat Pearson r values. `SelectKBest` lives **inside**
the pipeline rather than as a pre-filtering step, so gene selection is
refit on the training fold only every time — selecting by correlation
with the full target before splitting would leak.

## Lineage de-confounding

Fold-internal only. Within each CV fold, lineage means (for both X and y)
are estimated on the **training rows alone**, then subtracted from both
train and test rows of that fold; y is also divided by the training
lineage SD. Lineages present in a test fold but absent from that fold's
training rows are dropped — their mean can't be estimated without seeing
them. This is the single most safety-critical piece of code in the
project: an earlier version of the original computed lineage means over
the **whole dataset**, which used test-fold targets and reported a
leaked r=0.279 for the de-confounded association. The corrected,
fold-internal version reports r=0.182 — that gap is the size of the leak.
`tests/test_modeling_leak.py` is a synthetic regression guard against
this exact failure mode: on data that is pure lineage-encoded noise (y is
literally the lineage mean plus noise, X is independent random noise),
the de-confounded CV must score near zero.

## CRC handling

Colorectal (COREAD) is the lineage 5-FU is actually used to treat
clinically, but there are only 43–46 CRC lines in either screen — far too
few to train a transcriptome-wide model on directly. Rather than either
ignoring CRC or overfitting a CRC-only model, `transfer_to_crc` compares
three approaches: (a) a small Ridge model on the four modelled signature
scores, trained and cross-validated on CRC alone; (b) the pan-solid
ElasticNet model, trained on every solid line **except** CRC and applied
directly to CRC (transfer); (c) the CRC-only Ridge model with the
transfer prediction added as one extra feature. Transfer consistently
outperforms local training — the expected result given n≈43–46.

## Signature scoring

Every signature (DTP up/down, RSC, CBC, Fetal, MYC, CellCycle) is scored
the same way, in one shared module (`lib/signatures.py`) used by both the
cross-sectional GDSC arm and the HCT116 time-course arm, so the two arms
can never silently drift onto different gene sets under the same name.
Scoring is rank-based: each sample's genes are converted to within-sample
percentile ranks (`rank_matrix`), then a signature's score is
`(observed mean rank − population mean rank) / null SD`, where the null
SD is computed **analytically** from the exact finite-population-
correction formula for the mean of `n` values drawn without replacement
from a population of `N` (`background_score`). This replaced an earlier
Monte Carlo null (200 draws) that was found to be far too few — rescoring
with a different seed gave r=0.74 for MYC against itself, meaning most of
that score was sampling noise, not a real signal. The analytic null is
exact, deterministic and seed-free by construction.

`DTP` is bidirectional: `DTP_up − DTP_down`, which cancels technical
variation that shifts every gene the same way in a sample (library size,
RNA quality, tumour purity). Genes are matched to each dataset's gene
universe with light HGNC synonym resolution (`resolve_symbols`) —
necessary because the HCT116 time-course was annotated at an older HGNC
vintage than current GDSC/signature symbols; unresolved, DTP_up recovery
was 91% in GDSC vs 67% in the time-course, i.e. the two arms would silently
score different gene sets under the same name. Alias resolution only
accepts a synonym match when it is unambiguous (exactly one universe
entry maps to that official symbol); ties are dropped, never guessed.
Literal `"None"`/`"NA"`/etc. entries in the signature source files (failed
upstream ID conversions written as text) are filtered and reported, not
silently counted as genes.

**DTP and RSC/CBC/CellCycle are modelled features; Fetal and MYC are
reference-only** (`config.MODELED_MODULES` vs `REFERENCE_MODULES`).
Fetal shares 128 of RSC's 232 genes and correlates r=0.96 with it; MYC
correlates r=0.83 with CellCycle, a proliferation proxy. Both are
reported as sensitivity checks, never modelled as if independent.

## Enrichment

`lib/stats.py::hypergeometric_enrichment` tests whether the ~413
non-zero-coefficient genes from the final de-confounded model's full fit
are enriched for each curated signature, against the universe of every
gene the model could have selected from (the missingness-filtered,
haem-excluded expression matrix — not the raw unfiltered matrix, and not
just the query size). `p = scipy.stats.hypergeom.sf(k-1, N, K, n)`. This
is newly-implemented code: the original project never computed this test
at all, only printed raw gene-name overlaps between the model and each
signature.

## Assay-resolution stratification

Roughly 29% of solid lines sit at AUC>0.95, the assay's practical
ceiling — 5-FU barely affects them at any screenable concentration.
Pooling those lines with genuinely responsive ones dilutes the apparent
signal in both directions, so results are reported separately for
responsive lines (AUC≤0.95) and ceiling lines (AUC>0.95).

## Affine recalibration

The GDSC1-trained model is systematically compressed and offset when
applied to GDSC2, because the two screens have different AUC
distributions. `y_cal = a·ŷ + b`, fit by `np.polyfit` on **half** of
GDSC2 and evaluated on the other half (so the reported evaluation number
is honest rather than fit-and-scored on the same rows). For the
algebraically optimal a and b, R² becomes exactly r² — an affine
transform cannot change r. This is a **display fix** (so a shown
predicted AUC isn't systematically ~0.07 too low), not a modelling
improvement, and it requires labelled data from the target screen, which
a genuinely new, unlabelled dataset would not have.

## MSI and TP53 covariates

Partial correlation (`lib/stats.py::partial_correlation`, regressing both
DTP and AUC on the covariate then correlating the residuals) tests
whether DTP's association with resistance survives adjustment for
microsatellite instability (MSI) and, separately, TP53 mutation status.
This is a genuine confound test, not a feature the model itself uses —
MSI/TP53 are not inputs to the ElasticNet model.

## Multidrug residualization

GDSC1 screened ~264 other drugs on largely the same lines. "General
chemosensitivity" is defined as the mean z-scored AUC across those other
drugs per cell line; 5-FU AUC is residualized on it via ordinary least
squares (`lib/stats.py::residualize`) to isolate the 5-FU-**specific**
component, and the transcriptome model is re-evaluated on that residual.
This is the same logic as lineage de-confounding, applied to a different
nuisance axis: measure the generic signal, remove it, see what survives.

## Arm B (HCT116 time-course)

A second, independent line of evidence: does 5-FU **induce** these
programmes over time within one cell line (HCT116), rather than merely
correlating with resistance across lines? Scored with the identical
`lib/signatures.py` pipeline as the cross-sectional arm. The
compositional control (`lib/stats.py::compositional_null`) tests whether
each feature's correlation with time exceeds what a random gene set of
the same size gives, drawn from the same expression matrix — necessary
because rank-based scores are compositional: if cell-cycle genes collapse
under treatment, every other gene's rank rises mechanically, which alone
could manufacture an apparent "induction" signal for any feature. This
control is a genuinely different test from `background_score`'s analytic
null (a coarser question — does a *time-trend* exceed a random-gene-set
trend — where 200 Monte Carlo draws is an adequate sample size, unlike
scoring a single sample) and is deliberately kept as Monte Carlo, ported
unchanged including its 200-draw count and its specific empirical-p
formula (deviation from the null's own mean, computed with one
continuous `rng` stream shared across all features, exactly as the
original ran it).

## Data provenance

`data/raw/` is a byte-identical copy of the original project's raw
inputs (~12GB), not a symlink, so this repo is fully self-contained. It
is treated as read-only source of truth by every stage — nothing in
`stages/` ever writes there. `data/processed/` is entirely regenerated by
running `stages/01` through `stages/10` in order; nothing in it is
committed to git.
