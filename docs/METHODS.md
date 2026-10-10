# Methods

This document is the method spec: what was done and why,
one paragraph per decision. For the numbers that came out of it, see
[FINDINGS.md](FINDINGS.md). For what the numbers can't be stretched to
claim, see [LIMITATIONS.md](LIMITATIONS.md).

## Cohort

GDSC1 and GDSC2 are two independent dose-response screens run on
overlapping sets of cell lines, at different times, sometimes with
different assay protocols. Both are used: GDSC1 as the training screen
(twice the dynamic range), GDSC2 as an external validation screen for the
same drug on largely the same lines. Haematological lineages (leukaemias,
lymphomas, myelomas; `config.HAEM`) are excluded from every modelling
step: the DTP/regenerative programmes this project studies are defined in
solid epithelial tumours, and blood cancers are both biologically
different and easy for a model to identify by tissue-of-origin alone,
which would inflate apparent performance for the wrong reason.

## Target

**AUC**, not LN_IC50. GDSC's IC50 is right-censored (many solid lines are
never killed at screenable concentrations), which makes IC50 both
noisier and harder to interpret. AUC degrades more gracefully under
censoring. Stage 01 computes how well GDSC1 and GDSC2 agree with each
other on the same lines (`lib.stats.screen_agreement`), on the cohort
the models are fitted on: solid lineages, with expression, measured in
both screens (n=781). It writes Pearson and Spearman for AUC and LN_IC50
to `screen_agreement.csv`, and `lib.io.load_ceiling()` reads it back.
That agreement is the ceiling on what any model predicting one screen
from baseline expression could achieve, and every `pct_of_ceiling` is
a fraction of it. Stage 05 computes each other drug's ceiling on the
same line set.

## Features

Baseline (pre-treatment) bulk RNA-seq, `rsem_tpm` from Cell Model
Passports. That value is **already log2(TPM+1)**, confirmed empirically
(values top out around 18, per-sample sums around 6×10⁴ rather than the
~10⁶ raw TPM would give), so it is not re-logged. Genes missing in more than
10% of lines are dropped (`config.MAX_GENE_MISSING`); the small remainder
is median-filled, an unsupervised step applied identically to every fold
so it cannot leak test information (`lib/io.py::load_screen`).

## Model

`SelectKBest(f_regression, k=1500) → StandardScaler → ElasticNet(l1_ratio=0.5)`
(`lib/modeling.py::elasticnet_pipeline`), evaluated by 5×5 repeated
K-fold CV (`repeated_cv`): 5 folds, 5 repeats with different random
splits. `SelectKBest` lives **inside** the pipeline rather than as a
pre-filtering step, so gene selection is refit on the training fold only
every time; selecting by correlation with the full target before
splitting would leak.

**Nested tuning.** Alpha is chosen inside every outer training fold by a
3-fold inner CV on that fold's rows only (`TunedElasticNet`, which wraps
`ElasticNetCV`), so every reported r is still an outer-CV estimate. The
grid depends on the table's width: tables wider than
`config.FEW_FEATURES` (50) columns search `config.ELASTICNET_ALPHAS`,
0.002 to 0.5 in 13 log steps; narrower ones (the four-module models)
search `config.ELASTICNET_ALPHAS_FEW`, 1e-5 to 0.5 in 22 steps. The
gene-level floor is a run-time compromise: below 0.002 the coordinate
descent on 1,500 genes slows sharply, and no gene-level fit chose the
floor. `l1_ratio` stays at 0.5; tuning it over {0.2, 0.5, 0.9} made
each fit about ten times slower and did not raise out-of-fold r on
GDSC1 5-FU. Ridge
models (B4 and the module baselines) use `RidgeCV` over
`config.RIDGE_ALPHAS` (0.01 to 1000) the same way.

Every row records the median chosen alpha (`alpha_median`) and the
fraction of folds whose choice sat on a grid edge (`alpha_edge_frac`).
Two edges are hit, and neither changes a conclusion. The four-feature
modules models pick the floor 1e-5 in 20% (GDSC1 raw) to about half
(GDSC2) of folds: with four features that is ordinary least squares, so
a lower floor cannot change the fit.
The CRC-only Ridge models (43 lines) pick 100 to 1000 in 12 to 44% of
folds: the inner CV finds nothing to fit and shrinks towards a constant.

The previous fixed penalty (`config.ELASTICNET_PARAMS`, alpha=0.02) is
kept as a reference: stage 02 refits the transcriptome models with it on
the same splits (`fixed_elasticnet_pipeline`), and `model_comparisons.csv`
tests tuned against fixed with the corrected t-test.

`lib.modeling.cv_oof` keeps every out-of-fold prediction, and each
result row reports:

- `r_mean`, `r_sd`: Pearson r per repeat, then mean and SD over repeats.
- `r_lo`, `r_hi`: 2.5th and 97.5th percentiles of a cell-line bootstrap
  (`lib.stats.bootstrap_lines_r`, 2,000 draws). Each draw resamples lines
  and rescores every repeat on them. The predictions are held fixed, so
  the interval reflects which lines were screened, not training noise.
  It replaces a percentile over the 5 per-repeat r values, which is close
  to their minimum and maximum and says nothing about the lines.
- `rho_mean`: Spearman rho, mean over repeats.
- `cindex_mean`: Harrell's C (`lib.stats.concordance_index`), the share of
  line pairs the prediction orders correctly, mean over repeats.

**Model comparisons** (`model_comparisons.csv`). Two models scored on the
same lines and splits are compared on their per-split r differences (25
splits) with the corrected resampled t-test of Nadeau and Bengio (2003):
splits of repeated CV share training lines, so the variance is inflated
by (1/J + n_test/n_train), with J-1 degrees of freedom. The pairs are the
transcriptome and modules models against the B4 bar (lineage +
proliferation, a Ridge refit on the same solid lines and splits), and the
transcriptome against the modules model, raw and de-confounded. q is BH
across the four.

**Leave whole lineages out** (`cv_leave_lineage_out`). Fold-internal
de-confounding still trains on every tissue it tests on. Here each
lineage is predicted by a model fitted on every other lineage. Two lines
with no lineage label form one group. The rows report the pooled r over
all lines (which still rewards predicting a new tissue's average) and
the median within-lineage r over held-out lineages with at least
`MIN_LINEAGE_N` lines (which does not).

## Lineage de-confounding

Fold-internal only. Within each CV fold, lineage means (for both X and y)
are estimated on the **training rows alone**, then subtracted from both
train and test rows of that fold; y is also divided by the training
lineage SD. Lineages present in a test fold but absent from that fold's
training rows are dropped: their mean can't be estimated without seeing
them. This is the most safety-critical piece of code in the project:
computing lineage means over the **whole dataset** uses test-fold targets
and reports a leaked r=0.279 for the de-confounded association, against
r=0.182 for the fold-internal version (both at the fixed alpha=0.02).
That gap is the size of the leak.
`tests/test_library.py` has a synthetic regression guard against this
failure mode: on data that is pure lineage-encoded noise (y is
literally the lineage mean plus noise, X is independent random noise),
the de-confounded CV must score near zero.

## CRC handling

Colorectal (COREAD) is the lineage 5-FU is actually used to treat
clinically, but there are only 43–46 CRC lines in either screen, far too
few to train a transcriptome-wide model on directly. Rather than either
ignoring CRC or overfitting a CRC-only model, `transfer_to_crc` compares
three approaches: (a) a small Ridge model on the four modelled signature
scores, trained and cross-validated on CRC alone; (b) the pan-solid
ElasticNet model, trained on every solid line **except** CRC and applied
directly to CRC (transfer); (c) the CRC-only Ridge model with the
transfer prediction added as one extra feature. Transfer consistently
outperforms local training, the expected result given n≈43–46.

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
Monte Carlo null (200 draws) that was found to be far too few: rescoring
with a different seed gave r=0.74 for MYC against itself, meaning most of
that score was sampling noise, not a real signal. The analytic null is
exact, deterministic and seed-free by construction.

`DTP` is bidirectional: `DTP_up − DTP_down`, which cancels technical
variation that shifts every gene the same way in a sample (library size,
RNA quality, tumour purity). Genes are matched to each dataset's gene
universe with light HGNC synonym resolution (`resolve_symbols`),
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

`lib/stats.py::hypergeometric_enrichment` tests whether the non-zero-coefficient
genes from the final de-confounded model's full fit (156 since nested tuning)
are enriched for each curated signature, against the universe of every
gene the model could have selected from (the missingness-filtered,
haem-excluded expression matrix, not the raw unfiltered matrix and not
just the query size). `p = scipy.stats.hypergeom.sf(k-1, N, K, n)`.

The overlap test ignores direction: DTP_up genes with sensitivity-
direction coefficients would count the same as ones with resistance-
direction coefficients. `lib.stats.signed_set_test` adds direction on two
vectors, positive meaning higher AUC (resistant):

- `coef_*`: the model's non-zero coefficients. `coef_n_resistance` of
  `coef_n` set genes carry positive coefficients; `coef_p_sign` is a
  binomial test against one half, `coef_p_rank` a Mann-Whitney test of
  the set's coefficients against the other model genes.
- `genome_*`: every gene's correlation with the de-confounded target
  (lineage-centred expression against within-lineage z-scored AUC, on
  lines in lineages with at least `MIN_LINEAGE_N` lines), the preranked
  analogue of GSEA (`lib.stats.set_shift_test`). `genome_mean_r` is the
  set's mean correlation and `genome_background_r` the genome-wide mean
  within each expression decile, weighted by the set's genes per decile:
  curated genes are well expressed, and well-expressed genes carry higher
  correlations than the genome as a whole. `genome_shift` is the
  difference. `genome_p_perm` comes from 1,000 permutations of the target
  within lineage, recomputing both terms each time, so correlation among
  a set's genes is kept. A random-gene-set null would not keep it and
  called every set significant. The test is competitive: it asks whether
  a set tracks resistance more than similar genes do.

## Assay-resolution stratification

Roughly 29% of solid lines sit at AUC>0.95, the assay's practical
ceiling: 5-FU barely affects them at any screenable concentration.
Pooling those lines with genuinely responsive ones dilutes the apparent
signal in both directions, so results are reported separately for
responsive lines (AUC≤0.95) and ceiling lines (AUC>0.95).

## Affine recalibration

The GDSC1-trained model is systematically compressed and offset when
applied to GDSC2, because the two screens have different AUC
distributions. `y_cal = a·ŷ + b`, fit by `np.polyfit` on **half** of
GDSC2 and evaluated on the other half (so the reported evaluation number
is honest rather than fit-and-scored on the same rows). For the
algebraically optimal a and b, R² becomes exactly r²; an affine
transform cannot change r. This is a **display fix** (so a shown
predicted AUC isn't systematically ~0.07 too low), not a modelling
improvement, and it requires labelled data from the target screen, which
a genuinely new, unlabelled dataset would not have.

## MSI and TP53 covariates

Partial correlation (`lib/stats.py::partial_corr`, regressing both
DTP and AUC on the covariate then correlating the residuals) tests
whether DTP's association with resistance survives adjustment for
microsatellite instability (MSI) and, separately, TP53 mutation status.
This is a genuine confound test, not a feature the model itself uses;
MSI/TP53 are not inputs to the ElasticNet model. Stage 03 uses every
COREAD line with an MSI call; stage 08 also needs methylation and so
loses one line, and it reports the binary-MSI adjustment on both line
sets so the two stages can be reconciled.

## MSI as an effect modifier, and MMR coupling (stage 03, post hoc)

Adjustment asks whether MSI explains the DTP association. Stage 03 also
asks whether the association differs by MSI status
(`msi_effect_modification.csv`):

- DTP against AUC within MSS and within MSI COREAD lines, and the MSS
  minus MSI difference in r (bootstrap interval of the difference,
  Fisher z p for two independent correlations);
- OLS `AUC ~ DTP * MSI`, with DTP z-scored so coefficients are AUC per
  SD, in COREAD and in all solid lines with lineage fixed effects (56 and
  57 MSI lines there, against 16 in COREAD);
- the tuned transcriptome ElasticNet fitted on non-colorectal solid lines
  and scored on COREAD, MSS and MSI lines separately. Fitting on 27 to 30
  MSS lines alone was not attempted: the 43-line CRC-only model is already
  near zero.

Then DTP against mismatch repair (`dtp_mmr_coupling.csv`). The MMR score
is the mean of the z-scored expression of `config.MMR_GENES` (MLH1, MSH2,
MSH6, PMS2) within each analysed set; burden is log10 of
`mutational_burden` (`config.COVARIATES_NUM`). Each of DTP vs MMR score,
DTP vs burden and DTP vs AUC given burden is run in COREAD, COREAD MSS,
COREAD adjusted for MSI, and all solid lines adjusted for lineage with and
without MSI. MSI lowers MMR expression and raises burden, so only the
within-MSS and MSI-adjusted rows can separate a coupling from MSI again.
MMR score against MSI is the positive control.

Plain correlations carry the 500-resample line bootstrap interval;
partial correlations carry a Fisher z interval with one degree of freedom
removed per control (`lib/stats.py::fisher_ci`). Both tables are BH
corrected within screen and marked `declared = post hoc`. Pan-solid rows
need an MSI call and a lineage label, and the MMR table also a burden
value.

## Multiplicity

`config.CONFIRMATORY` names the tests the DTP claim rests on: DTP against
5-FU AUC within COREAD (GDSC1, replicated in GDSC2), DTP against 5-FU
with general chemosensitivity removed (`drug_specificity.csv`), and DTP
in the adjusted GPL570 clinical model. This list was written down after
these results had been seen, so it is a declared policy, not a
pre-registration. Every other test is exploratory. Each table carries a
Benjamini-Hochberg q (`lib.stats.bh_q`) within its own family: one
screen in `within_crc_association.csv`, one drug and adjustment in
`drug_specificity.csv`, one cohort and model in `clinical_validation.csv`,
and a `confirmatory` column (`lib.stats.confirmatory_mask`). Because the
confirmatory tests are few and small (n=43 to 189), the DTP case is
argued from cross-screen replication of sign and effect size, with
bootstrap CIs per screen (`same_sign`, `ci_overlap`), rather than from
any single p.

## Modules model across screens

Stage 03 refits the modules model on GDSC2 with one thing changed at a
time (responsive lines only, the lines shared with GDSC1, the target
scaled to unit SD), applies the GDSC1 fit to GDSC2 without refitting,
and splits every r into between- and within-lineage parts over lineages
with at least `config.MIN_LINEAGE_N` lines
(`module_screen_discrepancy.csv`).

It then refits the transcriptome and modules models on both screens,
raw and de-confounded, under four rules: the fixed alpha with the
target as is, z-scored on the training rows, or z-scored then rescaled
to the SD of GDSC1 5-FU solid AUC (the target the fixed alpha was
chosen on), through `lib.modeling.scaled_target`; and nested tuning
(`penalty_scale_sensitivity.csv`).

## Multidrug residualization

GDSC1 screened 352 other drugs on at least 300 of the same solid lines. "General
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
the same size gives, drawn from the same expression matrix. This is necessary
because rank-based scores are compositional: if cell-cycle genes collapse
under treatment, every other gene's rank rises mechanically, which alone
could manufacture an apparent "induction" signal for any feature. This
asks a coarser question than `background_score`'s analytic null (does a
*time-trend* exceed a random-gene-set trend?), for which 200 Monte Carlo
draws are adequate, unlike scoring a single sample. It stays Monte Carlo,
with an empirical p measured as deviation from the null's own mean and
one continuous `rng` stream shared across all features.

## Other drugs (stages 05 and 06)

Oxaliplatin, SN-38, irinotecan and cisplatin (`config.DRUGS`) get the same
target construction as 5-FU. Stage 05 picks one screen per drug from its
QC (`multidrug_qc.csv`): coverage, assay-ceiling fraction and GDSC1/GDSC2
agreement where both exist. A drug below `config.MIN_N_STOP` lines stops
the run rather than being fitted. It then runs the stage 02 pipeline,
with alpha tuned per fold by the same nested CV, on each drug, and tests
each model's selected genes for signature enrichment. Stage 06 works on
GDSC2 (the one screen with all five drugs) and asks three things: does
DTP track each drug after removing general chemosensitivity (partial
correlation, with and without MSI), do the drugs' models pick overlapping
genes (`hypergeometric_overlap` against the shared universe), and does
5-FU track oxaliplatin more than cisplatin (`bootstrap_r_difference`,
2,000 draws, the two arms resampled independently).

## Clinical cohorts (clinical_prep.R and stage 07)

Five GEO series of colorectal patients treated with FOLFOX. `clinical_prep.R`
downloads them, keeps the FOLFOX arm of each, and checks every regimen
against the series' own GEO summary rather than its column labels. The
four Affymetrix studies (GPL570: GSE28702, GSE19860, GSE69657, GSE72970)
are ComBat-corrected together. The Agilent study (GSE104645, GPL6480) is
two-colour log-ratio data, a different technology rather than a batch, so
it is never pooled. Covariates: tumour purity (ESTIMATE), CMS subtype
(CMScaller), and an MSI-like call from CMScaller's MSI template, since no
series reports a molecular MSI test. Stage 07 scores every sample with
`lib.cohorts.score_cohort` (the same rank-based scoring as the cell
lines) and fits a logistic regression of response on each z-scored
score, unadjusted and adjusted for purity, CMS, MSI and study.

Random gene sets often associate with outcome in expression data, so each
adjusted test is also compared with 1,000 random gene sets of the same
resolved size (two sets, up and down, for DTP), drawn from that cohort's
genes, scored with the same `background_score` and fitted with the same
adjusted logit on the same rows. `p_random_null` is the share of random
sets whose Wald |z| is at least the observed one.

## Methylation (the two methylation prep scripts, stages 08 and 09)

The GDSC 450K array (GEO GSE68379, processed beta values). `methylation_prep.R`
masks betas with detection p ≥ 0.01 and removes cross-reactive, SNP and
sex-chromosome probes; stage 08 aggregates probes to promoter
(TSS1500/TSS200) and gene-body matrices, as beta and M values. It then
repeats the COREAD MSI adjustment with continuous MLH1 promoter
methylation in place of the binary MSI call, and fits the stage 02
pipeline on methylation alone and as a late fusion (average of the
expression and methylation predictions), and scores DTP on promoter
M values with the unchanged `lib.signatures` scoring. `methylation_context_prep.R` and
stage 09 split promoter methylation by CpG-island relation (island, N/S shore,
N/S shelf, open sea) plus an enhancer stratum, and rerun the
methylation-DTP test in each, BH-corrected per screen. At least
`config.OVERLAP_MIN` lines must have both expression and methylation.

## TF activity and chromatin (stages 10 and 11)

Stage 10 infers transcription-factor activity from expression with
decoupler (ULM on the CollecTRI regulon), then models response from TF
activity and tests TFs against DTP and AUC within COREAD. Ten TFs were
pre-specified (TEAD1 to 4, MYC, E2F1, TP53, SOX9, HNF4A, CDX2) and are
reported at raw p; every other TF is exploratory and BH-corrected as its
own pool. If CollecTRI cannot be downloaded the stage stops rather than
silently switching regulon. Stage 11 counts TCGA COAD ATAC-seq peaks
(distal and promoter) assigned to each gene and compares each signature
with 2,000 random gene sets matched on expression and gene length (5 × 5
bins).

Stage 11 then splits DTP_up two ways (post hoc,
`dtp_subprogramme_results.csv`): genes with a COAD distal fraction of at
least 0.5 against the rest of those with a peak, and genes shared with
RSC, Fetal or CBC against the DTP-only rest. The splits are made once on
canonical HGNC symbols and mapped into each dataset's resolved names.
Each half is scored as half minus DTP_down and read out three ways, each
through the code that produced the published DTP result: Pearson r with
5-FU AUC in COREAD (both screens), the HCT116 trend r
(`lib/timecourse.py`, shared with stage 04) and the Wald z in the
adjusted GPL570 response model (`lib/cohorts.py::AdjustedLogitZ`, shared
with stage 07's random-signature null). The whole DTP_up through the same
code reproduces each published value. The p is the share of 1,000 random
splits of the same pool, at the same sizes, whose absolute difference
between halves is at least the observed one; the eight tests are one BH
family. The GPL570 readout is skipped if the clinical cohorts are absent.

## Data provenance

`data/raw/` holds the raw inputs (~12GB, not in git; see the README
for sources). It is treated as read-only source of truth by every stage; nothing in
`stages/` ever writes there. `data/processed/` is entirely regenerated by
running the stages in numeric order (`make run`), after the R
preparation steps (`make r-prep`); nothing in it is
committed to git.
