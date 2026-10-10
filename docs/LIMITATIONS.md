# Limitations

What the numbers in [FINDINGS.md](FINDINGS.md) do not show, collected in
one place rather than scattered as footnotes. None of these are fixed by
more careful analysis of the existing data; they are limits of the data
itself.

## GDSC2 is a re-measurement, not an independent holdout

GDSC2 largely re-screens the **same cell lines** as GDSC1, not new ones.
Agreement between the two screens (r=0.518 for AUC on the 781 solid
lines both measured, `screen_agreement.csv`)
demonstrates measurement robustness and provides an honest ceiling on
achievable r, but it is not evidence of generalisation to cell lines the
model has never seen anything about. The only genuinely new samples are
the colorectal transfer (lines held out of training) and the FOLFOX
patient cohorts, which carry their own caveat below.

## Arm B has no time-matched vehicle control

Untreated samples exist only at 0h; there is no untreated control at 6,
24 or 48h. Any change observed by 48h therefore confounds the effect of
5-FU with the effect of 48 more hours in culture (confluence, media
exhaustion, contact inhibition, etc.). No statistical adjustment fixes
this; only a vehicle arm would. The honest claim
[FINDINGS.md](FINDINGS.md) supports is "changed under 5-FU treatment over
time," not "induced by 5-FU specifically." The compositional-null control
rules out a *specific* artefact (mechanical rank inflation from the
cell-cycle collapse), not the vehicle-control confound in general.

## Arm B is n=3 per timepoint

Trend p-values and Cohen's d in `armB_induction_stats.csv` are
descriptive effect sizes with a clear direction, not confident inference.
A large Cohen's d computed on n=3 per group is still n=3 per group.

## MSI is a genuine, unresolved confound for the DTP association

DTP's association with 5-FU resistance in COREAD (r=+0.344, p=0.024)
weakens substantially after adjusting for microsatellite instability
status (r=+0.230, p=0.138, no longer significant at n=43,
`mutation_covariate_results.csv`), and the same happens in GDSC2
(r=+0.211, p=0.160). Adjusted for MSI and the other modules together,
GDSC1 keeps a signal (r=+0.318, p=0.038) but GDSC2 does not (r=+0.149,
p=0.325). MSI lines are both more 5-FU-sensitive
and lower in DTP score. This is the one deflationary explanation, of four
tested, that actually bites (lineage, TP53 and generic chemosensitivity
do not weaken the association). It is not resolved by this project's
data: a larger, MSI-stratified cohort would be needed.

The post hoc follow-ups (`msi_effect_modification.csv`) find no sign that
MSI modifies the association: within MSS lines DTP sits at r=+0.238
(n=27) and +0.242 (n=30), close to the MSI-adjusted values, and no
DTP × MSI interaction approaches significance (p≥0.13). That reads as
confounding rather than effect modification, but with 16 MSI lines per
screen the within-MSI estimates and the colorectal interaction tests can
only rule out large differences.

## The assay-ceiling artefact

Roughly 29% of solid lines sit at AUC>0.95: 5-FU barely affects them at
any screenable concentration, and the model has essentially nothing to
predict there (r=−0.091 among ceiling lines, vs r=0.467 among responsive
ones, `final_model_results.csv`). This is a property of the assay's
dynamic range, not of the biology; pooling ceiling and responsive lines
into one r understates the model's real ranking ability among lines that
are actually resistance-differentiated.

## CRC-specific claims are underpowered

n=43 (GDSC1) to 46 (GDSC2) colorectal lines is too small to support a
model trained on CRC alone: the four-module Ridge model scores r=−0.174
[−0.334, 0.019], and its inner CV picks the heaviest penalty on offer,
i.e. finds nothing to fit. Transfer from the pan-solid
model outperforms local CRC training, but "the pan-solid signal transfers
reasonably well to CRC" is a different and weaker claim than "this is a
validated colorectal-specific predictor."

## The compositional-score null is a scoring-artefact control, not a general validity check

`compositional_null` (Arm B) tests one specific failure mode: whether an
observed time-trend merely reflects rank-compositional inflation from
another feature (cell-cycle genes) collapsing. Clearing this bar means a
feature's trend is not an artefact of *that* mechanism; it does not mean
the trend reflects a causal or even a specific biological process; see
the vehicle-control limitation above for what it still cannot rule out.

## Fetal and MYC are reference-only, not independent validations

Fetal shares 128 of RSC's 232 genes (r=0.96 with RSC); MYC correlates
r=0.83 with CellCycle. Wherever Fetal or MYC numbers appear alongside RSC
or CellCycle numbers in this project's outputs, they should be read as
the *same* underlying signal scored twice under different names, not as
independent corroboration (`config.REFERENCE_MODULES`,
`config.OVERLAP_PAIRS`).

## Signature recovery is imperfect, and known to be

No signature in this project's 7-signature panel resolves at 100% against
either dataset's gene universe (see the per-signature recovery percentages
printed by `stages/01_build_tables.py` and `stages/04_armB_induction.py`).
Genes that fail to resolve, even after HGNC alias matching, are simply
absent from that signature's score for that dataset; this is a real,
quantified gap in coverage, not an approximation error to explain away.

## Affine recalibration is a display fix, not a modelling improvement

`calibration_results.csv`'s a/b transform cannot change r by construction:
for the algebraically optimal a and b, R² becomes exactly r². It also
requires labelled data from the target screen to fit, which a genuinely
new, unlabelled dataset would not have. Use it to display an unbiased
predicted AUC; do not read it as evidence the model improved.

## Confidence intervals are wide relative to reported differences

The 95% CIs on repeated-CV r resample cell lines and are roughly ±0.07
for the pan-solid models (see the CI columns throughout
`final_model_results.csv` and `multidrug_results.csv`), and far wider
for the CRC-only ones. Overlapping intervals are not a test, so the
model pairs that matter are compared directly in `model_comparisons.csv`
(corrected resampled t-test). Of those, the transcriptome beats both the
lineage + proliferation bar and the modules model on raw AUC; the
modules model does not beat the bar, and once lineage is removed the
transcriptome's lead over the modules is not established (p=0.34). Gaps
between other rows have not been tested and should be read as possibly
noise.

## The patient cohorts are prognostic only

Every patient in the five FOLFOX cohorts received FOLFOX, so a score that
associates with response means "did better on FOLFOX", not "benefits from
FOLFOX rather than something else". The one significant result (adjusted
DTP, GPL570, p=0.040) is a single test among several modules and two
cohorts, it does not replicate in the Agilent cohort, where DTP gene
recovery is low, and 10.8% of random gene sets of the same size do as
well in the same model. MSI in these cohorts is an expression-template proxy,
not a molecular test.

## Alpha is tuned per fit; l1_ratio and k are not

Alpha is now chosen by nested CV inside every outer training fold, for
every ElasticNet and Ridge model (docs/METHODS.md, "Nested tuning").
`l1_ratio=0.5` and `k=1500` are still fixed, as picked on GDSC1 5-FU
during development, and are reused for every drug, screen and feature
set.

Tuning was overdue because one fixed alpha behaves differently across
targets. ElasticNet's penalty is in target units and the raw-AUC target
is not standardised, so a screen with a narrower AUC range was shrunk
harder: GDSC2 5-FU has half GDSC1's AUC spread (SD 0.072 vs 0.149), and
its four-feature modules model scored r=0.076 at the fixed alpha. Tuned,
it scores 0.280, in line with GDSC1's 0.257 (`penalty_scale_sensitivity.csv`),
which resolves the discrepancy between screens.

On GDSC1 5-FU itself tuning changes little. The tuned transcriptome
model scores r=0.441 against 0.427 fixed (corrected t-test p=0.14), and
0.200 against 0.182 de-confounded (p=0.36; `model_comparisons.csv`).
Where the fixed alpha happened to suit a target, tuning does not beat
it; where it did not, tuning repairs it. Tuning can also lower an r:
the GDSC2 raw transcriptome model drops from 0.545 to 0.497, because a
3-fold inner CV on 80% of the lines does not recover the alpha that
happened to suit this target. The tuned figure is the honest estimate,
since the fixed alpha was never chosen blind to it.

The de-confounded GDSC1 model now picks alpha around 0.05, heavier than
0.02, and keeps fewer genes, so the model-gene list, its enrichment
overlap and the multidrug gene overlap are not comparable with releases
before nested tuning.

## The epigenetic correlate is unexplained

DTP methylation in N-shores and the open sea associates with response in
both screens, but those are not regulatory regions this data can
interpret, the effects are small (|r| ≤ 0.19), and the pooled promoter
score shows nothing. Treat it as a lead, not a mechanism.

## Median imputation runs before the split

`lib/io.py::load_expression` median-fills gaps on the whole matrix before
cross-validation. For expression this never fires: every gene with any
gap is already removed by the 10% missingness filter (0 gaps survive in
either screen). Raising `config.MAX_GENE_MISSING` would switch it on
silently.

## Exact reproduction needs the pinned environment

Expression is float32, and full-data ElasticNet fits on float32 input
shift by about 5×10⁻⁷ between library versions. `tests/test_golden.py`
compares at 1×10⁻⁹, so it passes only with `requirements.txt` as pinned.
Stage 10 also downloads the CollecTRI regulon at run time; if that
network changes upstream, the TF results change with it.
