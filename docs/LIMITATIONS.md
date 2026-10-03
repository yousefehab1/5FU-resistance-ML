# Limitations

What the numbers in [FINDINGS.md](FINDINGS.md) do not show, collected in
one place rather than scattered as footnotes. None of these are fixed by
more careful analysis of the existing data; they are limits of the data
itself.

## GDSC2 is a re-measurement, not an independent holdout

GDSC2 largely re-screens the **same cell lines** as GDSC1, not new ones.
Agreement between the two screens (r=0.604 for AUC, `config.CEILING_R`)
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
`mutation_covariate_results.csv`). MSI lines are both more 5-FU-sensitive
and lower in DTP score. This is the one deflationary explanation, of four
tested, that actually bites (lineage, TP53 and generic chemosensitivity
do not weaken the association). It is not resolved by this project's
data: a larger, MSI-stratified cohort would be needed.

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
transcriptome-wide model trained on CRC alone (r=0.116,
[−0.047, 0.236], a CI that includes zero). Transfer from the pan-solid
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

## Bootstrap/CV confidence intervals are wide relative to reported differences

Most 95% CIs on repeated-CV r in this project are roughly ±0.05–0.07
(see the CI columns throughout `final_model_results.csv` and
`multidrug_results.csv`). Several comparisons made across this project's
results table (e.g. "modules only" vs "transcriptome, de-confounded")
have overlapping CIs and should not be read as if the point-estimate gap
were established rather than plausibly noise.

## The patient cohorts are prognostic only

Every patient in the five FOLFOX cohorts received FOLFOX, so a score that
associates with response means "did better on FOLFOX", not "benefits from
FOLFOX rather than something else". The one significant result (adjusted
DTP, GPL570, p=0.040) is a single test among several modules and two
cohorts, and it does not replicate in the Agilent cohort, where DTP gene
recovery is low. MSI in these cohorts is an expression-template proxy,
not a molecular test.

## Hyperparameters were chosen once, on GDSC1 5-FU

`alpha=0.02, l1_ratio=0.5, k=1500` were picked by nested CV on GDSC1 5-FU
during development (the search is not part of the pipeline) and are
reused unchanged for the other four drugs, for GDSC2 and for the
methylation and TF-activity models. That keeps the comparisons like for
like, but the other models are not tuned and may be understated.

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
