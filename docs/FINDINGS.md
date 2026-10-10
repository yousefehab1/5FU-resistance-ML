# Findings

Every number below is read from a specific file in `data/processed/`,
written by a specific stage script and cited per row, so any number here can
be checked by opening that file after running the pipeline. See
[METHODS.md](METHODS.md) for how each was computed and
[LIMITATIONS.md](LIMITATIONS.md) for what it can't be stretched to claim.

## Model performance

| Result | r | 95% CI | Source |
|---|---|---|---|
| Noise ceiling: GDSC1 vs GDSC2 AUC, same 781 solid lines | 0.518 | n/a | `screen_agreement.csv` |
| Lineage only, all lineages (B2) | 0.412 | n/a | `baseline_results.csv` |
| Lineage only, solid tumours (B2b) | 0.300 | n/a | `baseline_results.csv` |
| Transcriptome, raw AUC | 0.441 | [0.372, 0.506] | `final_model_results.csv` |
| Transcriptome, lineage de-confounded | 0.200 | [0.155, 0.274] | `final_model_results.csv` |
| Transcriptome, 5-FU-specific (general chemosensitivity removed) | 0.329 | [0.263, 0.403] | `multidrug_results.csv` |
| Transcriptome → GDSC2, external, uncalibrated | 0.535 | n/a | `calibration_results.csv` |
| Modules only (DTP+RSC+CBC+CellCycle) | 0.257 | [0.183, 0.329] | `final_model_results.csv` |
| Lineage + proliferation (B4, Ridge on the same splits) | 0.280 | [0.214, 0.343] | `final_model_results.csv` |
| Transcriptome, whole lineages held out | 0.302 | [0.236, 0.364] | `final_model_results.csv` |
| Modules, whole lineages held out | 0.185 | [0.116, 0.254] | `final_model_results.csv` |
| Pan-solid model → CRC (transfer) | 0.292 | [0.029, 0.520] | `final_model_results.csv` |
| CRC-only modules model | −0.174 | [−0.334, 0.019] | `final_model_results.csv` |
| Permuted labels (B5, leakage tripwire) | −0.042 | n/a | `baseline_results.csv` |
| Out-of-fold, calibrated (dashboard display) | 0.475 | n/a | printed by `stages/12_dashboard_data.py` |

The 95% CIs resample cell lines (2,000 draws) with the out-of-fold
predictions held fixed, so they say how much r depends on which lines
were screened. They replace a percentile over the 5 per-repeat r values,
which was far too narrow (the transcriptome's was [0.407, 0.455]). Rank
metrics tell the same story: Spearman rho 0.340 and C-index 0.616 for
the raw transcriptome model, 0.232 and 0.578 de-confounded.

Every ElasticNet and Ridge model picks its penalty by nested CV inside
each training fold (METHODS, "Nested tuning"). The fixed alpha=0.02 used
before is kept as a reference row; tuning does not beat it on GDSC1 5-FU
(table below). The CRC-only modules model is a Ridge fit on 43 lines
whose inner CV picks the heaviest penalties on the grid: there is nothing
in those lines for it to fit, and its negative r is noise around zero.

The ceiling is how well the two screens agree on the same lines, so it
bounds any model's r. It is computed by stage 01 on the modelling cohort
(solid, with expression, in both screens) and every `pct_of_ceiling` is
a fraction of it. It replaces a hardcoded 0.604 that included haem lines,
so every percentage of ceiling is now higher than before (the full
baseline model is at 87% rather than 75%).

**Which differences are real** (`model_comparisons.csv`, corrected
resampled t-test on the 25 shared splits, q is BH across the six):

| Comparison | Mean r gain | t | p | q |
|---|---|---|---|---|
| Transcriptome vs lineage + proliferation (B4) | +0.156 | 4.45 | 0.0002 | 0.0005 |
| Modules vs B4 | −0.023 | −0.83 | 0.41 | 0.41 |
| Transcriptome vs modules, raw AUC | +0.179 | 4.92 | 0.00005 | 0.0003 |
| Transcriptome vs modules, de-confounded | +0.083 | 1.49 | 0.15 | 0.22 |
| Transcriptome, tuned vs fixed alpha, raw AUC | +0.014 | 1.53 | 0.14 | 0.22 |
| Transcriptome, tuned vs fixed alpha, de-confounded | +0.026 | 0.93 | 0.36 | 0.41 |

The transcriptome beats the tissue-plus-proliferation bar and the four
modules; the modules model does not beat that bar. Once lineage is
removed, the transcriptome's lead over the modules is not established.
Nested tuning gains about 0.01 to 0.03 in r over the fixed alpha, within
split-to-split noise.

**Whole lineages held out** (`final_model_results.csv`). Fold-internal
de-confounding still trains on every tissue it tests on. Predicting each
of the solid lineages from a model fitted on all the others, the
transcriptome keeps r=0.302 pooled and a median within-lineage r of
+0.198 over the 20 lineages with enough lines; the modules model keeps
r=0.185 (median within-lineage +0.116). Both carry to tissues they have
never seen, the transcriptome more. At the fixed alpha the modules model
fell to r=0.084; tuning raised its pooled r but not its within-lineage
median, so what it gained is the offset between tissues, not ranking
within them.

Assay-resolution split (`final_model_results.csv`): responsive lines
(AUC≤0.95) r=0.481; assay-ceiling lines (AUC>0.95, ~29% of the cohort)
r=−0.094; pooling these dilutes the signal in both directions.

Affine recalibration (`calibration_results.csv`): a=0.412, b=0.567, fit on
held-out half of GDSC2. A display fix; cannot change r by construction.

## The DTP hypothesis, every test run

| Finding | Effect | p | q | Source |
|---|---|---|---|---|
| DTP vs 5-FU resistance, GDSC1 COREAD (confirmatory) | +0.344 [0.100, 0.561] | 0.024 | 0.139 | `within_crc_association.csv` (n=43) |
| DTP vs 5-FU resistance, GDSC2 COREAD (confirmatory) | +0.294 [0.037, 0.478] | 0.048 | 0.143 | `within_crc_association.csv` (n=46; same sign, CIs overlap) |
| DTP adjusted for MSI, GDSC1 | +0.230 | 0.138 | n/a | `mutation_covariate_results.csv`: MSI is a confounder |
| DTP adjusted for MSI, GDSC2 | +0.211 | 0.160 | n/a | `mutation_covariate_results.csv` |
| DTP adjusted for MSI + other modules, GDSC1 | +0.318 | 0.038 | n/a | `mutation_covariate_results.csv` |
| DTP adjusted for MSI + other modules, GDSC2 | +0.149 | 0.325 | n/a | `mutation_covariate_results.csv`: **does not replicate** |
| DTP adjusted for TP53, GDSC1 | +0.338 | 0.027 | n/a | `mutation_covariate_results.csv`: TP53 is not |
| DTP vs generic drug sensitivity | +0.174 | 0.263 | n/a | `dtp_specificity_check.csv`: not generic fragility |
| DTP vs 5-FU-specific component | +0.302 | 0.049 | n/a | `dtp_specificity_check.csv`: specific to 5-FU |
| DTP vs 5-FU, general chemosensitivity removed, GDSC2 (confirmatory) | +0.340 | 0.021 | 0.063 | `drug_specificity.csv` |
| DTP induction over time-course | +0.822 | 0.001 | n/a | `armB_induction_stats.csv`: exceeds compositional null |
| DTP_up enrichment in model genes | 3.29× | 0.034 | n/a | `enrichment_results.csv`: 4 genes; RSC as strong |

Brackets are bootstrap 95% CIs (500 resamples). q is Benjamini-Hochberg
within the test's family (see METHODS, "Multiplicity"); n/a marks a
single exploratory test with no family. No confirmatory test reaches
q<0.05 on its own. The cell-line case for DTP rests instead on
replication: the same sign and overlapping effect sizes in two
independent measurements of the same drug, with both CIs excluding zero.
The MSI adjustment weakens it in both screens, and with the other
modules also in the model the GDSC2 estimate (+0.149, p=0.325) is half
the GDSC1 one (+0.318, p=0.038): that adjusted result does not replicate.

Four deflationary explanations of the DTP association have been tested:
lineage (de-confounded CV above), MSI (weakens it; the only one that
bites), TP53 (does not), generic chemosensitivity (does not).

## MSI, mismatch repair and DTP sub-programmes (post hoc)

None of these tests is in `config.CONFIRMATORY`; every row of the three
tables is marked `declared = post hoc`.

**Does the association differ by MSI status?**
(`msi_effect_modification.csv`). No detectable difference.

| Test | GDSC1 | GDSC2 |
|---|---|---|
| DTP vs AUC, COREAD MSS | +0.238 [−0.088, 0.533], n=27 | +0.242 [−0.007, 0.452], n=30 |
| DTP vs AUC, COREAD MSI | +0.233 [−0.372, 0.646], n=16 | +0.025 [−0.276, 0.402], n=16 |
| MSS minus MSI r | +0.004, p=0.99 | +0.217, p=0.51 |
| DTP × MSI interaction, COREAD (AUC per SD) | +0.025, p=0.72 | −0.015, p=0.69 |
| DTP slope in MSS, all solid lines, lineage fixed effects | +0.022 [0.009, 0.035], p=0.001 | +0.008 [0.002, 0.014], p=0.010 |
| DTP × MSI interaction, all solid lines | +0.043, p=0.13 | +0.015, p=0.28 |
| Pan-solid model scored on COREAD MSS lines | +0.042 [−0.243, 0.346] | +0.329 [0.024, 0.586] |
| Pan-solid model scored on COREAD MSI lines | +0.471 [0.063, 0.767] | +0.426 [−0.070, 0.758] |

Within MSS the DTP association keeps most of its size in both screens,
though neither interval excludes zero at n=27 to 30, and across 777 and
816 solid lines the DTP slope among MSS lines is positive in both. MSI
therefore looks like a confounder that shifts DTP and AUC together, not a
modifier. The transcriptome model trained outside the colon does not
replicate on MSS lines: GDSC1's falls to +0.04 there while GDSC2's holds
at +0.33, so whether its colorectal accuracy comes from MSI lines is
unresolved.

**Is DTP tied to mismatch repair?** (`dtp_mmr_coupling.csv`). The
positive control works: the MMR expression score is lower in MSI lines
(r=−0.372 and −0.288). Only the within-MSS and MSI-adjusted rows can
separate a coupling from MSI, and they say:

- **MMR expression.** Within COREAD, DTP is unrelated to the MMR score
  (r=−0.04 and −0.03; within MSS −0.26 and −0.17, p≥0.19). Across solid
  lines it is lower where DTP is higher: r=−0.174 and −0.173 adjusted for
  lineage, −0.208 and −0.203 adjusted for MSI too (q<0.001 in both
  screens). That pan-cancer coupling is not an MSI effect, but it does not
  show within colorectal lines.
- **Mutational burden.** DTP falls with burden in COREAD (r=−0.431 and
  −0.406, p≤0.005), mostly through MSI: adjusted for it, r=−0.270 and
  −0.243 (p=0.08 and 0.10), and across solid lines the link vanishes once
  MSI is held (r=−0.043 and −0.038). Within MSS the bootstrap intervals
  exclude zero (r=−0.285 [−0.514, −0.121] and −0.254 [−0.418, −0.098])
  while the p-values do not (0.15 and 0.18). Three hypermutated MSS lines
  (burden 660 to 916 against an MSS median of 40) drive that mismatch, so
  those rows are not evidence either way.
- **DTP and AUC with burden held.** r=+0.248 and +0.201 in COREAD
  (p=0.11 and 0.18), like the MSI-adjusted values; +0.139 and +0.105
  across solid lines adjusted for lineage and MSI (p≤0.003).

**Does part of DTP_up carry the signal?** (`dtp_subprogramme_results.csv`).
No. Each half is scored as half minus DTP_down; p is against 1,000 random
splits of the same sizes.

| Split (canonical genes) | Readout | Half A | Half B | Whole DTP | p |
|---|---|---|---|---|---|
| distal-rich (71) vs promoter-dominant (134) | GDSC1 COREAD r | +0.307 | +0.384 | +0.344 | 0.063 |
| | GDSC2 COREAD r | +0.283 | +0.266 | +0.294 | 0.68 |
| | HCT116 trend r | +0.520 | +0.562 | +0.822 | 0.93 |
| | GPL570 adjusted z | +2.035 | +2.033 | +2.053 | 0.99 |
| shared with RSC/Fetal/CBC (18) vs DTP-only (262) | GDSC1 COREAD r | +0.336 | +0.340 | +0.344 | 0.92 |
| | GDSC2 COREAD r | +0.245 | +0.286 | +0.294 | 0.62 |
| | HCT116 trend r | +0.228 | +0.736 | +0.822 | 0.97 |
| | GPL570 adjusted z | +1.507 | +1.975 | +2.053 | 0.56 |

No difference between halves stands out from chance (smallest p 0.063,
all q≥0.50). Distal-rich and promoter-dominant genes carry the same
signal, and the 18 genes DTP shares with RSC, Fetal or CBC account for
neither the cell-line nor the clinical association. The HCT116 gap
between the shared and DTP-only halves looks large but is ordinary for an
18-gene half (p=0.97). The pools hold 205 and 280 canonical genes; half sizes
per dataset after resolution are in the CSV (in the time-course, for
example, 65 distal-rich and 116 promoter-dominant). The signal is spread
across DTP_up rather than concentrated in a chromatin class or a shared
stem module.

## Enrichment (`enrichment_results.csv`)

All 7 curated signatures tested for enrichment among the 156 non-zero
genes of the final de-confounded model, against the full missingness-
filtered, haem-excluded gene universe (36,416 genes). The overlap test
ignores direction, so two signed tests sit next to it (positive means
higher AUC, i.e. resistant; see METHODS):

| Signature | Fold | p | Model genes toward resistance | p_sign | Genome shift | p_perm |
|---|---|---|---|---|---|---|
| RSC | 4.08× | 0.017 | 4 of 4 | 0.13 | +0.017 | 0.10 |
| DTP_up | 3.29× | 0.034 | 4 of 4 | 0.13 | +0.029 | 0.001 |
| Fetal | 2.49× | 0.12 | 3 of 3 | 0.25 | +0.027 | 0.011 |
| CellCycle | 1.43× | 0.41 | 2 of 2 | 0.50 | −0.025 | 0.063 |
| CBC | 1.29× | 0.46 | 1 of 2 | 1.00 | +0.004 | 0.37 |
| MYC | 0.97× | 0.64 | 1 of 1 | 1.00 | −0.041 | 0.001 |
| DTP_down | 0.61× | 0.81 | 1 of 1 | 1.00 | −0.031 | 0.001 |

The overlap test is weaker than it was. With alpha fixed at 0.02 the
model kept 413 genes, 15 of them DTP_up (4.66×, p=1.1×10⁻⁶), and the
unsupervised rediscovery of the DTP programme was the headline. Nested
tuning picks a heavier penalty for this model (median alpha 0.05) and
keeps 156 genes. Four are DTP_up (3.29×, p=0.034), and RSC overlaps as
much (4 genes, 4.08×, p=0.017); neither would survive BH correction
across the seven sets (q=0.12 for both). The model still picks DTP_up
genes more often than chance, but no longer distinctively. All four
carry resistance-direction coefficients, which a coin would match one
time in eight (p=0.13), and an elastic net splits weight among
correlated genes, so single coefficients are a poor read of direction.
The genome-wide test does not depend on the model or its penalty and is
unchanged. Across the whole genome the direction is settled: DTP_up genes correlate with resistance more than genes of
the same expression level (shift +0.029) and DTP_down genes less (−0.031),
both p=0.001 against a within-lineage permutation of the target. The two
halves of the signature point in opposite directions, as the hypothesis
predicts. Fetal leans toward resistance too, and MYC toward sensitivity.
The genome test is competitive (each set against similar genes), so it
says DTP genes track resistance more than other genes do, not that the
set as a whole is associated with it.

## Arm B: induction over the HCT116 time-course

`armB_induction_stats.csv`, trend vs time (0/6/24/48h, n=3 each):

| Feature | trend r | p | 48h−0h |
|---|---|---|---|
| Fetal | +0.961 | <0.001 | +2.12 |
| RSC | +0.959 | <0.001 | +1.94 |
| CBC | +0.954 | <0.001 | +0.81 |
| DTP | +0.822 | 0.001 | +0.73 |
| CellCycle | −0.884 | <0.001 | −1.02 |
| MYC | −0.962 | <0.001 | −0.70 |

Compositional control (`armB_compositional_control.csv`): all six trends
exceed a random-gene-set-of-the-same-size null at z≈2.0–2.5, so not an
artefact of rank compositionality from the cell-cycle collapse alone.
Confirmed directly: DTP's trend survives adjusting for CellCycle
(`armB_cellcycle_adjusted.csv`, r=+0.913, p<0.001).

Alias-fix recovery jump: DTP_up signature recovery in the time-course
went from 67.4% (pre-fix, hardcoded HGNC vintage mismatch) to 86.7%
(post-fix, printed by `stages/04_armB_induction.py`), close to GDSC's own
91.1% recovery, confirming the alias resolution genuinely fixes the
cross-arm symbol-vintage mismatch rather than papering over it.

## Multidrug / mechanism

`drug_similarity.csv`, `gene_target_check.csv` (784–786 solid lines with
expression and multi-drug data):

- 5-FU AUC vs general chemosensitivity: r=0.483 (23% of 5-FU AUC variance
  is the generic axis; 77% is 5-FU-specific).
- Across 352 other GDSC1 drugs, 5-FU most resembles **CX-5461**
  (r=+0.722), an RNA Pol I / ribosome-biogenesis inhibitor, not a
  DNA-damage or antimetabolite agent, which is not the mechanism the
  textbook description of 5-FU would predict.
- **TYMS**, 5-FU's own canonical DNA-directed target (thymidylate
  synthase), shows essentially no association with response (r=+0.024,
  p=0.51, n=786), a clean negative result. The rest of the pyrimidine
  pathway (DPYD, UPP1, UCK2, TK1, ABCB6) is similarly weak; ABCB6 (an
  efflux transporter, also a DTP_up gene) is the strongest single
  pharmacogene at r=+0.190, still weaker than several transcriptome
  features.

Taken together, response variation in this panel looks more RNA/ribosome-
biogenesis-driven than TS-driven, which the project did not set out to
look for.

## Driver mutations (`driver_mutation_correlations.csv`)

Only TP53 shows a robust, reproducible association with resistance across
both screens in the full solid cohort (GDSC1 r=+0.137, p=4.0×10⁻⁵;
GDSC2 r=+0.233, p=4.1×10⁻¹³); it does not reach significance within
COREAD alone at n=43–46. KRAS, BRAF, PIK3CA, APC, SMAD4, PTEN and NRAS
show no consistent signal in either cohort.

## Other drugs

`multidrug_model_results.csv`: the stage 02 pipeline, alpha tuned per fold, run on
each drug in the screen stage 05's QC chose for it (`multidrug_qc.csv`):

| Drug | Screen | n | Raw r [95% CI] | De-confounded r [95% CI] |
|---|---|---|---|---|
| 5-FU | GDSC1 | 786 | 0.441 [0.372, 0.506] | 0.200 [0.155, 0.274] |
| Oxaliplatin | GDSC2 | 828 | 0.525 [0.453, 0.589] | 0.276 [0.231, 0.329] |
| SN-38 | GDSC2 | 819 | 0.254 [0.202, 0.311] | 0.123 [0.074, 0.176] |
| Irinotecan | GDSC2 | 826 | 0.595 [0.511, 0.698] | 0.402 [0.332, 0.491] |
| Cisplatin | GDSC1 | 772 | 0.374 [0.308, 0.483] | 0.222 [0.171, 0.273] |

- **Is DTP specific to 5-FU?** (`drug_specificity.csv`, COREAD lines in
  GDSC2, n=42 to 46.) After removing general chemosensitivity, DTP still
  tracks resistance to 5-FU (r=+0.340, p=0.021) and oxaliplatin
  (r=+0.330, p=0.025), but not SN-38 (p=0.22), irinotecan (p=0.40) or
  cisplatin (p=0.73).
- **Shared genes** (`drug_specificity_gene_overlap.csv`, hypergeometric
  against the shared gene universe). With alpha tuned the models keep 124
  to 293 genes, so overlaps are counted in single digits. The 5-FU and
  oxaliplatin models share 3 genes (5.6× expected, p=0.017) and 5-FU and
  irinotecan share 4 (7.5×, p=0.002). 5-FU and cisplatin share 1, no more
  than chance (1.4×, p=0.50). The overlap is not special to 5-FU:
  oxaliplatin shares 5 genes each with irinotecan (11.7×) and cisplatin
  (8.9×).
- **Ribosome-biogenesis prediction.** With general chemosensitivity
  removed, 5-FU tracks oxaliplatin (r=+0.342) slightly more than cisplatin
  (r=+0.290), the direction the ribosome-biogenesis account predicts, but
  the bootstrap 95% CI on the difference, [−0.057, +0.165], includes zero.

## Patients treated with FOLFOX

`clinical_validation.csv`: five GEO cohorts, logistic regression of
response on each score, odds ratio per SD. Four Affymetrix studies are
pooled after ComBat (GPL570, n=189); the Agilent study (GSE104645,
GPL6480, n=97, FOLFOX arms only) is kept separate.

| Cohort | DTP, unadjusted OR | DTP, adjusted OR (purity, CMS, MSI, study) |
|---|---|---|
| GPL570 | 1.25 [0.93, 1.67], p=0.13 | 1.47 [1.02, 2.11], p=0.040, q=0.28 (confirmatory) |
| GPL6480 | 0.99 [0.66, 1.49], p=0.98 (low gene recovery) | 1.38 [0.77, 2.44], p=0.28 |

q is BH across the seven features within each cohort and model. The one
nominal hit does not survive that correction (q=0.28).

Random gene sets often associate with outcome in expression data
(Venet et al. 2011), so each adjusted test was also run on 1,000 random
gene sets of the same size, scored and fitted the same way
(`p_random_null`). The adjusted GPL570 DTP result is beaten by 10.8% of
them (p_random_null=0.108): its z of +2.05 is not unusual for an
arbitrary gene set in this cohort.

No other module and no model-derived score (from the 156 model genes)
reaches p<0.05 in either cohort; the closest is the adjusted GPL570
model-derived score (OR 0.70 per SD, p=0.051, in the sensitive
direction). Every patient received FOLFOX, so this is a
prognostic association, not a predictive one (see
[LIMITATIONS.md](LIMITATIONS.md)).

## Epigenome

- **MLH1 against binary MSI** (`mlh1_msi_results.csv`, COREAD, n=42 and
  45). MLH1 promoter methylation separates MSI from MSS lines (r=0.47,
  p≤0.002 in both screens). Yet adjusting DTP's association for continuous
  MLH1 methylation leaves it intact (GDSC1 r=+0.348, p=0.024; GDSC2
  r=+0.300, p=0.045), while the binary MSI call weakens it (r=+0.244 and
  +0.228, p>0.11). Whatever MSI captures here is not MLH1 promoter
  methylation alone. These binary-MSI values differ from the table above
  (+0.230 and +0.211) only because one line, SIDM00841, has no
  methylation data: on all 43 and 46 MSI-called lines this stage
  reproduces stage 03 exactly (`r_adj_binary_MSI_all_msi_called`).
- **Methylation as features** (`methylation_model_results.csv`).
  Methylation-only models reach r=0.420 (GDSC1) and 0.530 (GDSC2), close
  to expression on the same lines (0.451 / 0.490). Late fusion adds at
  most 0.026 over the better of the two.
- **DTP gene promoters** (`reports/methylation_models.md`). Methylation
  of DTP gene promoters tracks the expression DTP score in the silencing
  direction (r=−0.26 in both screens), but a methylation-derived DTP score
  does not predict response (p=0.25 and 0.26).
- **Transcription-factor activity** (`tf_activity_results.csv`,
  `tf_dtp_associations.csv`). Within COREAD, TEAD1, TP53 and SOX9
  activity track the DTP score (r=+0.45 to +0.76) but not AUC. Seven
  exploratory TFs pass q<0.10 against AUC in GDSC1 and none in GDSC2, so
  no TF replicates across screens.
- **Colon tumour chromatin** (`regulatory_permutation_results.csv`, TCGA
  ATAC-seq). DTP genes are not enhancer-rich against a matched background
  (distal fraction q=0.71 for DTP_up, 0.23 for DTP_down). RSC and Fetal
  genes are (q=0.020 and 0.024).
- **CpG context** (`methylation_context_results.csv`). Split by island
  relation, DTP gene methylation in N-shores (r=−0.161 / −0.128) and the
  open sea (r=+0.190 / +0.137) associates with response in both screens
  after BH correction. Islands, S-shores, S-shelves and the enhancer
  stratum do not; N-shelves reach significance in GDSC2 only.

So there is a replicated epigenetic correlate of response, but it sits
outside promoters and enhancers, and this data cannot say what it means
mechanistically.

## Modules model across screens (`module_screen_discrepancy.csv`)

This section records why the screens disagreed before nested tuning,
when every model used alpha=0.02. With alpha tuned per fold the modules
model reaches r=0.257 on GDSC1 and 0.280 on GDSC2, and the discrepancy
is gone (`module_screen_discrepancy.csv` now holds the tuned values; the
fixed-alpha table below is from commit 3627c43, and its all-solid rows
survive as the `fixed alpha` rows of `penalty_scale_sensitivity.csv`).

At the fixed alpha the modules model (DTP, RSC, CBC, CellCycle) reached
r=0.251 on GDSC1 but 0.076 on GDSC2. Changing one thing at a time:

| GDSC2 refit on | r | AUC sd |
|---|---|---|
| all solid lines | 0.076 | 0.072 |
| responsive lines only (AUC≤0.95) | 0.107 | 0.071 |
| the 781 lines also in GDSC1 | 0.074 | 0.072 |
| all solid lines, AUC scaled to unit SD | **0.270** | 0.072 |
| (GDSC1 model applied to GDSC2, no refit) | 0.198 | 0.072 |

Neither the larger ceiling fraction (35% vs 29%) nor line composition
explains the gap. GDSC2's AUC has half GDSC1's spread (SD 0.072 vs
0.149), and ElasticNet's penalty acts on the target's scale, so the
fixed `alpha=0.02` shrinks the GDSC2 refit to almost nothing (the DTP
coefficient is exactly zero). Scaled to unit SD, GDSC2 gives r=0.270,
the same as GDSC1. The lineage de-confounded models z-score the target,
which is why GDSC2's de-confounded modules model (0.217 at the fixed
alpha) beat its raw one.

This was not a target-scale bug with a target-scale fix. Refitting the
5-FU models under three scaling rules (`penalty_scale_sensitivity.csv`)
showed the transcriptome models want more shrinkage than the modules
models, so any single rescaling helped some models and hurt others. The
underlying problem was one untuned alpha shared by models of very
different size, and the fourth rule in that table, nested tuning, is the
fix the pipeline now uses. See LIMITATIONS.

## Summary

Four independent lines of evidence point the same way for DTP: enrichment
in an unsupervised model (weaker since nested tuning: 4 genes, p=0.034,
no more than RSC), association in colorectal lines, change under
treatment in HCT116, and an adjusted clinical signal (p=0.040, which does
not survive BH correction, q=0.28, and which 10.8% of random gene sets
of the same size match). Each has a different weakness, none
is decisive alone, and the obvious confound (MSI) was tested and does
weaken the cell-line result, with the fully adjusted estimate failing to
replicate in GDSC2. Post hoc, MSI acts as a confounder rather than a
modifier (the association holds its size within MSS lines, with no
interaction), and no chromatin or shared-module subset of DTP_up carries
the signal on its own.
