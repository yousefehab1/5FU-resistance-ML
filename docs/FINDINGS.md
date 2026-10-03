# Findings

Every number below is read from a specific file in `data/processed/`,
written by a specific stage script and cited per row, so any number here can
be checked by opening that file after running the pipeline. See
[METHODS.md](METHODS.md) for how each was computed and
[LIMITATIONS.md](LIMITATIONS.md) for what it can't be stretched to claim.

## Model performance

| Result | r | 95% CI | Source |
|---|---|---|---|
| Lineage only, all lineages (B2) | 0.417 | n/a | `baseline_results.csv` |
| Lineage only, solid tumours (B2b) | 0.296 | n/a | `baseline_results.csv` |
| Transcriptome, raw AUC | 0.427 | [0.407, 0.455] | `final_model_results.csv` |
| Transcriptome, lineage de-confounded | 0.182 | [0.092, 0.250] | `final_model_results.csv` |
| Transcriptome, 5-FU-specific (general chemosensitivity removed) | 0.340 | [0.271, 0.389] | `multidrug_results.csv` |
| Transcriptome → GDSC2, external, uncalibrated | 0.508 | n/a | `calibration_results.csv` |
| Modules only (DTP+RSC+CBC+CellCycle) | 0.251 | [0.246, 0.261] | `final_model_results.csv` |
| Pan-solid model → CRC (transfer) | 0.271 | [0.012, 0.492] | `final_model_results.csv` |
| CRC-only model | 0.116 | [−0.047, 0.236] | `final_model_results.csv` |
| Permuted labels (B5, leakage tripwire) | −0.031 | n/a | `baseline_results.csv` |
| Out-of-fold, calibrated (dashboard display) | 0.457 | n/a | printed by `stages/12_dashboard_data.py` |

Assay-resolution split (`final_model_results.csv`): responsive lines
(AUC≤0.95) r=0.467; assay-ceiling lines (AUC>0.95, ~29% of the cohort)
r=−0.091; pooling these dilutes the signal in both directions.

Affine recalibration (`calibration_results.csv`): a=0.514, b=0.482, fit on
held-out half of GDSC2. A display fix; cannot change r by construction.

## The DTP hypothesis, every test run

| Finding | Effect | p | Source |
|---|---|---|---|
| DTP vs 5-FU resistance, GDSC1 COREAD | +0.344 | 0.024 | `within_crc_association.csv` (n=43) |
| DTP vs 5-FU resistance, GDSC2 COREAD | +0.294 | 0.048 | `within_crc_association.csv` (n=46, replicates) |
| DTP adjusted for MSI | +0.230 | 0.138 | `mutation_covariate_results.csv`: MSI is a confounder |
| DTP adjusted for TP53 | +0.338 | 0.027 | `mutation_covariate_results.csv`: TP53 is not |
| DTP vs generic drug sensitivity | +0.174 | 0.263 | `dtp_specificity_check.csv`: not generic fragility |
| DTP vs 5-FU-specific component | +0.302 | 0.049 | `dtp_specificity_check.csv`: specific to 5-FU |
| DTP induction over time-course | +0.822 | 0.001 | `armB_induction_stats.csv`: exceeds compositional null |
| DTP_up enrichment in model genes | 4.66× | 1.1×10⁻⁶ | `enrichment_results.csv`: unsupervised rediscovery |

Four deflationary explanations of the DTP association have been tested:
lineage (de-confounded CV above), MSI (weakens it; the only one that
bites), TP53 (does not), generic chemosensitivity (does not).

## Enrichment (`enrichment_results.csv`)

All 7 curated signatures tested for enrichment among the ~413 non-zero
genes of the final de-confounded model, against the full missingness-
filtered, haem-excluded gene universe (36,416 genes):

| Signature | Fold | p |
|---|---|---|
| DTP_up | 4.66× | 1.1×10⁻⁶ |
| RSC | 4.24× | 6.9×10⁻⁵ |
| Fetal | 4.08× | 2.3×10⁻⁵ |
| DTP_down | 2.08× | 0.031 |
| CBC | 1.94× | 0.056 |
| MYC | 0.37× | 0.94 (depleted, ns) |
| CellCycle | 0.54× | 0.89 (depleted, ns) |

An unsupervised ElasticNet, selecting from 36,416 genes with no knowledge
of the curated signatures, rediscovers the DTP programme.

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

`multidrug_model_results.csv`: the stage 02 pipeline, not retuned, run on
each drug in the screen stage 05's QC chose for it (`multidrug_qc.csv`):

| Drug | Screen | n | Raw r [95% CI] | De-confounded r [95% CI] |
|---|---|---|---|---|
| 5-FU | GDSC1 | 786 | 0.427 [0.407, 0.455] | 0.182 [0.092, 0.250] |
| Oxaliplatin | GDSC2 | 828 | 0.537 [0.523, 0.549] | 0.225 [0.212, 0.241] |
| SN-38 | GDSC2 | 819 | 0.254 [0.236, 0.271] | 0.123 [0.114, 0.134] |
| Irinotecan | GDSC2 | 826 | 0.631 [0.520, 0.670] | 0.327 [0.202, 0.399] |
| Cisplatin | GDSC1 | 772 | 0.374 [0.360, 0.386] | 0.222 [0.207, 0.241] |

- **Is DTP specific to 5-FU?** (`drug_specificity.csv`, COREAD lines in
  GDSC2, n=42 to 46.) After removing general chemosensitivity, DTP still
  tracks resistance to 5-FU (r=+0.340, p=0.021) and oxaliplatin
  (r=+0.330, p=0.025), but not SN-38 (p=0.22), irinotecan (p=0.40) or
  cisplatin (p=0.73).
- **Shared genes** (`drug_specificity_gene_overlap.csv`, hypergeometric
  against the shared gene universe). The 5-FU and oxaliplatin models share
  29 genes (6.6× expected, p=1.7×10⁻¹⁵) and 5-FU and irinotecan share 22
  (4.8×, p=1.8×10⁻⁹). 5-FU and cisplatin share 6, no more than chance
  (1.2×, p=0.35).
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
| GPL570 | 1.25 [0.93, 1.67], p=0.13 | 1.47 [1.02, 2.11], p=0.040 |
| GPL6480 | 0.99 [0.66, 1.49], p=0.98 (low gene recovery) | 1.38 [0.77, 2.44], p=0.28 |

No other module and no model-derived score (from the 413 genes) reaches
p<0.05 in either cohort. Every patient received FOLFOX, so this is a
prognostic association, not a predictive one (see
[LIMITATIONS.md](LIMITATIONS.md)).

## Epigenome

- **MLH1 against binary MSI** (`mlh1_msi_results.csv`, COREAD, n=42 and
  45). MLH1 promoter methylation separates MSI from MSS lines (r=0.47,
  p≤0.002 in both screens). Yet adjusting DTP's association for continuous
  MLH1 methylation leaves it intact (GDSC1 r=+0.348, p=0.024; GDSC2
  r=+0.300, p=0.045), while the binary MSI call weakens it (r=+0.244 and
  +0.228, p>0.11). Whatever MSI captures here is not MLH1 promoter
  methylation alone.
- **Methylation as features** (`methylation_model_results.csv`).
  Methylation-only models reach r=0.417 (GDSC1) and 0.460 (GDSC2), close
  to expression on the same lines (0.441 / 0.552). Late fusion adds at
  most 0.019.
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
  (distal fraction q=0.71 for DTP_up, 0.22 for DTP_down). RSC and Fetal
  genes are (q=0.020 and 0.024).
- **CpG context** (`methylation_context_results.csv`). Split by island
  relation, DTP gene methylation in N-shores (r=−0.161 / −0.128) and the
  open sea (r=+0.190 / +0.137) associates with response in both screens
  after BH correction. Islands, S-shores, S-shelves and the enhancer
  stratum do not; N-shelves reach significance in GDSC2 only.

So there is a replicated epigenetic correlate of response, but it sits
outside promoters and enhancers, and this data cannot say what it means
mechanistically.

## Summary

Four independent lines of evidence point the same way for DTP: enrichment
in an unsupervised model, association in colorectal lines, change under
treatment in HCT116, and an adjusted clinical signal. Each has a different
weakness, none is decisive alone, and the obvious confound (MSI) was
tested and does weaken the cell-line result.
