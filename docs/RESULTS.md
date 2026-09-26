# Results

Every number below is in a file under `data/processed/` (frozen in `tests/golden/`). r is Pearson correlation between predicted and measured AUC unless stated otherwise; higher AUC means more resistant. The ceiling is r = 0.604, the agreement between GDSC1 and GDSC2 on the same lines.

## 1. Predicting 5-FU response (Arm A)

`baseline_results.csv`, `refined_model_results.csv`, `final_model_results.csv`, `multidrug_results.csv`

| Model | r | 95% CI | Note |
|---|---|---|---|
| Lineage only, all tissues | 0.417 | | 69% of ceiling: the bar to beat |
| Lineage only, solid tumours | 0.296 | | bar after removing blood cancers |
| Proliferation only | -0.017 | | |
| **Transcriptome, raw AUC** | **0.427** | 0.407 to 0.455 | 5x5 repeated CV, solid tumours |
| responsive lines (AUC ≤ 0.95, n = 557) | 0.467 | | |
| lines at assay ceiling (AUC > 0.95, n = 229) | -0.091 | | unresolvable by the assay |
| Transcriptome → GDSC2 | 0.470 | | a re-measurement, not new lines |
| **Lineage de-confounded** | **0.182** | 0.092 to 0.250 | signal within tissue |
| **5-FU-specific** (general chemosensitivity removed) | **0.340** | 0.271 to 0.389 | |
| Modules only | 0.251 | 0.246 to 0.261 | |
| Pan-solid → colorectal transfer | 0.271 | 0.012 to 0.492 | |
| Colorectal-only modules | 0.116 | -0.047 to 0.236 | n = 43 is too small |
| Permuted labels | -0.011 | | leakage check, passes |

Expression carries real, tissue-independent and drug-specific signal. It orders the lines the assay can resolve and says nothing about the 29% at the ceiling.

## 2. The model rediscovers the DTP gene space

`multidrug_enrichment.csv`. The de-confounded model selects 413 of 36,416 genes without seeing any signature.

| Signature | Selected | Expected | Fold | p |
|---|---|---|---|---|
| **DTP_up** | 15 | 3.2 | **4.7** | **1.1e-6** |
| RSC | 11 | 2.6 | 4.2 | 6.9e-5 |
| Fetal | 13 | 3.2 | 4.1 | 2.3e-5 |
| DTP_down | 9 | 4.3 | 2.1 | 0.03 |
| CellCycle | 2 | 3.7 | 0.5 | 0.89 |
| MYC | 1 | 2.7 | 0.4 | 0.94 |

## 3. DTP and resistance in colorectal lines

`dtp_variant_comparison.csv`, `mlh1_msi_results.csv`, `drug_specificity.csv`

| Test | r | p |
|---|---|---|
| DTP vs AUC, GDSC1 colorectal (n = 43) | +0.344 | 0.024 |
| DTP vs AUC, GDSC2 colorectal (n = 46) | +0.294 | 0.048 |
| adjusted for binary MSI, GDSC1 / GDSC2 | +0.244 / +0.228 | 0.12 / 0.13 |
| adjusted for MLH1 promoter methylation, GDSC1 / GDSC2 | +0.348 / +0.300 | 0.02 / 0.05 |
| adjusted for general chemosensitivity (GDSC2) | +0.340 | 0.02 |
| DTP up-only instead of bidirectional, GDSC1 | +0.279 | 0.07 |

DTP is not explained by lineage, TP53 or general drug sensitivity. **MSI status does weaken it** below significance at this sample size. Continuous MLH1 methylation does not separate the two any better than the binary flag.

## 4. Induction under 5-FU (Arm B)

`armB_induction_stats.csv`, `armB_compositional_control.csv`. HCT116, 0/6/24/48 h, triplicate.

| Programme | Trend r | p | z vs random gene sets |
|---|---|---|---|
| Fetal | +0.96 | 7e-7 | 2.5 |
| RSC | +0.96 | 8e-7 | 2.2 |
| CBC | +0.95 | 1e-6 | 2.2 |
| **DTP** | **+0.82** | **0.001** | **2.0** (empirical p = 0.015) |
| CellCycle | -0.88 | 1e-4 | -2.1 |
| MYC | -0.96 | 6e-7 | -2.3 |

Regenerative and DTP programmes rise as proliferation programmes fall, beyond what the collapse of cell-cycle genes alone would produce in rank space.

## 5. Other drugs

`multidrug_model_results.csv`, `drug_specificity.csv`, `drug_specificity_gene_overlap.csv`

| Drug | Screen | Raw r | De-confounded r |
|---|---|---|---|
| 5-FU | GDSC1 | 0.427 | 0.182 |
| Oxaliplatin | GDSC2 | 0.537 | 0.225 |
| SN-38 | GDSC2 | 0.254 | 0.123 |
| Irinotecan | GDSC2 | 0.631 | 0.327 |
| Cisplatin | GDSC1 | 0.374 | 0.222 |

- DTP survives general-chemosensitivity adjustment for 5-FU and oxaliplatin only, not for SN-38, irinotecan or cisplatin.
- 5-FU and oxaliplatin models share 29 genes (6.6x expected, p = 2e-15). 5-FU and cisplatin share no more than chance.
- Across 264 drugs, 5-FU response most resembles CX-5461, an RNA Pol I inhibitor (r = 0.72). TYMS expression shows no association (r = 0.02). Variation in 5-FU sensitivity here looks driven by the RNA / ribosome-biogenesis arm rather than by thymidylate synthase.
- After adjustment, 5-FU tracks oxaliplatin (r = 0.34) slightly more than cisplatin (r = 0.29), in the direction the ribosome-biogenesis hypothesis predicts, but the difference CI includes zero (-0.06 to 0.17).

## 6. Patients treated with FOLFOX

`clinical_validation.csv`. Five GEO cohorts: four Affymetrix (n = 189) and one Agilent (n = 97). Logistic regression of response on each score, per SD.

| Cohort | DTP, unadjusted OR | DTP, adjusted OR (purity, CMS, MSI, study) |
|---|---|---|
| Affymetrix | 1.25 (0.93 to 1.67), p = 0.13 | 1.47 (1.02 to 2.11), p = 0.04 |
| Agilent | 0.99, p = 0.98 (low gene recovery) | 1.38, p = 0.28 |

No other module or the 413-gene model score reached significance. Every patient received FOLFOX, so this is prognostic, not predictive.

## 7. Epigenome

`methylation_model_results.csv`, `tf_activity_results.csv`, `regulatory_permutation_results.csv`, `methylation_context_results.csv`

- **Methylation as features.** Methylation-only models reach r = 0.42 (GDSC1) and 0.46 (GDSC2), close to expression on the same lines (0.44 / 0.55). Late fusion adds under 0.02.
- **DTP gene promoters.** Their methylation tracks the expression DTP score in the silencing direction (r = -0.26, both screens), but a methylation-derived DTP score does not predict response (p > 0.24).
- **TF activity.** No transcription factor associates with response in both screens. TEAD1, TP53 and SOX9 activity track the DTP score (r = 0.45 to 0.76) but not AUC.
- **Colon tumour chromatin.** DTP genes are not enhancer-rich against a matched background. RSC and Fetal genes are (q < 0.03).
- **CpG context.** Split by island relation, methylation of DTP genes in N-shores (r = -0.16 / -0.13) and open sea (r = +0.19 / +0.14) associates with response in both screens after correction. The enhancer stratum does not.

So there is a replicated epigenetic correlate of response, but it sits outside promoters and enhancers, and this data cannot say what it means mechanistically.

## Summary

Four independent lines of evidence point the same way for DTP: enrichment in an unsupervised model, association in colorectal lines, induction under treatment, and an adjusted clinical signal. Each has a different weakness, none is decisive alone, and the obvious confound (MSI) was tested and does weaken the cell-line result.
