# Findings

Every number below is read from a specific file in `data/processed/`,
written by a specific stage script — cited per row, so any number here can
be checked by opening that file after running the pipeline. See
[METHODS.md](METHODS.md) for how each was computed and
[LIMITATIONS.md](LIMITATIONS.md) for what it can't be stretched to claim.

## Model performance

| Result | r | 95% CI | Source |
|---|---|---|---|
| Lineage only, all lineages (B2) | 0.417 | — | `baseline_results.csv` |
| Lineage only, solid tumours (B2b) | 0.296 | — | `baseline_results.csv` |
| Transcriptome, raw AUC | 0.427 | [0.407, 0.455] | `final_model_results.csv` |
| Transcriptome, lineage de-confounded | 0.182 | [0.092, 0.250] | `final_model_results.csv` |
| Transcriptome, 5-FU-specific (general chemosensitivity removed) | 0.340 | [0.271, 0.389] | `multidrug_results.csv` |
| Transcriptome → GDSC2, external, uncalibrated | 0.508 | — | `calibration_results.csv` |
| Modules only (DTP+RSC+CBC+CellCycle) | 0.251 | [0.246, 0.261] | `final_model_results.csv` |
| Pan-solid model → CRC (transfer) | 0.271 | [0.012, 0.492] | `final_model_results.csv` |
| CRC-only model | 0.116 | [−0.047, 0.236] | `final_model_results.csv` |
| Permuted labels (B5, leakage tripwire) | −0.031 | — | `baseline_results.csv` |
| Out-of-fold, calibrated (dashboard display) | 0.457 | — | printed by `stages/10_build_dashboard_data.py` |

Assay-resolution split (`final_model_results.csv`): responsive lines
(AUC≤0.95) r=0.467; assay-ceiling lines (AUC>0.95, ~29% of the cohort)
r=−0.091 — pooling these dilutes the signal in both directions.

Affine recalibration (`calibration_results.csv`): a=0.514, b=0.482, fit on
held-out half of GDSC2. A display fix; cannot change r by construction.

## The DTP hypothesis, every test run

| Finding | Effect | p | Source |
|---|---|---|---|
| DTP vs 5-FU resistance, GDSC1 COREAD | +0.344 | 0.024 | `within_crc_association.csv` (n=43) |
| DTP vs 5-FU resistance, GDSC2 COREAD | +0.294 | 0.048 | `within_crc_association.csv` (n=46, replicates) |
| DTP adjusted for MSI | +0.230 | 0.138 | `mutation_covariate_results.csv` — MSI is a confounder |
| DTP adjusted for TP53 | +0.338 | 0.027 | `mutation_covariate_results.csv` — TP53 is not |
| DTP vs generic drug sensitivity | +0.174 | 0.263 | `dtp_specificity_check.csv` — not generic fragility |
| DTP vs 5-FU-specific component | +0.302 | 0.049 | `dtp_specificity_check.csv` — specific to 5-FU |
| DTP induction over time-course | +0.822 | 0.001 | `armB_induction_stats.csv` — exceeds compositional null |
| DTP_up enrichment in model genes | 4.66× | 1.1×10⁻⁶ | `enrichment_results.csv` — unsupervised rediscovery |

Four deflationary explanations of the DTP association have been tested:
lineage (de-confounded CV above), MSI (weakens it — the only one that
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

`armB_induction_stats.csv` — trend vs time (0/6/24/48h, n=3 each):

| Feature | trend r | p | 48h−0h |
|---|---|---|---|
| Fetal | +0.961 | <0.001 | +2.12 |
| RSC | +0.959 | <0.001 | +1.94 |
| CBC | +0.954 | <0.001 | +0.81 |
| DTP | +0.822 | 0.001 | +0.73 |
| CellCycle | −0.884 | <0.001 | −1.02 |
| MYC | −0.962 | <0.001 | −0.70 |

Compositional control (`armB_compositional_control.csv`): all six trends
exceed a random-gene-set-of-the-same-size null at z≈2.0–2.5 — not an
artefact of rank compositionality from the cell-cycle collapse alone.
Confirmed directly: DTP's trend survives adjusting for CellCycle
(`armB_cellcycle_adjusted.csv`, r=+0.913, p<0.001).

Alias-fix recovery jump: DTP_up signature recovery in the time-course
went from 67.4% (pre-fix, hardcoded HGNC vintage mismatch) to 86.7%
(post-fix, printed by `stages/08_armB_induction.py`), close to GDSC's own
91.1% recovery — confirming the alias resolution genuinely fixes the
cross-arm symbol-vintage mismatch rather than papering over it.

## Multidrug / mechanism

`drug_similarity.csv`, `gene_target_check.csv` (784–786 solid lines with
expression and multi-drug data):

- 5-FU AUC vs general chemosensitivity: r=0.483 (23% of 5-FU AUC variance
  is the generic axis; 77% is 5-FU-specific).
- Across 352 other GDSC1 drugs, 5-FU most resembles **CX-5461**
  (r=+0.722), an RNA Pol I / ribosome-biogenesis inhibitor — not a
  DNA-damage or antimetabolite agent, which is not the mechanism the
  textbook description of 5-FU would predict.
- **TYMS**, 5-FU's own canonical DNA-directed target (thymidylate
  synthase), shows essentially no association with response (r=+0.024,
  p=0.51, n=786) — a clean negative result. The rest of the pyrimidine
  pathway (DPYD, UPP1, UCK2, TK1, ABCB6) is similarly weak; ABCB6 (an
  efflux transporter, also a DTP_up gene) is the strongest single
  pharmacogene at r=+0.190, still weaker than several transcriptome
  features.

Taken together, response variation in this panel looks more RNA/ribosome-
biogenesis-driven than TS-driven — an unexpected finding the original
project did not set out to look for.

## Driver mutations (`driver_mutation_correlations.csv`)

Only TP53 shows a robust, reproducible association with resistance across
both screens in the full solid cohort (GDSC1 r=+0.137, p=4.0×10⁻⁵;
GDSC2 r=+0.233, p=4.1×10⁻¹³); it does not reach significance within
COREAD alone at n=43–46. KRAS, BRAF, PIK3CA, APC, SMAD4, PTEN and NRAS
show no consistent signal in either cohort.
