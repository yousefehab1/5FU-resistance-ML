# Methods

How the pipeline works as it stands now. For the numbers see [RESULTS.md](RESULTS.md); for caveats see [LIMITATIONS.md](LIMITATIONS.md).

## Data

All files go in `data/raw/` and are never modified.

| File(s) | Source |
|---|---|
| `GDSC1_fitted_dose_response_24Jul22.csv`, `GDSC2_fitted_dose_response_24Jul22.csv`, `screened_compounds_rel_8.4.csv`, `Cell_Lines_Details.xlsx` | GDSC release 8.4, https://ftp.sanger.ac.uk/pub/project/cancerrxgene/releases/current_release/ |
| `rnaseq_all_20260323.csv`, `model_list_20260724.csv`, `mutations_summary_20260724.csv` | Cell Model Passports, https://cellmodelpassports.sanger.ac.uk/downloads |
| `methylation/GSE68379_Matrix.processed.txt.gz` | GEO GSE68379 (GDSC 450K methylation) |
| `methylation/humanmethylation450_manifest.csv` | Exported by `scripts/export_450k_manifest.R` |
| `geo_clinical/` | GEO GSE28702, GSE19860, GSE69657, GSE72970, GSE104645 (fetched by `scripts/17a_clinical_covariates.R`) |
| `atac/` | TCGA ATAC-seq (Corces et al. 2018), https://gdc.cancer.gov/about-data/publications/ATACseq-AWG; `gene_tss_hg38.csv` from `scripts/export_gene_tss_hg38.R` |
| `hgnc_alias_map.csv` | Exported by `scripts/export_hgnc_aliases.R` (org.Hs.eg.db) |
| `signatures/` (DTP_UP, DTP_Down, CBC, CellCycle), `sigs.csv` | Curated gene sets from the MSc project |
| `Sup_Table_2_HCT116_5FU_timecourse_treatment.txt` | HCT116 5-FU time course, raw counts, 0/6/24/48 h in triplicate |

## Pipeline

Run order is the `Makefile`. Every script writes to `data/processed/`.

| Script | What it does |
|---|---|
| **Targets and features** | |
| `01_explore_drug_response` | Extracts 5-FU from both screens; measures the GDSC1 vs GDSC2 agreement that sets the ceiling (r ≈ 0.60). |
| `02_build_modelling_table` | Streams the 5.7 GB expression file into `X_GDSC{1,2}.parquet` and joins targets into `y_GDSC{1,2}.parquet`. |
| `03_baselines` | Scores the signatures and fits the five baselines: mean, lineage, proliferation, lineage + proliferation, permuted labels. |
| **5-FU model** | |
| `06_model` | ElasticNet, Ridge and random forest on the transcriptome; nested CV picks alpha = 0.02, l1_ratio = 0.5. |
| `07_refined_model` | Restricts to solid tumours, adds MSI/TP53 covariates, compares DTP variants (up, down, bidirectional). |
| `08_final_model` | Repeated 5x5 CV with CIs, lineage de-confounding, colorectal transfer, assay-resolution strata; saves the 413-gene model. |
| `09_calibration_and_mutations` | Affine recalibration for the dashboard display; mutation features. Prints only. |
| `10_multidrug` | Splits 5-FU response into general chemosensitivity and a 5-FU-specific residual. |
| **Other drugs** | |
| `12_multidrug_targets` | Targets and QC for oxaliplatin, SN-38, irinotecan, cisplatin. |
| `13_multidrug_models` | The `08` pipeline per drug, with signature enrichment of each model's genes. |
| `14_drug_specificity` | Is DTP specific to 5-FU? Cross-drug gene overlap; ribosome-biogenesis test. |
| **Arm B and external cohorts** | |
| `05_armB_induction` | Scores the time course; trend tests plus a random gene-set null for compositional shifts. |
| `16_score_external` | Scores any external expression matrix with the same signatures (`fivefu.cohorts.score_cohort`). |
| `17a_clinical_covariates.R`, `17_clinical_validation` | Five FOLFOX cohorts; logistic regression of response on each score, unadjusted and adjusted for purity, CMS, MSI and study. |
| **Epigenome** | |
| `18a_methylation_prep.R`, `18_methylation_ingest` | 450K beta/M values aggregated to promoter and gene-body matrices. |
| `19_mlh1_msi` | Repeats the MSI adjustment with continuous MLH1 promoter methylation. |
| `20_methylation_models` | Methylation-only and late-fusion models; DTP scored on methylation. |
| `21_tf_activity` | TF activity (decoupler, CollecTRI) and its association with DTP and response. |
| `22_regulatory_architecture` | Are DTP genes enhancer-rich in colon tumour chromatin? Matched-background permutations. |
| `23a_methylation_context_prep.R`, `23_methylation_context` | DTP methylation vs response, split by CpG-island context. |
| **Output** | |
| `11_build_dashboard_data` | Compact tables for `app.py`. Runs last. |

## Core techniques

**Signature scoring** (`fivefu.signatures`). Genes are ranked within each sample; a signature's score is its mean percentile rank, z-scored against the exact null for random gene sets of the same size. DTP is scored as up minus down. Ranks make RNA-seq TPM and raw counts comparable without normalisation. Old symbols are resolved through the HGNC alias map at load.

**Model** (`fivefu.modeling`). `SelectKBest(k=1500) → StandardScaler → ElasticNet(alpha=0.02, l1_ratio=0.5)`, all inside the CV fold. Reported r is the mean over 5 repeats of 5-fold CV, with a 95% interval across repeats.

**Lineage de-confounding.** Within each fold, tissue means of X and y (and the tissue SD of y) are estimated on training rows only and removed from both train and test. The target becomes "resistance relative to lines of the same tissue".

**Statistics** (`fivefu.stats`). Bootstrap CIs on correlations, partial correlations by residualising on covariates, hypergeometric enrichment of selected genes against each signature.

## Decisions

| # | Decision | Reason |
|---|---|---|
| D1 | Pan-cancer is the main modelling arm; colorectal-only uses module scores | Only 43 to 46 colorectal lines |
| D2 | Target is AUC, not LN_IC50 | 53 to 70% of IC50s are extrapolated past the top dose |
| D3 | Train on GDSC1, check on GDSC2 | GDSC1 has twice the dynamic range for 5-FU |
| D4 | Report r against the r ≈ 0.60 screen-agreement ceiling | Assay noise caps any model |
| D5 | Lineage and proliferation baselines are mandatory | Lineage alone reaches 69% of the ceiling |
| D7 | Do not log-transform `rsem_tpm` | It is already log2(TPM+1) |
| D8 | Use Cell Model Passports' `duplicate` flag | Some lines were sequenced twice |
| D10 | Describe GDSC2 as a re-measurement | 882 of its 943 lines are also in GDSC1 |
| D11 | Cell Model Passports over DepMap | Shared `SANGER_MODEL_ID`, no name matching |
| D12 | Signatures: DTP (bidirectional), RSC, CBC, Fetal; CellCycle as confounder | DTP overlaps RSC by only 3% |
| D13 | Rank-based scoring, ssGSEA alongside | Per-sample scoring needed for Arm B and the dashboard |
| D14 | Fetal and MYC not used as model features | Fetal r = 0.96 with RSC; MYC tracks CellCycle |
| D16 | Repair corrupted symbols at load, never in `data/raw` | e.g. `VNN1.00` |
| D17 | Exact analytic null, not Monte Carlo | 200 random draws reproduced at only r = 0.74 |
| D18 | Solid tumours only | The signatures are epithelial programmes |
| D19 | Treat MSI as a confounder of DTP | Associated with both AUC and DTP |
| D21 | Report performance split by assay resolution | The model has no signal among lines at the AUC ceiling |

(D6, D9, D15 and D20 were minor or superseded; they remain in git history.)

## Corrections made along the way

| Error | Effect | Fix |
|---|---|---|
| Monte Carlo null of 200 draws | MYC and CellCycle scores were mostly noise | Exact closed-form null |
| Tissue means computed on all rows | De-confounded r reported as 0.279 | Means from training rows only: 0.182 |
| Single train/test split for colorectal | Reported r = -0.06 | Repeated CV: 0.116 |
| Symbol vintage mismatch | DTP recovered 91% of genes in GDSC, 67% in the time course | HGNC alias resolution |
| Literal `None` entries in DTP lists | Recovery looked 9 points worse | Placeholders removed at load |
