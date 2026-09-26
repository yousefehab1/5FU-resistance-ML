# 5FU-resistance-ML

Predicting 5-FU chemotherapy resistance from baseline gene expression in
the GDSC cell-line screens, and testing one candidate mechanism, the
drug-tolerant persister (DTP) programme, from several independent angles:
an HCT116 5-FU time-course, four other chemotherapies, five cohorts of
FOLFOX-treated colorectal patients, and the GDSC DNA methylation data.

- [`docs/METHODS.md`](docs/METHODS.md): what was done and why.
- [`docs/FINDINGS.md`](docs/FINDINGS.md): the numbers, each citing the file it comes from.
- [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md): what those numbers cannot be stretched to claim.

**One rule holds throughout:** every reported number is written by a
stage to a file in `data/processed/`, and everything downstream, the
dashboard included, reads only from those files. Nothing is hand-typed.

## Setup

```bash
python3.12 -m venv .venv
make setup                # pip install -r requirements.txt (pinned)
```

The R preparation steps need R with the Bioconductor packages listed at
the bottom of `requirements.txt`.

## Running

```bash
make r-prep               # R steps 13, 15, 21: GEO download, ComBat, 450K probe filtering
make run                  # every Python stage, in numeric order
make test                 # library tests + golden gate
make dashboard            # streamlit run app.py
```

`make r-prep` only needs rerunning when its inputs change; its outputs
land in `data/processed/` like everything else. Stage 01 caches the 5.7 GB
RNA-seq file as `data/processed/expression_tpm.parquet` on first run.
Stage 11 (five drugs through repeated CV) is the slowest stage.

| Stage | What it does |
|---|---|
| **5-FU model** | |
| `01_build_modelling_table` | 5-FU rows from both screens + expression, as `X_/y_GDSC{1,2}.parquet` |
| `02_score_signatures` | Rank-based signature scores for both screens |
| `03_baselines` | Mean, lineage, proliferation and permuted-label baselines |
| `04_final_model` | ElasticNet, 5×5 repeated CV, lineage de-confounding, CRC transfer |
| `05_enrichment` | Hypergeometric enrichment of the model's genes per signature |
| `06_calibration` | Affine recalibration to GDSC2 (a display fix) |
| `07_mutation_covariates` | Driver mutations; DTP adjusted for MSI and TP53 |
| `08_armB_induction` | HCT116 time-course: trend tests + compositional null |
| `09_multidrug` | General chemosensitivity, drug similarity, TYMS/pathway check |
| **Other drugs** | |
| `10_multidrug_targets` | Targets and QC for oxaliplatin, SN-38, irinotecan, cisplatin |
| `11_multidrug_models` | The stage 04 pipeline per drug, with gene enrichment |
| `12_drug_specificity` | Is DTP specific to 5-FU? Gene overlap; ribosome-biogenesis test |
| **Patients** | |
| `13_clinical_prep.R` | Five FOLFOX GEO cohorts; ComBat; purity, CMS and MSI-like covariates |
| `14_clinical_validation` | Logistic regression of response on each score |
| **Epigenome** | |
| `15_methylation_prep.R` | 450K probe masking and filtering |
| `16_methylation_ingest` | Promoter and gene-body methylation matrices |
| `17_mlh1_msi` | MSI adjustment redone with continuous MLH1 methylation |
| `18_methylation_models` | Methylation-only and late-fusion models; DTP on methylation |
| `19_tf_activity` | TF activity (decoupler, CollecTRI) vs DTP and response |
| `20_regulatory_architecture` | Are DTP genes enhancer-rich in colon tumour chromatin? |
| `21_methylation_context_prep.R` | Promoter probes split by CpG-island context |
| `22_methylation_context` | DTP methylation vs response per CpG context |
| **Output** | |
| `23_build_dashboard_data` | Compact tables for `app.py`; runs last |

Stages 10 to 22 also write a short markdown report each to
`data/processed/reports/`.

## Tests

```bash
make test
```

- `test_library.py`: unit tests for `lib/`, including a synthetic guard
  that fold-internal lineage de-confounding cannot leak (on pure
  lineage-encoded noise the CV must score near zero), hypergeometric
  edge cases, partial correlation and the signature scoring null.
- `test_score_cohort.py`: `lib.cohorts.score_cohort` reproduces the
  time-course scores in both gene orientations.
- `test_golden.py`: every result table frozen in `tests/golden/` must be
  reproduced in `data/processed/` to within 1e-9. Skipped when the pipeline has not
  been run. After an intentional change, `make golden-update` re-freezes it.

## Layout

```
config.py            every constant: paths, hyperparameters, cohorts, drug list
lib/
  io.py              loading, missingness filter, haem exclusion, mutation flags
  signatures.py      alias resolution, rank scoring, the analytic null
  modeling.py        pipeline, fold-internal de-confounded CV, CRC transfer
  stats.py           enrichment, partial correlation, bootstrap, compositional null
  cohorts.py         scoring external cohorts (clinical, time-course)
  report.py          compute -> persist -> print helpers
stages/              the pipeline, 01 to 23, run in numeric order
scripts/one_time/    R exports of annotation (HGNC aliases, 450K manifest, TSS)
tests/               library tests, cohort-scoring test, golden gate
app.py               Streamlit dashboard over stage 23's output
docs/                METHODS, FINDINGS, LIMITATIONS
```

`data/raw/`, `data/processed/`, `models/` and `figures/` are gitignored.

## Data

All inputs go in `data/raw/` (about 12 GB) and are never modified.

| File(s) | Source |
|---|---|
| `GDSC{1,2}_fitted_dose_response_24Jul22.csv`, `screened_compounds_rel_8.4.csv`, `Cell_Lines_Details.xlsx` | GDSC release 8.4, https://www.cancerrxgene.org/downloads |
| `rnaseq_all_20260323.csv`, `model_list_20260724.csv`, `mutations_summary_20260724.csv` | Cell Model Passports, https://cellmodelpassports.sanger.ac.uk/downloads |
| `methylation/GSE68379_Matrix.processed.txt.gz` | GEO GSE68379 (GDSC 450K methylation) |
| `methylation/humanmethylation450_manifest.csv` | `scripts/one_time/export_450k_manifest.R` |
| `geo_clinical/` | GEO GSE28702, GSE19860, GSE69657, GSE72970, GSE104645 (fetched by stage 13) |
| `atac/` | TCGA ATAC-seq (Corces et al. 2018), https://gdc.cancer.gov/about-data/publications/ATACseq-AWG; `gene_tss_hg38.csv` from `scripts/one_time/export_gene_tss_hg38.R` |
| `hgnc_alias_map.csv` | `scripts/one_time/export_hgnc_aliases.R` (org.Hs.eg.db) |
| `signatures/*.txt`, `sigs.csv` | Curated gene sets: DTP up/down, RSC, CBC, Fetal, MYC, CellCycle |
| `Sup_Table_2_HCT116_5FU_timecourse_treatment.txt` | HCT116 5-FU time-course, raw counts, 0/6/24/48 h in triplicate |
