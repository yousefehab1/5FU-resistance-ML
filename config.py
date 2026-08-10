"""
config.py
=========

Single source of every constant used across this pipeline.

This project is a clean reimplementation of `5FU-resistance-ML`, targeting
ONLY that project's final, settled methodology (see docs/METHODS.md for the
rationale behind each decision below). Every stage script imports its
constants from here rather than redefining them, which is the main thing the
original codebase did not do consistently (the same `HAEM` set, ElasticNet
hyperparameters, and missingness threshold were copy-pasted independently
into 5-6 scripts each, and drifted the moment one copy was edited without
the others).
"""

from pathlib import Path

# ============================================================================
# PATHS
# ============================================================================
PROJECT_ROOT = Path(__file__).resolve().parent
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
DASHBOARD_DIR = PROCESSED / "dashboard"
SIGDIR = RAW / "signatures"

# Raw input files (named constants so a filename change is a one-line edit).
GDSC1_FILE = RAW / "GDSC1_fitted_dose_response_24Jul22.csv"
GDSC2_FILE = RAW / "GDSC2_fitted_dose_response_24Jul22.csv"
RNASEQ_FILE = RAW / "rnaseq_all_20260323.csv"
MODEL_LIST_FILE = RAW / "model_list_20260724.csv"
MUTATIONS_FILE = RAW / "mutations_summary_20260724.csv"
TIMECOURSE_FILE = RAW / "Sup_Table_2_HCT116_5FU_timecourse_treatment.txt"
ALIAS_MAP_FILE = RAW / "hgnc_alias_map.csv"

# ============================================================================
# COHORT / SCREEN DEFINITION
# ============================================================================
TRAIN = "GDSC1"       # trained on
TEST = "GDSC2"        # independent re-measurement, used for external validation
TARGET = "AUC"        # not LN_IC50 -- see docs/METHODS.md (censoring)
LINEAGE_COL = "TCGA_DESC"
CRC = "COREAD"

# Leukaemias, lymphomas and myelomas -- DTP/regenerative programmes are
# defined in solid epithelial tumours and are not meaningful in blood cancers.
# NB (neuroblastoma) and MB (medulloblastoma) are solid and are kept, though
# neither is epithelial -- worth revisiting if the epithelial argument is
# taken strictly.
HAEM = {"LAML", "ALL", "DLBC", "LCML", "MM", "CLL"}

MIN_LINEAGE_N = 10    # z-scoring a lineage group smaller than this is meaningless

# Measurement-noise ceiling: how well GDSC1 and GDSC2 agree on the SAME
# cell lines. No model predicting one screen from expression can reasonably
# be expected to beat this -- it is the ceiling on achievable r, not 1.0.
CEILING_R = {"AUC": 0.604, "LN_IC50": 0.560}

# ============================================================================
# EXPRESSION / FEATURE PREPROCESSING
# ============================================================================
ID_COL = "SANGER_MODEL_ID"
MAX_GENE_MISSING = 0.10     # drop genes missing in more than this fraction of lines
CHUNK_ROWS = 2_000_000      # streaming chunk size for the 5.7GB rnaseq file

# rsem_tpm from Cell Model Passports is ALREADY log2(TPM+1), not raw TPM
# (confirmed empirically: values top out ~18, per-sample sums ~6e4 rather
# than the ~1e6 raw TPM would give). Do NOT log-transform it again.
EXPR_IS_LOG2 = True

MODEL_META = ["cancer_type", "tissue", "cancer_type_detail", "growth_properties",
              "msi_status", "mutational_burden", "model_name", "gender",
              "age_at_sampling"]
DRUG_COLS = ["CELL_LINE_NAME", "TCGA_DESC", "LN_IC50", "AUC",
             "RMSE", "Z_SCORE", "MIN_CONC", "MAX_CONC"]

# ============================================================================
# SIGNATURES -- D14 decision: DTP/RSC/CBC/CellCycle are MODELED features.
# Fetal and MYC are REFERENCE-ONLY: excluded from modeling as redundant
# (Fetal shares 128 of RSC's 232 genes, r=0.96 with RSC; MYC correlates
# r=0.83 with CellCycle, a proliferation proxy). Never model these two;
# report them only as sensitivity checks.
# ============================================================================
MODELED_MODULES = ["DTP", "RSC", "CBC", "CellCycle"]
REFERENCE_MODULES = ["Fetal", "MYC"]
ALL_MODULES = MODELED_MODULES + REFERENCE_MODULES

# Shared-gene diagnostic pairs (D15) -- signatures sharing genes correlate
# partly by construction; this is a diagnostic, not a reason to trim the
# published lists (the shared genes are the YAP/TAZ wound-response core).
OVERLAP_PAIRS = [("RSC", "CBC"), ("RSC", "Fetal"), ("MYC", "CellCycle")]

# IBD has a canonical-name mapping (sigtools-style CANON dict) but currently
# has NO source .txt file in data/raw/signatures/ -- documented here rather
# than silently dropped. If a signatures/IBD.txt is ever added, it will be
# picked up automatically; nothing else needs to change.
CANON = {"dtp_up": "DTP_up", "dtp_down": "DTP_down",
         "rsc": "RSC", "cbc": "CBC", "fetal": "Fetal",
         "cellcycle": "CellCycle", "cell_cycle": "CellCycle",
         "myc": "MYC", "ibd": "IBD"}
PLACEHOLDERS = {"None", "NA", "NaN", "null", "-", ""}

# ============================================================================
# MODELING
# ============================================================================
SEED = 0
K_GENES = 1500
ELASTICNET_PARAMS = dict(alpha=0.02, l1_ratio=0.5, max_iter=3000, random_state=SEED)
N_REPEATS, N_FOLDS = 5, 5

# ============================================================================
# COVARIATES / MUTATIONS
# ============================================================================
DRIVER_GENES = ["TP53", "KRAS", "BRAF", "PIK3CA", "APC", "SMAD4", "PTEN", "NRAS"]
COVARIATES_CAT = ["msi_status", "growth_properties"]
COVARIATES_NUM = ["mutational_burden"]

# ============================================================================
# MULTIDRUG
# ============================================================================
FU_DRUG_NAME = "5-Fluorouracil"
MIN_DRUG_LINES = 300    # a drug must be screened on at least this many lines

# ============================================================================
# ARM B (HCT116 TIME-COURSE)
# ============================================================================
TIMECOURSE_GENE_COL = "NAME"
TIMECOURSE_MIN_DETECTED = 6   # keep genes with a non-zero count in >=6 of 12 samples
TIMECOURSE_SAMPLES = {
    "PN0129B_P_0h_Ctrl_black":  (0,  "Ctrl"), "PN0129B_P_0h_Ctrl_blue":  (0,  "Ctrl"),
    "PN0129B_P_0h_Ctrl_red":    (0,  "Ctrl"),
    "PN0129B_P_6h_5FU_black":   (6,  "5FU"),  "PN0129B_P_6h_5FU_blue":   (6,  "5FU"),
    "PN0129B_P_6h_5FU_red":     (6,  "5FU"),
    "PN0129B_P_24h_5FU_black":  (24, "5FU"),  "PN0129B_P_24h_5FU_blue":  (24, "5FU"),
    "PN0129B_P_24h_5FU_red":    (24, "5FU"),
    "PN0129B_P_48h_5FU_black":  (48, "5FU"),  "PN0129B_P_48h_5FU_blue":  (48, "5FU"),
    "PN0129B_P_48h_5FU_red":    (48, "5FU"),
}
TIMECOURSE_TIMEPOINTS = [0, 6, 24, 48]
COMPOSITIONAL_NULL_DRAWS = 200   # Arm B's random-gene-set trend null (a
# DIFFERENT test from signature scoring's analytic background_score null --
# do not conflate the two; see docs/LIMITATIONS.md).
