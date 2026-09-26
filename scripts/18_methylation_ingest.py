"""Turn the GDSC 450K methylation data into promoter and gene-body matrices.

Inputs:  data/processed/methylation_*_raw.csv, methylation_qc_*_raw.csv (from 18a)
         data/raw/model_list_20260724.csv
Outputs: data/processed/methylation_{promoter,body}_{M,beta}.parquet,
         methylation_qc.csv, reports/18_methylation_ingest.md
Run:     Rscript scripts/18a_methylation_prep.R && python scripts/18_methylation_ingest.py
"""

from pathlib import Path
import sys

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
PROC = ROOT / "data" / "processed"
RAW = ROOT / "data" / "raw"
DOCS = ROOT / "data" / "processed" / "reports"  # generated reports

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import OVERLAP_MIN
from fivefu.report import banner


banner("LOADING R-SIDE OUTPUTS (scripts/18a_methylation_prep.R)")
required = [
    "methylation_promoter_beta_raw.csv", "methylation_promoter_M_raw.csv",
    "methylation_body_beta_raw.csv", "methylation_body_M_raw.csv",
    "methylation_qc_probe_raw.csv", "methylation_qc_sample_raw.csv",
    "methylation_probe_filter_counts.csv", "methylation_sample_ids.csv",
]
for f in required:
    if not (PROC / f).exists():
        print(f"  !! {PROC / f} missing. Run scripts/18a_methylation_prep.R first. Stopping.")
        sys.exit(1)

promoter_beta = pd.read_csv(PROC / "methylation_promoter_beta_raw.csv", index_col="gene")
promoter_M = pd.read_csv(PROC / "methylation_promoter_M_raw.csv", index_col="gene")
body_beta = pd.read_csv(PROC / "methylation_body_beta_raw.csv", index_col="gene")
body_M = pd.read_csv(PROC / "methylation_body_M_raw.csv", index_col="gene")
qc_probe = pd.read_csv(PROC / "methylation_qc_probe_raw.csv")
qc_sample = pd.read_csv(PROC / "methylation_qc_sample_raw.csv")
filter_counts = pd.read_csv(PROC / "methylation_probe_filter_counts.csv")
meth_ids = pd.read_csv(PROC / "methylation_sample_ids.csv")["model_id"].tolist()

print(f"  promoter: {promoter_beta.shape[0]} genes x {promoter_beta.shape[1]} samples")
print(f"  body:     {body_beta.shape[0]} genes x {body_beta.shape[1]} samples")
print(f"  methylation samples (SANGER_MODEL_ID): {len(meth_ids)}")

# ============================================================================
# Overlap with the expression cohort (stop and report if < 300)
# ============================================================================
banner("OVERLAP WITH THE EXPRESSION COHORT")

expr = pd.read_parquet(PROC / "expression_tpm.parquet")
models = pd.read_csv(RAW / "model_list_20260724.csv")
expr_models = models[models.model_id.isin(expr.index)]
is_heme = expr_models.tissue == "Haematopoietic and Lymphoid"
solid_ids = set(expr_models.loc[~is_heme, "model_id"])

print(f"  expression cohort (data/processed/expression_tpm.parquet): {expr.shape[0]} lines total")
print(f"  of which solid (tissue != 'Haematopoietic and Lymphoid'): {len(solid_ids)}")
print(f"  (the modelling tables use 786 solid lines; this model_list split gives {len(solid_ids)}, "
      "most likely a model_list version difference)")

meth_id_set = set(meth_ids)
overlap = solid_ids & meth_id_set
print(f"\n  methylation samples: {len(meth_id_set)}")
print(f"  solid expression samples: {len(solid_ids)}")
print(f"  overlap (solid, has both expression AND methylation): {len(overlap)}")

if len(overlap) < OVERLAP_MIN:
    print(f"\n  !! Overlap {len(overlap)} < {OVERLAP_MIN} required. STOPPING -- not proceeding to write outputs.")
    sys.exit(1)
print(f"\n  Overlap {len(overlap)} >= {OVERLAP_MIN}: acceptance criterion met, proceeding.")

# ============================================================================
# Finalize QC report
# ============================================================================
banner("QC SUMMARY")
qc_probe["retained_after_filters"] = qc_probe.probe_id.isin(promoter_beta.index.union(body_beta.index))
qc = pd.DataFrame({
    "metric": [
        "n_probes_total", "n_probes_cross_reactive_removed", "n_probes_snp_overlap_removed",
        "n_probes_sex_chr_removed", "n_probes_retained",
        "n_genes_promoter", "n_genes_body",
        "n_methylation_samples_mapped", "n_expression_cohort_solid",
        "n_overlap_solid_expr_and_methylation",
        "probe_missingness_median", "probe_missingness_max",
        "sample_missingness_median", "sample_missingness_max",
    ],
    "value": [
        int(filter_counts.set_index("filter").loc["total_probes", "n"]),
        int(filter_counts.set_index("filter").loc["cross_reactive", "n"]),
        int(filter_counts.set_index("filter").loc["snp_overlap", "n"]),
        int(filter_counts.set_index("filter").loc["sex_chromosome", "n"]),
        int(filter_counts.set_index("filter").loc["retained", "n"]),
        promoter_beta.shape[0], body_beta.shape[0],
        len(meth_id_set), len(solid_ids), len(overlap),
        round(qc_probe.missing_frac.median(), 5), round(qc_probe.missing_frac.max(), 5),
        round(qc_sample.missing_frac.median(), 5), round(qc_sample.missing_frac.max(), 5),
    ],
})
print(qc.to_string(index=False))
qc.to_csv(PROC / "methylation_qc.csv", index=False)
print(f"\n  -> {PROC / 'methylation_qc.csv'}")

# ============================================================================
# Write final parquet outputs (M-values for modelling, beta retained for
# interpretation -- both kept, promoter and body always separate matrices)
# ============================================================================
banner("WRITING PARQUET OUTPUTS")
promoter_M.T.to_parquet(PROC / "methylation_promoter_M.parquet")
body_M.T.to_parquet(PROC / "methylation_body_M.parquet")
promoter_beta.T.to_parquet(PROC / "methylation_promoter_beta.parquet")
body_beta.T.to_parquet(PROC / "methylation_body_beta.parquet")
print(f"  -> {PROC / 'methylation_promoter_M.parquet'}  ({promoter_M.shape[1]} samples x {promoter_M.shape[0]} genes)")
print(f"  -> {PROC / 'methylation_body_M.parquet'}  ({body_M.shape[1]} samples x {body_M.shape[0]} genes)")
print(f"  -> {PROC / 'methylation_promoter_beta.parquet'}  (for interpretation)")
print(f"  -> {PROC / 'methylation_body_beta.parquet'}  (for interpretation)")

# ============================================================================
# Write the doc
# ============================================================================
banner("WRITING DOC")
lines = []
lines.append("# Methylation ingest\n")
lines.append(
    "**Why the epigenome step runs:** the clinical analysis found one adjusted association, DTP on the "
    "GPL570 clinical cohort, adjusted for purity + CMS + MSI-NTP + study, OR=1.47/SD, "
    "p=0.040 -- the only significant adjusted result across "
    "every module and both cohorts. Proceeding on that basis.\n"
)
lines.append("## Data source\n")
lines.append(
    "The processed matrix on the GDSC1000 web resources "
    "(Iorio 2016) is no longer available: that page and its direct data URL both return HTTP 410 "
    "Gone. Cell Model Passports' current downloads page (this project's other GDSC data "
    "source) lists Mutation/Expression/CopyNumber/Fusion/Proteomics under Multi-Omic "
    "Datasets, no Methylation category -- not a redirect, genuinely absent. GEO itself "
    "(GSE68379) hosts a processed supplementary file "
    "directly on the series, `GSE68379_Matrix.processed.txt.gz` (beta values at probe level, "
    "confirmed by inspection, not raw signal), fetched via the same NCBI FTP route already "
    "used for the clinical cohorts. IDATs are "
    "not reprocessed. Full detail in "
    "scripts/18a_methylation_prep.R's docstring.\n"
)
lines.append("## Detection p-values: available, and used\n")
lines.append(
    "Detection p-values are available: the GEO file "
    "interleaves an `_AVG.Beta` and a `_Detection.PVal` column per cell line. Beta calls with "
    "detection p >= 0.01 (minfi's own standard threshold) were masked to NA before any "
    "missingness statistics or region aggregation, not silently kept as if confidently "
    "called.\n"
)
lines.append("## Cell-line mapping and QC\n")
qc_dict = dict(zip(qc.metric, qc.value))
lines.append(
    f"- {int(qc_dict['n_methylation_samples_mapped'])} methylation samples mapped to a "
    "SANGER_MODEL_ID (name-normalized match against model_list's `model_name` and "
    "`synonyms` fields; ties from >1 methylation replicate per model_id averaged after "
    "detection-p masking).\n"
    f"- Probe missingness (post detection-p masking): median "
    f"{qc_dict['probe_missingness_median']}, max {qc_dict['probe_missingness_max']}.\n"
    f"- Sample missingness: median {qc_dict['sample_missingness_median']}, max "
    f"{qc_dict['sample_missingness_max']}.\n"
)
lines.append("## Probe filters\n")
lines.append(
    f"- Total probes: {int(qc_dict['n_probes_total']):,}\n"
    f"- Cross-reactive (Chen 2013 + Price 2013 lists, via the `maxprobes` package): "
    f"{int(qc_dict['n_probes_cross_reactive_removed']):,} removed\n"
    f"- SNP-overlapping (any annotated Probe/CpG/single-base-extension rs ID, minfi's own "
    f"default maf=0 threshold -- any known SNP, not just common ones): "
    f"{int(qc_dict['n_probes_snp_overlap_removed']):,} removed\n"
    f"- Sex chromosomes (chrX/chrY): {int(qc_dict['n_probes_sex_chr_removed']):,} removed\n"
    f"- Retained: {int(qc_dict['n_probes_retained']):,} "
    f"({100*qc_dict['n_probes_retained']/qc_dict['n_probes_total']:.1f}%)\n"
)
lines.append("## Region aggregation: promoter and gene body kept separate\n")
lines.append(
    f"- Promoter (TSS1500 or TSS200): {int(qc_dict['n_genes_promoter']):,} genes\n"
    f"- Gene body (Body): {int(qc_dict['n_genes_body']):,} genes\n\n"
    "**Never merged.** Promoter methylation represses transcription; gene-body methylation "
    "correlates positively with it. Averaging the two across probes assigned to a gene "
    "cancels the signal. Each "
    "region aggregated independently as the mean beta across its assigned probes per gene "
    "per sample (probes contributing to both regions for different transcripts of the same "
    "gene are counted in both).\n"
)
lines.append("## Overlap with the expression cohort\n")
lines.append(
    f"- Methylation samples (mapped): {int(qc_dict['n_methylation_samples_mapped'])}\n"
    f"- Solid-tissue lines with expression (data/processed/expression_tpm.parquet, "
    "tissue != 'Haematopoietic and Lymphoid'): "
    f"{int(qc_dict['n_expression_cohort_solid'])}\n"
    f"- **Overlap: {int(qc_dict['n_overlap_solid_expr_and_methylation'])}** "
    f"(>= {OVERLAP_MIN} required by the acceptance criteria: "
    f"{'met' if qc_dict['n_overlap_solid_expr_and_methylation'] >= OVERLAP_MIN else 'FAILED'})\n"
)
lines.append("## M-values for modelling, beta retained for interpretation\n")
lines.append(
    "`M = log2(beta / (1 - beta))`, beta clamped to [1e-4, 1-1e-4] before conversion to avoid "
    "+-Inf at the boundary. Both forms written out (`*_M.parquet` for modelling, "
    "`*_beta.parquet` for interpretation).\n"
)
lines.append("## Outputs\n")
lines.append(
    "```\n"
    "data/processed/methylation_promoter_M.parquet\n"
    "data/processed/methylation_body_M.parquet\n"
    "data/processed/methylation_promoter_beta.parquet\n"
    "data/processed/methylation_body_beta.parquet\n"
    "data/processed/methylation_qc.csv\n"
    "```\n"
)

DOCS.mkdir(parents=True, exist_ok=True)
out_path = DOCS / "18_methylation_ingest.md"
out_path.write_text("\n".join(lines))
print(f"  -> {out_path}")

banner("DONE")
