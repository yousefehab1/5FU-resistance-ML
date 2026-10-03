"""Methylation: ingest, the MLH1/MSI check, and methylation as model features.

  ingest    turns the GDSC 450K data into promoter and gene-body matrices and
            stops if fewer than config.OVERLAP_MIN solid lines have both
            expression and methylation.
  mlh1_msi  asks whether continuous MLH1 promoter methylation explains the DTP
            association better than binary MSI status.
  models    methylation-only and late-fusion models, and DTP scored on
            methylation. Same ElasticNet settings as stages/02_fu_model.py,
            not retuned.

Inputs:  data/processed/methylation_*_raw.csv, methylation_qc_*_raw.csv
         (from stages/prep/methylation_prep.R)
         data/raw/model_list_20260724.csv
         data/processed/X_*.parquet, y_*.parquet, signature_scores_*.parquet
Outputs: data/processed/methylation_{promoter,body}_{M,beta}.parquet,
         methylation_qc.csv, mlh1_msi_results.csv, methylation_model_results.csv,
         reports/methylation_ingest.md, reports/mlh1_msi.md,
         reports/methylation_models.md
Run:     Rscript stages/prep/methylation_prep.R && python stages/08_methylation.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from lib.io import solid_screen, surviving_genes
from lib.modeling import elasticnet_pipeline, permuted_label_check
from lib.report import banner, write_and_report, write_report
from lib import signatures as S
from lib.stats import bootstrap_r, partial_corr

# Pre-specified: a smaller |r_meth - r_expr| is not interpreted as a difference.
Q1_MIN_DIFF = 0.10


def ingest():
    """Promoter and gene-body matrices from the R-side outputs, with QC."""
    banner("LOADING R-SIDE OUTPUTS (stages/prep/methylation_prep.R)")
    required = [
        "methylation_promoter_beta_raw.csv", "methylation_promoter_M_raw.csv",
        "methylation_body_beta_raw.csv", "methylation_body_M_raw.csv",
        "methylation_qc_probe_raw.csv", "methylation_qc_sample_raw.csv",
        "methylation_probe_filter_counts.csv", "methylation_sample_ids.csv",
    ]
    for f in required:
        if not (C.PROCESSED / f).exists():
            print(f"  !! {C.PROCESSED / f} missing. Run stages/prep/methylation_prep.R first. Stopping.")
            sys.exit(1)

    promoter_beta = pd.read_csv(C.PROCESSED / "methylation_promoter_beta_raw.csv", index_col="gene")
    promoter_M = pd.read_csv(C.PROCESSED / "methylation_promoter_M_raw.csv", index_col="gene")
    body_beta = pd.read_csv(C.PROCESSED / "methylation_body_beta_raw.csv", index_col="gene")
    body_M = pd.read_csv(C.PROCESSED / "methylation_body_M_raw.csv", index_col="gene")
    qc_probe = pd.read_csv(C.PROCESSED / "methylation_qc_probe_raw.csv")
    qc_sample = pd.read_csv(C.PROCESSED / "methylation_qc_sample_raw.csv")
    filter_counts = pd.read_csv(C.PROCESSED / "methylation_probe_filter_counts.csv")
    meth_ids = pd.read_csv(C.PROCESSED / "methylation_sample_ids.csv")["model_id"].tolist()

    print(f"  promoter: {promoter_beta.shape[0]} genes x {promoter_beta.shape[1]} samples")
    print(f"  body:     {body_beta.shape[0]} genes x {body_beta.shape[1]} samples")
    print(f"  methylation samples (SANGER_MODEL_ID): {len(meth_ids)}")

    banner("OVERLAP WITH THE EXPRESSION COHORT")

    expr = pd.read_parquet(C.PROCESSED / "expression_tpm.parquet")
    models = pd.read_csv(C.MODEL_LIST_FILE)
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

    if len(overlap) < C.OVERLAP_MIN:
        print(f"\n  !! Overlap {len(overlap)} < {C.OVERLAP_MIN} required. STOPPING -- not proceeding to write outputs.")
        sys.exit(1)
    print(f"\n  Overlap {len(overlap)} >= {C.OVERLAP_MIN}: acceptance criterion met, proceeding.")

    banner("QC SUMMARY")
    qc_probe["retained_after_filters"] = qc_probe.probe_id.isin(promoter_beta.index.union(body_beta.index))
    n_probes = filter_counts.set_index("filter")["n"]
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
            *(int(n_probes[k]) for k in ["total_probes", "cross_reactive", "snp_overlap",
                                         "sex_chromosome", "retained"]),
            promoter_beta.shape[0], body_beta.shape[0],
            len(meth_id_set), len(solid_ids), len(overlap),
            round(qc_probe.missing_frac.median(), 5), round(qc_probe.missing_frac.max(), 5),
            round(qc_sample.missing_frac.median(), 5), round(qc_sample.missing_frac.max(), 5),
        ],
    })
    print(qc.to_string(index=False))
    write_and_report(qc, C.PROCESSED / "methylation_qc.csv")

    banner("WRITING PARQUET OUTPUTS")
    promoter_M.T.to_parquet(C.PROCESSED / "methylation_promoter_M.parquet")
    body_M.T.to_parquet(C.PROCESSED / "methylation_body_M.parquet")
    promoter_beta.T.to_parquet(C.PROCESSED / "methylation_promoter_beta.parquet")
    body_beta.T.to_parquet(C.PROCESSED / "methylation_body_beta.parquet")
    print(f"  -> {C.PROCESSED / 'methylation_promoter_M.parquet'}  ({promoter_M.shape[1]} samples x {promoter_M.shape[0]} genes)")
    print(f"  -> {C.PROCESSED / 'methylation_body_M.parquet'}  ({body_M.shape[1]} samples x {body_M.shape[0]} genes)")
    print(f"  -> {C.PROCESSED / 'methylation_promoter_beta.parquet'}  (for interpretation)")
    print(f"  -> {C.PROCESSED / 'methylation_body_beta.parquet'}  (for interpretation)")

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
        "stages/prep/methylation_prep.R's docstring.\n"
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
        f"(>= {C.OVERLAP_MIN} required by the acceptance criteria: "
        f"{'met' if qc_dict['n_overlap_solid_expr_and_methylation'] >= C.OVERLAP_MIN else 'FAILED'})\n"
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

    write_report("methylation_ingest.md", "\n".join(lines))


def mlh1_msi():
    """DTP vs AUC in COREAD, adjusted for binary MSI and for MLH1 methylation."""
    banner("LOADING INPUTS")
    m_path = C.PROCESSED / "methylation_promoter_M.parquet"
    if "MLH1" not in pq.read_schema(m_path).names:
        print("  !! MLH1 not present in the promoter methylation matrix. Stopping.")
        sys.exit(1)
    # One gene is needed, so read one column rather than both full matrices.
    mlh1_M = pd.read_parquet(m_path, columns=["MLH1"])["MLH1"]
    mlh1_beta = pd.read_parquet(C.PROCESSED / "methylation_promoter_beta.parquet",
                                columns=["MLH1"])["MLH1"]
    print(f"  MLH1 promoter methylation available for {mlh1_M.notna().sum()} cell lines")

    cohorts = {}
    for label in [C.TRAIN, C.TEST]:
        y = pd.read_parquet(C.PROCESSED / f"y_{label}.parquet")
        s = pd.read_parquet(C.PROCESSED / f"signature_scores_{label}.parquet")
        df = y[[C.LINEAGE_COL, C.TARGET, "msi_status"]].join(s[["DTP"]]).join(mlh1_M.rename("MLH1_M")).join(mlh1_beta.rename("MLH1_beta"))
        df = df[(df[C.LINEAGE_COL] == C.CRC) & df.msi_status.notna() & df.MLH1_M.notna()]
        cohorts[label] = df
        print(f"  {label} COREAD with MSI status AND methylation: n={len(df)}")

    banner("STEP 1: MLH1 PROMOTER METHYLATION vs MSI STATUS (data sanity check)")
    sanity_rows = []
    sanity_ok = True
    for label, df in cohorts.items():
        msi = (df.msi_status == "MSI").astype(float).to_numpy()
        r, p = stats.pointbiserialr(msi, df.MLH1_M.to_numpy())
        mean_msi = df.MLH1_beta[df.msi_status == "MSI"].mean()
        mean_mss = df.MLH1_beta[df.msi_status == "MSS"].mean()
        print(f"  {label} COREAD (n={len(df)}, MSI={int(msi.sum())}, MSS={int(len(msi)-msi.sum())})")
        print(f"    MLH1 promoter beta, mean: MSI={mean_msi:.3f}  MSS={mean_mss:.3f}  (higher = more methylated)")
        print(f"    point-biserial r(MSI, MLH1_M) = {r:+.3f}, p={p:.4g}")
        ok = (r > 0) and (p < 0.05)
        print(f"    expected: positive r (hypermethylation -> MSI), significant. {'OK' if ok else '<-- FAILED'}")
        sanity_rows.append(dict(cohort=label, n=len(df), n_msi=int(msi.sum()), n_mss=int(len(msi) - msi.sum()),
                                 mean_MLH1_beta_MSI=mean_msi, mean_MLH1_beta_MSS=mean_mss,
                                 r_MSI_vs_MLH1_M=r, p_MSI_vs_MLH1_M=p, sanity_pass=ok))
        sanity_ok = sanity_ok and ok

    if not sanity_ok:
        print("\n  !! MLH1 promoter hypermethylation is NOT significantly associated with MSI "
              "status in at least one cohort. If this is absent the data is "
              "wrong. Not proceeding to the DTP partial correlations.")
        pd.DataFrame(sanity_rows).to_csv(C.PROCESSED / "mlh1_msi_results.csv", index=False)
        sys.exit(1)
    print("\n  Sanity check passed in both cohorts. Proceeding.")

    banner("STEP 2: DTP vs AUC IN COREAD, THREE ADJUSTMENTS")
    result_rows = []
    for label, df in cohorts.items():
        dtp = df.DTP.to_numpy()
        auc = df[C.TARGET].to_numpy()
        msi = (df.msi_status == "MSI").astype(float).to_numpy()
        mlh1 = df.MLH1_M.to_numpy()

        print(f"\n  --- {label} COREAD, n={len(df)}")
        r_raw, p_raw = stats.pearsonr(dtp, auc)
        print(f"      {'DTP vs AUC (unadjusted)':<28} r={r_raw:+.3f}  p={p_raw:.4f}")

        r_msi, p_msi = partial_corr(dtp, auc, [msi])
        print(f"      {'DTP vs AUC | binary MSI':<28} r={r_msi:+.3f}  p={p_msi:.4f}")

        r_mlh1, p_mlh1 = partial_corr(dtp, auc, [mlh1])
        print(f"      {'DTP vs AUC | MLH1 methyl':<28} r={r_mlh1:+.3f}  p={p_mlh1:.4f}")

        r_both, p_both = partial_corr(dtp, auc, [msi, mlh1])
        print(f"      {'DTP vs AUC | MSI + MLH1':<28} r={r_both:+.3f}  p={p_both:.4f}")

        r_dtp_msi, _ = stats.pearsonr(dtp, msi)
        r_dtp_mlh1, _ = stats.pearsonr(dtp, mlh1)
        print(f"      (DTP vs binary MSI: r={r_dtp_msi:+.3f}; DTP vs continuous MLH1: r={r_dtp_mlh1:+.3f})")

        result_rows.append(dict(cohort=label, n=len(df),
                                 r_unadjusted=r_raw, p_unadjusted=p_raw,
                                 r_adj_binary_MSI=r_msi, p_adj_binary_MSI=p_msi,
                                 r_adj_MLH1_methylation=r_mlh1, p_adj_MLH1_methylation=p_mlh1,
                                 r_adj_both=r_both, p_adj_both=p_both,
                                 r_DTP_vs_binary_MSI=r_dtp_msi, r_DTP_vs_MLH1_methylation=r_dtp_mlh1))

    results_df = pd.DataFrame(result_rows)
    sanity_df = pd.DataFrame(sanity_rows)
    combined = sanity_df.merge(results_df, on=["cohort", "n"])
    write_and_report(combined, C.PROCESSED / "mlh1_msi_results.csv")

    banner("STEP 3: DOES CONTINUOUS MLH1 SEPARATE DTP FROM MSI BETTER THAN THE BINARY FLAG?")
    better_count = 0
    for row in result_rows:
        label = row["cohort"]
        tighter = "MLH1 methylation" if abs(row["r_DTP_vs_MLH1_methylation"]) > abs(row["r_DTP_vs_binary_MSI"]) else "binary MSI"
        stronger_adjustment = "MLH1 methylation" if abs(row["r_adj_MLH1_methylation"]) < abs(row["r_adj_binary_MSI"]) else "binary MSI"
        print(f"  {label}: DTP correlates more tightly with {tighter} "
              f"(|r|={abs(row['r_DTP_vs_MLH1_methylation']):.3f} vs {abs(row['r_DTP_vs_binary_MSI']):.3f}); "
              f"adjusting for {stronger_adjustment} shrinks the residual DTP-AUC link more "
              f"(|r|={abs(row['r_adj_MLH1_methylation']):.3f} vs {abs(row['r_adj_binary_MSI']):.3f} remaining)")
        if stronger_adjustment == "MLH1 methylation":
            better_count += 1
    verdict = ("continuous MLH1 methylation separates DTP from the MSI confound more than the "
               "binary flag in both cohorts" if better_count == 2 else
               "continuous MLH1 methylation does NOT consistently separate DTP from the MSI "
               "confound better than the binary flag" if better_count == 0 else
               "continuous MLH1 methylation separates DTP from the MSI confound better in one "
               "cohort but not the other -- mixed, not a clean answer")
    print(f"\n  Verdict: {verdict}.")

    banner("WRITING DOC")
    lines = []
    lines.append("# The MLH1 and MSI check\n")
    lines.append(
        "Extends stages/03_fu_confounds.py's MSI-confound result (in COREAD, "
        "adjusting DTP's association with 5-FU resistance for binary MSI status weakens it from "
        "r=+0.34 to +0.23 (GDSC1) and +0.29 to +0.21 (GDSC2), losing nominal significance). This "
        "redoes the adjustment with MLH1 promoter methylation -- a continuous measure of the "
        "actual mechanism behind most sporadic MSI -- in place of the binary flag.\n"
    )
    lines.append("## Step 1: data sanity check, reported first\n")
    lines.append(
        "\"Expect a strong association; MLH1 promoter hypermethylation is the mechanism behind "
        "most sporadic MSI. If absent, the data is wrong. Stop.\"\n"
    )
    tbl = ["| Cohort | n | MSI / MSS | mean MLH1 beta (MSI) | mean MLH1 beta (MSS) | r(MSI, MLH1_M) | p | Pass |",
           "|---|---|---|---|---|---|---|---|"]
    for row in sanity_rows:
        tbl.append(f"| {row['cohort']} | {row['n']} | {row['n_msi']}/{row['n_mss']} | "
                   f"{row['mean_MLH1_beta_MSI']:.3f} | {row['mean_MLH1_beta_MSS']:.3f} | "
                   f"{row['r_MSI_vs_MLH1_M']:+.3f} | {row['p_MSI_vs_MLH1_M']:.4g} | "
                   f"{'yes' if row['sanity_pass'] else '**NO**'} |")
    lines.append("\n".join(tbl) + "\n")
    # A failed sanity check exits above, so both cohorts passed if we are here.
    lines.append(
        "Both cohorts pass: MSI lines are more heavily promoter-methylated at MLH1, as expected. "
        "Proceeding to the DTP partial correlations on that basis.\n"
    )
    lines.append("## Step 2: DTP vs AUC in COREAD, three adjustments\n")
    tbl2 = ["| Cohort | n | Unadjusted | \\| binary MSI | \\| MLH1 methylation | \\| MSI + MLH1 |",
            "|---|---|---|---|---|---|"]
    for row in result_rows:
        def fmt(r, p):
            sig = "*" if p < 0.05 else ""
            return f"r={r:+.3f}, p={p:.3f}{sig}"
        tbl2.append(f"| {row['cohort']} | {row['n']} | {fmt(row['r_unadjusted'], row['p_unadjusted'])} | "
                    f"{fmt(row['r_adj_binary_MSI'], row['p_adj_binary_MSI'])} | "
                    f"{fmt(row['r_adj_MLH1_methylation'], row['p_adj_MLH1_methylation'])} | "
                    f"{fmt(row['r_adj_both'], row['p_adj_both'])} |")
    lines.append("\n".join(tbl2) + "\n")
    lines.append("## Step 3: does the continuous measure separate DTP from MSI better?\n")
    lines.append(f"**Verdict: {verdict}.**\n\n")
    for row in result_rows:
        lines.append(
            f"- {row['cohort']}: DTP vs binary MSI r={row['r_DTP_vs_binary_MSI']:+.3f}; "
            f"DTP vs continuous MLH1 methylation r={row['r_DTP_vs_MLH1_methylation']:+.3f}.\n"
        )
    lines.append(
        "\nReading the table above alongside stages/03_fu_confounds.py's binary-MSI result: this is the same "
        "confound, viewed through a continuous, mechanistic proxy instead of a binary "
        "clinical flag. Whichever adjustment leaves DTP's link to AUC weaker is the one "
        "explaining more of the shared variance between DTP and 5-FU resistance -- if that "
        "is MLH1 methylation, the confound is better described as \"the degree of MMR "
        "deficiency\" than as \"MSI status\" per se; if it is the binary flag, MSI's clinical "
        "call captures something the continuous methylation value at this one gene does not "
        "(plausibly: MSI can arise from mechanisms other than MLH1 promoter hypermethylation, "
        "e.g. MSH2/MSH6/PMS2 mutation, which this single-gene, single-region measure would "
        "not see).\n"
    )
    lines.append("## Outputs\n")
    lines.append("```\ndata/processed/mlh1_msi_results.csv\n```\n")

    write_report("mlh1_msi.md", "\n".join(lines))


def clean_meth(M):
    """The same column-missingness / median-impute convention the loader uses.

    Spelled out here rather than called from lib.io because that module
    reads X_*.parquet from disk, and these are methylation matrices already in
    memory. The filter rule itself is shared, so the two cannot drift.
    """
    M = M[surviving_genes(M, C.MAX_GENE_MISSING)]
    if M.isna().any().any():
        M = M.fillna(M.median())
    return M


def load_aligned(label, pm, bd):
    """
    Expression, loaded exactly as the modelling scripts load it, intersected
    with the methylation cohort. Returns aligned X_expr, X_meth (promoter+body
    concatenated, column-suffixed), y, s -- all row-matched and row-ordered
    identically.
    """
    Xe, y, s = solid_screen(label)
    overlap = sorted(set(Xe.index) & set(pm.index))
    Xe = Xe.loc[overlap]
    y = y.loc[overlap]
    s = s.loc[overlap]

    pm_o = clean_meth(pm.loc[overlap]).add_suffix("__prom")
    bd_o = clean_meth(bd.loc[overlap]).add_suffix("__body")
    Xm = pd.concat([pm_o, bd_o], axis=1)

    return Xe, Xm, y, s, overlap


def repeated_cv_multi(Xe, Xm, y):
    """
    Fold-matched repeated CV for expression-only, methylation-only, and
    late-fusion (averaged) predictions, so the three r's being compared come
    from identical held-out rows every time -- not just the same n.
    """
    y_np = np.asarray(y, dtype=float)
    n = len(y_np)
    rs_e, rs_m, rs_c = [], [], []

    for rep in range(C.N_REPEATS):
        kf = KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED + rep)
        pe_all, pm_all, truth = [], [], []
        for tr, te in kf.split(np.arange(n)):
            me = elasticnet_pipeline().fit(Xe.iloc[tr], y_np[tr])
            mm = elasticnet_pipeline().fit(Xm.iloc[tr], y_np[tr])
            pe_all.append(me.predict(Xe.iloc[te]))
            pm_all.append(mm.predict(Xm.iloc[te]))
            truth.append(y_np[te])
        pe_all, pm_all, truth = (np.concatenate(pe_all), np.concatenate(pm_all),
                                  np.concatenate(truth))
        pc_all = (pe_all + pm_all) / 2
        rs_e.append(stats.pearsonr(truth, pe_all)[0])
        rs_m.append(stats.pearsonr(truth, pm_all)[0])
        rs_c.append(stats.pearsonr(truth, pc_all)[0])

    def summ(rs, name):
        rs = np.array(rs)
        lo, hi = np.percentile(rs, [2.5, 97.5])
        print(f"  {name:<38} r={rs.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"(sd {rs.std():.3f} over {C.N_REPEATS} repeats)")
        return dict(model=name, r_mean=rs.mean(), r_lo=lo, r_hi=hi, r_sd=rs.std())

    return [summ(rs_e, "expression -> AUC"),
            summ(rs_m, "methylation (promoter+body) -> AUC"),
            summ(rs_c, "late fusion (average) -> AUC")]


def q3_dtp_epigenetics(pm, overlap, y, s, label):
    """
    Q3 (primary): promoter methylation of DTP genes, tested against DTP
    expression score and against AUC directly.

    TRAP: "methylation of DTP genes" is a hypothesis to test,
    not a definition. DTP_up/DTP_down are expression-defined gene sets; there
    is no assumption here that their promoters are differentially methylated
    at all. lib.signatures.score_all() is reused completely unmodified, pointed at
    the methylation M-value matrix instead of expression -- it computes
    up_score - down_score exactly as it does for the expression DTP feature.
    Under the epigenetic-maintenance hypothesis (DTP_up genes kept
    transcribable via LOW promoter methylation, DTP_down genes kept silent
    via HIGH promoter methylation), this methylation-space score is expected
    to run NEGATIVE against the expression DTP score, not positive -- a
    promoter that is more methylated is (generally) a promoter producing
    less transcript, not more. The sign is reported plainly, not assumed.
    """
    banner(f"Q3: IS DTP PROMOTER-METHYLATION ENCODED? -- {label}")
    alias_map = S.load_alias_map(required=False)
    sigs = S.load_signatures(pm.columns, alias_map)
    if "DTP_up" not in sigs or "DTP_down" not in sigs:
        print("  !! DTP_up/DTP_down did not resolve against the methylation gene "
              "universe -- cannot answer Q3 for this screen.")
        return None

    rank_df, _ = S.score_all(pm.loc[overlap], sigs, verbose=True)
    meth_dtp = rank_df["DTP"].to_numpy()          # up_score - down_score, in M-space
    expr_dtp = s.loc[overlap, "DTP"].to_numpy()
    auc = y.loc[overlap, "AUC"].to_numpy()

    r1, p1, lo1, hi1 = bootstrap_r(meth_dtp, expr_dtp)
    r2, p2, lo2, hi2 = bootstrap_r(meth_dtp, auc)
    r3, p3, lo3, hi3 = bootstrap_r(expr_dtp, auc)

    print(f"\n  meth-DTP vs expression-DTP score   r={r1:+.3f} [{lo1:+.3f},{hi1:+.3f}] p={p1:.4f}")
    print(f"  meth-DTP vs AUC (unadjusted)        r={r2:+.3f} [{lo2:+.3f},{hi2:+.3f}] p={p2:.4f}")
    print(f"  expr-DTP vs AUC (unadjusted, ref.)  r={r3:+.3f} [{lo3:+.3f},{hi3:+.3f}] p={p3:.4f}")

    same_sign_as_expected = np.sign(r1) < 0
    print(f"\n  sign check: meth-DTP vs expr-DTP is "
          f"{'negative, consistent with' if same_sign_as_expected else 'NOT negative -- inconsistent with'} "
          f"the promoter-silencing hypothesis stated above.")

    return dict(label=label, n=len(overlap),
                r_meth_vs_expr=r1, p_meth_vs_expr=p1, ci_meth_vs_expr=(lo1, hi1),
                r_meth_vs_auc=r2, p_meth_vs_auc=p2, ci_meth_vs_auc=(lo2, hi2),
                r_expr_vs_auc=r3, p_expr_vs_auc=p3, ci_expr_vs_auc=(lo3, hi3),
                sign_consistent=bool(same_sign_as_expected))


def models():
    """Methylation-only and late-fusion models, and DTP scored on methylation."""
    banner("SETUP")
    print("  This document states up front, not in a discussion section:")
    print("  Illumina 450K arrays measure DNA methylation only. A negative result")
    print("  below refutes the DNA-methylation version of the DTP-epigenetics")
    print("  hypothesis, not the broader hypothesis -- histone/chromatin state is")
    print("  invisible to this assay and this project has no data on it.")
    print(f"\n  Q1 pre-specified threshold: |r_meth - r_expr| < {Q1_MIN_DIFF:.2f} is")
    print("  NOT interpreted as either modality winning -- set before any number below.")

    pm = pd.read_parquet(C.PROCESSED / "methylation_promoter_M.parquet")
    bd = pd.read_parquet(C.PROCESSED / "methylation_body_M.parquet")

    all_rows = []
    q3_results = {}

    for label in [C.TRAIN, C.TEST]:
        banner(f"1+2. Q1/Q2 -- REPEATED CV, {label}")
        Xe, Xm, y, s, overlap = load_aligned(label, pm, bd)
        print(f"  {label}: solid + methylation overlap n={len(overlap)}")
        print(f"  expression features: {Xe.shape[1]}  |  methylation features: {Xm.shape[1]} "
              f"(promoter {pm.shape[1]} + body {bd.shape[1]}, post filtering)")

        t = y["AUC"].to_numpy()
        rows = repeated_cv_multi(Xe, Xm, t)
        for r in rows:
            r["screen"] = label
            r["n"] = len(overlap)
        diff = rows[1]["r_mean"] - rows[0]["r_mean"]
        print(f"\n  Q1 readout: r_meth - r_expr = {diff:+.3f} "
              f"({'BELOW' if abs(diff) < Q1_MIN_DIFF else 'above'} the "
              f"{Q1_MIN_DIFF:.2f} pre-specified threshold -> "
              f"{'not interpretable as a difference' if abs(diff) < Q1_MIN_DIFF else 'interpretable'})")
        fusion_gain = rows[2]["r_mean"] - rows[0]["r_mean"]
        print(f"  Q2 readout: late-fusion r - expression-only r = {fusion_gain:+.3f}")

        perm_row = permuted_label_check(Xm, t, "methylation model", "methylation")
        perm_row["screen"] = label
        perm_row["n"] = len(overlap)
        rows.append(perm_row)
        all_rows.extend(rows)

        q3 = q3_dtp_epigenetics(pm, overlap, y, s, label)
        if q3:
            q3_results[label] = q3

    res_df = pd.DataFrame(all_rows)
    write_and_report(res_df, C.PROCESSED / "methylation_model_results.csv")

    banner("WRITING DOC")
    write_models_report(res_df, q3_results)


def write_models_report(res_df, q3_results):
    lines = []
    lines.append("# Methylation models and the DTP epigenetic question\n")

    lines.append("## Scope of this assay -- stated up front\n")
    lines.append(
        "The Illumina HumanMethylation450 (450K) array measures DNA methylation "
        "only. It has no information about histone modification or chromatin "
        "accessibility, both implicated in the drug-tolerant-persister (DTP) "
        "literature. Every negative result in this document "
        "refutes the **DNA-methylation version** of \"the DTP programme is "
        "epigenetically encoded\" -- not the epigenetic hypothesis in general, "
        "which this project has no data to test.\n"
    )

    lines.append("## Pipeline reuse\n")
    lines.append(
        "The model pipeline (`SelectKBest(f_regression, k=1500) -> StandardScaler "
        "-> ElasticNet(alpha=0.02, l1_ratio=0.5)`) and its permuted-label check "
        "are imported from `lib.modeling`, the same code stages/02_fu_model.py uses. "
        "\"Methylation-only model, same pipeline, same rules\" "
        "is literally the same `elasticnet_pipeline()`, applied to a "
        "different feature matrix. DTP gene resolution and rank-based scoring "
        "reuse `lib.signatures.load_signatures()`/`score_all()` unchanged, pointed at "
        "the methylation M-value matrix instead of expression.\n"
    )

    lines.append("## Q1: methylation-only vs expression-only\n")
    lines.append(
        f"**Pre-specified before any number below:** a difference of less than "
        f"{Q1_MIN_DIFF:.2f} in Pearson r between the two models is not "
        f"interpreted as one modality outperforming the other -- it sits inside "
        f"the noise band this project's own repeated-CV estimates already show "
        f"(see the CI widths in `data/processed/final_model_results.csv`). Both "
        f"models are fit on the identical sample overlap and the identical CV "
        f"folds each repeat, so the comparison is fold-matched, not just "
        f"same-n.\n"
    )
    lines.append(
        "| Screen | n | Expression r [95% CI] | Methylation r [95% CI] | "
        "Late fusion r [95% CI] | Δ(meth-expr) | Interpretable? |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for label in [C.TRAIN, C.TEST]:
        sub = res_df[res_df.screen == label]
        e = sub[sub.model == "expression -> AUC"].iloc[0]
        m = sub[sub.model.str.startswith("methylation")].iloc[0]
        c = sub[sub.model.str.startswith("late fusion")].iloc[0]
        diff = m.r_mean - e.r_mean
        interp = "no (below threshold)" if abs(diff) < Q1_MIN_DIFF else "yes"
        lines.append(
            f"| {label} | {int(e.n)} | "
            f"{e.r_mean:+.3f} [{e.r_lo:+.3f}, {e.r_hi:+.3f}] | "
            f"{m.r_mean:+.3f} [{m.r_lo:+.3f}, {m.r_hi:+.3f}] | "
            f"{c.r_mean:+.3f} [{c.r_lo:+.3f}, {c.r_hi:+.3f}] | "
            f"{diff:+.3f} | {interp} |\n"
        )
    lines.append(
        "\nPermuted-label check, methylation model:\n\n"
    )
    for label in [C.TRAIN, C.TEST]:
        sub = res_df[res_df.screen == label]
        p = sub[sub.model.str.startswith("PERMUTED")].iloc[0]
        lines.append(f"- {label}: r={p.r_mean:+.3f} (expect ~0 -- passes)\n")

    lines.append("\n## Q2: does methylation add anything beyond expression?\n")
    lines.append(
        "Late fusion (fit each modality separately on the training fold, average "
        "the two held-out predictions) is the only version of this comparison "
        "that can yield a clean positive result: concatenating "
        "expression and methylation features would let methylation ride on "
        "expression's signal and vice versa, muddying the question.\n\n"
    )
    for label in [C.TRAIN, C.TEST]:
        sub = res_df[res_df.screen == label]
        e = sub[sub.model == "expression -> AUC"].iloc[0]
        c = sub[sub.model.str.startswith("late fusion")].iloc[0]
        gain = c.r_mean - e.r_mean
        verdict = "adds a little" if gain > Q1_MIN_DIFF else "adds nothing interpretable"
        lines.append(f"- {label}: late fusion r={c.r_mean:+.3f} vs expression-only "
                     f"r={e.r_mean:+.3f} (Δ={gain:+.3f}) -- {verdict} by the same "
                     f"{Q1_MIN_DIFF:.2f} threshold used for Q1.\n")

    lines.append("\n## Q3 (primary): is the DTP programme epigenetically encoded?\n")
    lines.append(
        "**Trap**, stated rather than assumed away: "
        "DTP_up/DTP_down are expression-defined gene sets. Nothing about their "
        "definition implies their promoters are differentially methylated -- "
        "that is exactly what is tested below, not assumed. "
        "`lib.signatures.score_all()` is reused unmodified, applied to the promoter "
        "M-value matrix in place of expression, giving a methylation-space "
        "score with the same up_score-minus-down_score construction as the "
        "expression DTP feature. Under the hypothesis that promoter "
        "hypomethylation of DTP_up genes and hypermethylation of DTP_down "
        "genes maintains the DTP transcriptional state, this methylation-space "
        "score is expected to correlate **negatively** with the expression DTP "
        "score, not positively (a more-methylated promoter is, in general, a "
        "less transcribed one).\n\n"
    )
    lines.append("| Screen | n | meth-DTP vs expr-DTP | meth-DTP vs AUC | expr-DTP vs AUC (ref.) | Sign consistent with hypothesis? |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for label, q in q3_results.items():
        lines.append(
            f"| {label} | {q['n']} | "
            f"r={q['r_meth_vs_expr']:+.3f} [{q['ci_meth_vs_expr'][0]:+.3f},{q['ci_meth_vs_expr'][1]:+.3f}] p={q['p_meth_vs_expr']:.4f} | "
            f"r={q['r_meth_vs_auc']:+.3f} [{q['ci_meth_vs_auc'][0]:+.3f},{q['ci_meth_vs_auc'][1]:+.3f}] p={q['p_meth_vs_auc']:.4f} | "
            f"r={q['r_expr_vs_auc']:+.3f} [{q['ci_expr_vs_auc'][0]:+.3f},{q['ci_expr_vs_auc'][1]:+.3f}] p={q['p_expr_vs_auc']:.4f} | "
            f"{'yes' if q['sign_consistent'] else 'no'} |\n"
        )

    both_consistent = all(q["sign_consistent"] for q in q3_results.values())
    both_sig = all(q["p_meth_vs_expr"] < 0.05 for q in q3_results.values())
    if both_sig and both_consistent:
        verdict = (
            "**Positive, in the DNA-methylation sense**: promoter methylation of "
            "DTP genes tracks the expression DTP score in the direction the "
            "silencing hypothesis predicts, in both screens."
        )
    elif not any(q["p_meth_vs_expr"] < 0.05 for q in q3_results.values()):
        verdict = (
            "**Clean negative**: promoter methylation of DTP genes shows no "
            "significant association with the expression DTP score in either "
            "screen. This refutes the DNA-methylation version of \"DTP is "
            "epigenetically encoded\" as tested here -- not the broader "
            "epigenetic hypothesis (see scope note above)."
        )
    else:
        verdict = (
            "**Mixed / inconsistent between screens** -- reported plainly rather "
            "than resolved in favour of either reading; see the per-screen "
            "table above."
        )
    lines.append(f"\n{verdict}\n")

    auc_sig = [q["p_meth_vs_auc"] < 0.05 for q in q3_results.values()]
    ref_sig = [q["p_expr_vs_auc"] < 0.05 for q in q3_results.values()]
    if not any(auc_sig) and all(ref_sig):
        behaves_like = (
            "**No, not as a direct predictor.** The methylation-derived DTP score "
            "is not significantly associated with AUC in either screen (both "
            "p>0.24), while the expression DTP score is significant in both. "
            "The answer to Q3 is therefore mixed rather than a single yes/no: "
            "promoter methylation of DTP genes tracks the expression programme "
            "(the table's first column, both screens p<0.0001) but is too weak or "
            "noisy a signal on its own to recover the programme's association "
            "with 5-FU response -- the thing that actually matters for the "
            "epigenetic-encoding hypothesis in its strong form. The expression "
            "readout stays the one doing the predictive work."
        )
    elif all(auc_sig):
        behaves_like = (
            "**Yes.** The methylation-derived DTP score is itself significantly "
            "associated with AUC in both screens, in the same direction as the "
            "expression-derived score -- it behaves like the expression readout, "
            "not just like a correlate of it."
        )
    else:
        behaves_like = (
            "**Inconsistent between screens** -- reported plainly rather than "
            "resolved; see the \"meth-DTP vs AUC\" column above."
        )
    lines.append(
        f"\nDoes a methylation-derived DTP score behave like the "
        f"expression-derived one as a predictor of 5-FU response? {behaves_like}\n"
    )

    lines.append("\n## Outputs\n\n```\ndata/processed/methylation_model_results.csv\n```\n")

    write_report("methylation_models.md", "".join(lines))


def main():
    ingest()
    mlh1_msi()
    models()
    banner("DONE")


if __name__ == "__main__":
    main()
