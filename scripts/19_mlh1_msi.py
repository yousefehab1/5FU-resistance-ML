"""Does continuous MLH1 promoter methylation explain the DTP association better than binary MSI?

Inputs:  data/processed/methylation_promoter_{M,beta}.parquet, signature_scores_*.parquet, y_*.parquet
Outputs: data/processed/mlh1_msi_results.csv, reports/19_mlh1_msi.md
Run:     python scripts/19_mlh1_msi.py
"""

from pathlib import Path
import sys

import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "data" / "processed" / "reports"  # generated reports

TARGET, CRC = "AUC", "COREAD"


from fivefu.io import LINEAGE
from fivefu.report import banner
from fivefu.stats import partial_corr


banner("LOADING INPUTS")
promoter_M = pd.read_parquet(PROC / "methylation_promoter_M.parquet")
promoter_beta = pd.read_parquet(PROC / "methylation_promoter_beta.parquet")
if "MLH1" not in promoter_M.columns:
    print("  !! MLH1 not present in the promoter methylation matrix. Stopping.")
    sys.exit(1)
mlh1_M = promoter_M["MLH1"]
mlh1_beta = promoter_beta["MLH1"]
print(f"  MLH1 promoter methylation available for {mlh1_M.notna().sum()} cell lines")

cohorts = {}
for label in ["GDSC1", "GDSC2"]:
    y = pd.read_parquet(PROC / f"y_{label}.parquet")
    s = pd.read_parquet(PROC / f"signature_scores_{label}.parquet")
    df = y[[LINEAGE, TARGET, "msi_status"]].join(s[["DTP"]]).join(mlh1_M.rename("MLH1_M")).join(mlh1_beta.rename("MLH1_beta"))
    df = df[(df[LINEAGE] == CRC) & df.msi_status.notna() & df.MLH1_M.notna()]
    cohorts[label] = df
    print(f"  {label} COREAD with MSI status AND methylation: n={len(df)}")

# ============================================================================
# Step 1 (must come first): data sanity check
# ============================================================================
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
    pd.DataFrame(sanity_rows).to_csv(PROC / "mlh1_msi_results.csv", index=False)
    sys.exit(1)
print("\n  Sanity check passed in both cohorts. Proceeding.")

# ============================================================================
# Step 2: DTP vs AUC partial correlation, three adjustments
# ============================================================================
banner("STEP 2: DTP vs AUC IN COREAD, THREE ADJUSTMENTS")
result_rows = []
for label, df in cohorts.items():
    dtp = df.DTP.to_numpy()
    auc = df[TARGET].to_numpy()
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
combined.to_csv(PROC / "mlh1_msi_results.csv", index=False)
print(f"\n  -> {PROC / 'mlh1_msi_results.csv'}")

# ============================================================================
# Step 3: does continuous MLH1 separate DTP from the MSI confound better?
# ============================================================================
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

# ============================================================================
# Write doc
# ============================================================================
banner("WRITING DOC")
lines = []
lines.append("# The MLH1 and MSI check\n")
lines.append(
    "Extends scripts/07_refined_model.py section 4 (the D19 MSI-confound result: in COREAD, "
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
lines.append(
    f"{'Both cohorts pass: MSI lines are more heavily promoter-methylated at MLH1, as expected. '  if sanity_ok else '**At least one cohort failed this check. Stopping -- results below are not reported.**'}"
    "Proceeding to the DTP partial correlations on that basis.\n" if sanity_ok else "\n"
)
if sanity_ok:
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
        "\nReading the table above alongside D19's original finding: this is the same "
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

DOCS.mkdir(parents=True, exist_ok=True)
out_path = DOCS / "19_mlh1_msi.md"
out_path.write_text("\n".join(lines))
print(f"  -> {out_path}")

banner("DONE")
