"""
stages/23_build_dashboard_data.py
==================================

Precompute a compact dataset for the Streamlit dashboard (`app.py`).

STRUCTURAL RULE: this script reads EXCLUSIVELY from files earlier stages
already persisted. It contains no fresh statistical test and no hardcoded
number -- if a headline figure the dashboard needs does not already exist
in a CSV, that is a bug in an earlier stage, not something to patch here.
Hand-typed literals go stale silently when the analysis is rerun; reading
persisted outputs cannot.

The one piece of real computation here is out-of-fold PREDICTIONS for the
"Predict" tab's scatter plot -- these are per-line numbers no upstream
stage had reason to persist, computed with the exact same model
(`lib.modeling.elasticnet_pipeline`) and then affine-recalibrated using
the a/b this project already fit and persisted in stage 06 (NOT refit here).

INPUTS   data/processed/{X,y,signature_scores}_{GDSC1,GDSC2}.parquet
         data/processed/baseline_results.csv, final_model_results.csv,
         multidrug_results.csv, calibration_results.csv,
         mutation_covariate_results.csv, within_crc_association.csv,
         enrichment_results.csv, drug_similarity.csv, gene_target_check.csv,
         dtp_specificity_check.csv, timecourse_scores.csv,
         armB_induction_stats.csv, armB_compositional_control.csv,
         armB_cellcycle_adjusted.csv
OUTPUTS  data/processed/dashboard/{cell_lines,results,findings,timecourse}.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as C
from lib.io import exclude_haem, load_mutation_flags, load_screen
from lib.modeling import oof_predictions
from lib.report import banner, write_and_report

DASH = C.PROCESSED / "dashboard"


def _row(df, key_col, key_val):
    """One row from a persisted CSV, by its natural key -- fails loudly
    (KeyError) if a stage's output shape changed, rather than silently
    falling back to a stale number."""
    hits = df[df[key_col] == key_val]
    if hits.empty:
        raise KeyError(f"{key_val!r} not found in column {key_col!r}")
    return hits.iloc[0]


def main():
    DASH.mkdir(parents=True, exist_ok=True)

    banner("1. CELL LINE TABLE")
    X1, y1, s1 = load_screen(C.TRAIN, with_signatures=True)
    X2, y2 = load_screen(C.TEST, with_signatures=False)
    genes = sorted(set(X1.columns) & set(X2.columns))

    keep, y1s, X1s, s1s = exclude_haem(y1, X1[genes], s1)
    ts = y1s[C.TARGET].to_numpy()

    # Out-of-fold predictions for display -- same model, same genes as
    # stage 06's calibration fit, so applying stage 06's a/b to them is a
    # like-for-like recalibration rather than a mismatched one.
    oof = oof_predictions(X1s, ts)
    print(f"  out-of-fold predictions for {len(ts):,} solid lines "
          f"(r = {np.corrcoef(ts, oof)[0, 1]:.3f})")

    cal = pd.read_csv(C.PROCESSED / "calibration_results.csv")
    cal_row = _row(cal, "stage", "calibrated_heldout")
    a, b = cal_row["a"], cal_row["b"]
    predicted = a * oof + b
    print(f"  calibrated with stage 06's a={a:.3f}, b={b:.3f} "
          f"(held-out-half GDSC2 fit, not refit here)")

    tp53_flags = load_mutation_flags(["TP53"], model_ids=y1s.index)
    tp53 = ["mutant" if tp53_flags.loc[i, "TP53"] == 1 else "wild-type/unknown"
            for i in y1s.index]

    tab = pd.DataFrame({
        "model_id": y1s.index,
        "cell_line": y1s["CELL_LINE_NAME"].values,
        "lineage": y1s[C.LINEAGE_COL].values,
        "msi": y1s["msi_status"].fillna("Unknown").values,
        "tp53": tp53,
        "auc_gdsc1": ts,
        "predicted_auc": predicted,
        "auc_gdsc2": [y2[C.TARGET].get(i, np.nan) for i in y1s.index],
    })
    for m in C.ALL_MODULES:
        if m in s1s.columns:
            tab[m] = s1s[m].values
    write_and_report(tab, DASH / "cell_lines.csv", "cell_lines.csv")

    banner("2. TIME-COURSE")
    tc_path = C.PROCESSED / "timecourse_scores.csv"
    if tc_path.exists():
        tc = pd.read_csv(tc_path)
        cols = ["sample", "timepoint_h"] + [m for m in C.ALL_MODULES if m in tc.columns]
        write_and_report(tc[cols], DASH / "timecourse.csv", "timecourse.csv")
    else:
        print(f"  !! {tc_path.name} not found -- run stage 08 first. Tab will be hidden.")

    banner("3. RESULTS SUMMARY")
    baseline = pd.read_csv(C.PROCESSED / "baseline_results.csv")
    final = pd.read_csv(C.PROCESSED / "final_model_results.csv")
    multidrug = pd.read_csv(C.PROCESSED / "multidrug_results.csv")

    def from_final(name):
        r = _row(final, "model", name)
        return r["r_mean"], r["r_lo"], r["r_hi"]

    def from_multidrug(name):
        r = _row(multidrug, "model", name)
        return r["r_mean"], r["r_lo"], r["r_hi"]

    b2 = _row(baseline, "baseline", "B2 lineage only")
    b2b = _row(baseline, "baseline", "B2b lineage only (solid tumours)")
    b5 = _row(baseline, "baseline", "B5 PERMUTED LABELS")
    gdsc2_ext = _row(cal, "stage", "uncalibrated")

    rows = [
        ("Lineage only (all lineages)", b2["pearson_r"], np.nan, np.nan,
         "the baseline bar"),
        ("Lineage only (solid tumours)", b2b["pearson_r"], np.nan, np.nan,
         "the bar the transcriptome model actually has to clear"),
        ("Transcriptome, raw AUC", *from_final("transcriptome -> raw AUC"),
         "most of this is tissue identity"),
        ("Transcriptome, lineage de-confounded",
         *from_final("transcriptome -> AUC, lineage de-confounded"),
         "within-tissue signal"),
        ("Transcriptome, 5-FU specific",
         *from_multidrug("expression -> 5-FU SPECIFIC (general removed)"),
         "generic drug-sensitivity removed"),
        ("Transcriptome -> GDSC2 (external, uncalibrated)",
         gdsc2_ext["pearson_r"], np.nan, np.nan,
         "independent screen, shared genes only"),
        ("Modules only", *from_final("modules -> raw AUC"), ""),
        ("Pan-solid model -> CRC (transfer)",
         *from_final("transfer pan-solid -> CRC"), "beats training on CRC directly"),
        ("CRC-only model", *from_final("CRC-only modules"), "n=43 too small"),
        ("Permuted labels (leakage check)", b5["pearson_r"], np.nan, np.nan,
         "must be ~0"),
    ]
    write_and_report(pd.DataFrame(rows, columns=["model", "r", "ci_lo", "ci_hi", "note"]),
                      DASH / "results.csv", "results.csv")

    banner("4. THE DTP HYPOTHESIS, EVERY TEST RUN")
    crc_assoc = pd.read_csv(C.PROCESSED / "within_crc_association.csv")
    mut = pd.read_csv(C.PROCESSED / "mutation_covariate_results.csv")
    dtp_spec = pd.read_csv(C.PROCESSED / "dtp_specificity_check.csv")
    arm_b = pd.read_csv(C.PROCESSED / "armB_induction_stats.csv")
    enrich = pd.read_csv(C.PROCESSED / "enrichment_results.csv")

    dtp_crc = _row(crc_assoc, "feature", "DTP")
    msi_adj = mut[(mut.screen == "GDSC1") & (mut.test == "DTP vs AUC | MSI")].iloc[0]
    tp53_adj = mut[(mut.screen == "GDSC1") & (mut.test == "DTP vs AUC | TP53")].iloc[0]
    gen = _row(dtp_spec, "target", "general chemosensitivity")
    spec = _row(dtp_spec, "target", "5-FU specific (general removed)")
    dtp_trend = _row(arm_b, "feature", "DTP")
    dtp_enrich = _row(enrich, "signature", "DTP_up")

    findings = [
        ("DTP vs 5-FU resistance, GDSC1 COREAD",
         f"{dtp_crc['r_gdsc1']:+.3f}", f"{dtp_crc['p_gdsc1']:.3f}",
         f"n={int(dtp_crc['n_gdsc1'])}"),
        ("DTP vs 5-FU resistance, GDSC2 COREAD",
         f"{dtp_crc['r_gdsc2']:+.3f}", f"{dtp_crc['p_gdsc2']:.3f}",
         f"n={int(dtp_crc['n_gdsc2'])}, replicates"),
        ("DTP adjusted for MSI", f"{msi_adj['r']:+.3f}", f"{msi_adj['p']:.3f}",
         "MSI is a confounder"),
        ("DTP adjusted for TP53", f"{tp53_adj['r']:+.3f}", f"{tp53_adj['p']:.3f}",
         "TP53 is not"),
        ("DTP vs generic drug sensitivity", f"{gen['r_with_dtp']:+.3f}",
         f"{gen['p_value']:.3f}", "not generic fragility"),
        ("DTP vs 5-FU-specific component", f"{spec['r_with_dtp']:+.3f}",
         f"{spec['p_value']:.3f}", "specific to 5-FU"),
        ("DTP induction over time-course", f"{dtp_trend['trend_r']:+.3f}",
         f"{dtp_trend['trend_p']:.3f}", "exceeds compositional null"),
        ("DTP_up enrichment in model genes", f"{dtp_enrich['fold_enrichment']:.1f}x",
         f"{dtp_enrich['p_value']:.1e}", "unsupervised rediscovery"),
    ]
    write_and_report(pd.DataFrame(findings, columns=["finding", "effect", "p", "note"]),
                      DASH / "findings.csv", "findings.csv")

    banner("DONE")
    print(f"  dashboard data -> {DASH}")
    print("  now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
