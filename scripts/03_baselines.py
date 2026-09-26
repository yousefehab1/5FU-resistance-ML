"""Score the signatures and fit the baselines any model has to beat.

  B1 training mean   B2 lineage   B3 proliferation   B4 lineage + proliferation
  B5 permuted labels (leakage check, must score about 0)

Inputs:  data/processed/X_*.parquet, y_*.parquet; data/raw/signatures/, hgnc_alias_map.csv
Outputs: data/processed/signature_scores_{GDSC1,GDSC2}.parquet (and _ssgsea_),
         baseline_results.csv
Run:     python scripts/03_baselines.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

PROJECT_ROOT = project_root()
PROCESSED = PROJECT_ROOT / "data" / "processed"

TRAIN = "GDSC1"          # D3
TEST = "GDSC2"
TARGET = "AUC"           # D2
CRC = "COREAD"

# D14: Fetal and MYC are excluded from the model and reported separately.
# Fetal shares 128 of RSC's 232 genes and scores at r = 0.96 with it -- one
# axis, not two. MYC correlates 0.83 with CellCycle: a proliferation proxy.
# The pair itself is config's REFERENCE_MODULES, not a list kept here.
HYPOTHESIS_FEATURES = ["DTP", "RSC", "CBC"]
CONFOUNDERS = ["CellCycle"]
OVERLAP_PAIRS = [("RSC", "CBC"), ("RSC", "Fetal"), ("MYC", "CellCycle")]

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import CEILING_R, SEED
from fivefu.io import LINEAGE, load_expression
from fivefu.modeling import ridge_cv_predict
from fivefu.report import banner
from fivefu import signatures as S


def load_and_score(label, alias_map):
    """Load one screen, handle missing values, score signatures.

    Not fivefu.io.load_screen: this is the script that WRITES
    signature_scores_{label}.parquet, so it cannot read it back. It loads the
    expression matrix through the shared loader and produces the scores itself.

    The missingness filter and the median fill inside load_expression are both
    UNSUPERVISED -- they never look at the target -- so doing them outside
    cross-validation cannot leak. That distinction matters: an unsupervised
    filter outside CV is fine, a filter that ranks genes by correlation with
    the target is not.
    """
    X = load_expression(label, impute=True)
    y = pd.read_parquet(PROCESSED / f"y_{label}.parquet")

    print(f"  {label}: {X.shape[0]:,} lines x {X.shape[1]:,} genes")
    sigs = S.load_signatures(X.columns, alias_map)
    scores, ss = S.score_all(X, sigs)
    return X, y, sigs, scores, ss


def evaluate(name, y_true, y_pred, ceiling, results):
    r, _ = stats.pearsonr(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - ss_res / ((y_true - y_true.mean()) ** 2).sum()
    rmse = np.sqrt(ss_res / len(y_true))
    print(f"  {name:<28} r={r:6.3f}  rho={rho:6.3f}  R2={r2:7.3f}  "
          f"RMSE={rmse:.4f}   {100 * r / ceiling:5.1f}% of ceiling")
    results.append(dict(baseline=name, pearson_r=r, spearman_rho=rho,
                        r2=r2, rmse=rmse, pct_of_ceiling=100 * r / ceiling))


def main():

    banner("1. SIGNATURES AND SCORES")
    alias_map = S.load_alias_map(required=False)
    if alias_map:
        print(f"  HGNC alias map: {len(alias_map):,} entries")
    X, y, sigs, scores, ss = load_and_score(TRAIN, alias_map)
    scores.to_parquet(PROCESSED / f"signature_scores_{TRAIN}.parquet")
    ss.to_parquet(PROCESSED / f"signature_scores_ssgsea_{TRAIN}.parquet")

    print(f"\n  score correlations (pan-cancer):")
    print(scores.corr().round(2).to_string())
    if {"RSC", "CBC"} <= set(scores.columns):
        r_rc = scores["RSC"].corr(scores["CBC"])
        print(f"\n  RSC vs CBC: r = {r_rc:+.3f}  "
              f"({'FAILS - should be negative' if r_rc > 0 else 'anti-correlated, control passes'})")

    banner("2. BASELINES")
    target = y[TARGET].to_numpy()
    ceiling = CEILING_R[TARGET]
    lin = y[[LINEAGE]].fillna("UNKNOWN")
    results = []

    rmse0 = np.sqrt(((target - target.mean()) ** 2).mean())
    print(f"  {'B1 mean only':<28} r= 0.000  rho= 0.000  R2=  0.000  "
          f"RMSE={rmse0:.4f}     0.0% of ceiling")
    results.append(dict(baseline="B1 mean only", pearson_r=0.0, spearman_rho=0.0,
                        r2=0.0, rmse=rmse0, pct_of_ceiling=0.0))

    evaluate("B2 lineage only", target,
             ridge_cv_predict(lin, target, (LINEAGE,)), ceiling, results)

    if "CellCycle" in scores.columns:
        evaluate("B3 proliferation only", target,
                 ridge_cv_predict(scores[["CellCycle"]], target), ceiling, results)
        combo = lin.copy()
        combo["CellCycle"] = scores["CellCycle"].to_numpy()
        evaluate("B4 lineage + proliferation", target,
                 ridge_cv_predict(combo, target, (LINEAGE,)), ceiling, results)

    # B5: shuffle the target and rerun the pipeline unchanged. With no real
    # relationship left, an honest pipeline must score ~0.
    allf = lin.copy()
    for c in scores.columns:
        allf[c] = scores[c].to_numpy()
    evaluate("B5 PERMUTED LABELS", np.random.default_rng(SEED).permutation(target),
             ridge_cv_predict(allf, np.random.default_rng(SEED).permutation(target),
                        (LINEAGE,)), ceiling, results)

    modelled = [c for c in HYPOTHESIS_FEATURES + CONFOUNDERS if c in scores.columns]
    mf = lin.copy()
    for c in modelled:
        mf[c] = scores[c].to_numpy()
    evaluate("   lineage + prolif + modules", target,
             ridge_cv_predict(mf, target, (LINEAGE,)), ceiling, results)
    print(f"\n  modelled: {modelled}")
    print(f"  excluded: {[c for c in scores.columns if c not in modelled]}  (D14)")

    pd.DataFrame(results).to_csv(PROCESSED / "baseline_results.csv", index=False)

    banner("3. SHARED-GENE DIAGNOSTIC  (D15)")
    # Signatures that share genes correlate partly by construction. This
    # quantifies how much. The trimmed lists are a DIAGNOSTIC, not the primary
    # analysis: trimming a published signature redefines it, and the shared
    # genes here are the YAP/TAZ wound-response core, not noise.
    ranks = S.rank_matrix(X)
    crc_mask = (y[LINEAGE] == CRC).to_numpy()
    for a, b in OVERLAP_PAIRS:
        if a not in sigs or b not in sigs:
            continue
        ga, gb = set(sigs[a]), set(sigs[b])
        sh = ga & gb
        if not sh:
            print(f"  {a} vs {b}: no shared genes")
            continue
        fa, fb = S.background_score(ranks, sorted(ga)), S.background_score(ranks, sorted(gb))
        ta, tb = S.background_score(ranks, sorted(ga - sh)), S.background_score(ranks, sorted(gb - sh))
        print(f"  {a} vs {b}: {len(sh)} shared ({100 * len(sh) / min(len(ga), len(gb)):.0f}% of smaller)")
        print(f"    pan-cancer  full r={stats.pearsonr(fa, fb)[0]:+.3f} -> "
              f"trimmed r={stats.pearsonr(ta, tb)[0]:+.3f}")
        print(f"    COREAD      full r={stats.pearsonr(fa[crc_mask], fb[crc_mask])[0]:+.3f} -> "
              f"trimmed r={stats.pearsonr(ta[crc_mask], tb[crc_mask])[0]:+.3f}")

    banner("4. WITHIN-COREAD ASSOCIATION, AND REPLICATION IN GDSC2")
    print(f"  Loading {TEST} for replication ...")
    X2, y2, _, scores2, ss2 = load_and_score(TEST, alias_map)
    scores2.to_parquet(PROCESSED / f"signature_scores_{TEST}.parquet")
    ss2.to_parquet(PROCESSED / f"signature_scores_ssgsea_{TEST}.parquet")

    feats = [c for c in scores.columns]
    print(f"\n  higher {TARGET} = more resistant. * = p < 0.05 uncorrected.")
    print(f"  {'feature':<12} {TRAIN + ' (n=' + str(int(crc_mask.sum())) + ')':>22}"
          f" {TEST + ' (n=' + str(int((y2[LINEAGE] == CRC).sum())) + ')':>22}"
          f" {'ssGSEA agree':>13}")
    crc2 = (y2[LINEAGE] == CRC).to_numpy()
    for c in feats:
        r1, p1 = stats.pearsonr(scores.loc[crc_mask, c], y.loc[crc_mask, TARGET])
        r2_, p2 = stats.pearsonr(scores2.loc[crc2, c], y2.loc[crc2, TARGET])
        rho = stats.spearmanr(scores[c], ss[c]).statistic
        print(f"  {c:<12} r={r1:+.3f} p={p1:.3f}{'*' if p1 < .05 else ' '}"
              f"      r={r2_:+.3f} p={p2:.3f}{'*' if p2 < .05 else ' '}"
              f"      {rho:+.3f}")
    print(f"\n  {len(feats)} features -> Bonferroni threshold p < {0.05 / len(feats):.4f}")
    print(f"  NOTE (D10): {TEST} re-measures largely the SAME cell lines, so this")
    print(f"  tests robustness to experimental noise, not generalisation to new lines.")

    banner("5. READ BEFORE MODELLING")
    res = pd.DataFrame(results)
    b5 = res[res.baseline.str.contains("PERMUTED")].iloc[0]
    print(f"  B5 permuted-label R2 = {b5.r2:+.4f}")
    print("  ** LEAKAGE - do not proceed **" if abs(b5.r2) > 0.05
          else "  OK - pipeline is not leaking.")
    b4 = res[res.baseline.str.startswith("B4")]
    if len(b4):
        print(f"\n  Bar to beat: B4 lineage + proliferation, r = {b4.iloc[0].pearson_r:.3f}")


if __name__ == "__main__":
    main()
