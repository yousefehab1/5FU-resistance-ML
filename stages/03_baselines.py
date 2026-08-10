"""
stages/03_baselines.py
=======================

Establishes what has to be beaten before any real model is trained. This
script deliberately contains no machine learning worth the name -- that is
the point: no claim about expression carrying drug-specific signal until
these are on the table.

  B1  predict the training mean         -> defines R2 = 0
  B2  lineage only (TCGA_DESC)          -> tissue of origin
  B3  proliferation only (CellCycle)    -> 5-FU kills dividing cells
  B4  lineage + proliferation           -> the real bar to clear
  B5  PERMUTED LABELS                   -> leakage tripwire, must score ~0

B5 is the one to rerun after any change to modelling code: shuffle the
target, rerun unchanged, confirm ~0. If it doesn't, something in the
pipeline is leaking test information into training.

Also runs the shared-gene diagnostic (RSC/CBC/Fetal/MYC/CellCycle share
genes by construction, so their raw correlations partly reflect that, not
biology) and the within-COREAD association + GDSC2 replication table.

INPUTS   data/processed/{X,y,signature_scores}_{GDSC1,GDSC2}.parquet
OUTPUTS  data/processed/baseline_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from lib.io import load_screen
from lib.report import banner, report_metric, write_and_report
from lib.signatures import background_score, load_alias_map, load_signatures, rank_matrix

CV = KFold(n_splits=5, shuffle=True, random_state=C.SEED)


def cv_predict(X, y, categorical=()):
    """
    Cross-validated Ridge predictions. Scaling/encoding live INSIDE the
    Pipeline so they refit on the training part of each fold only -- doing
    them before cross_val_predict would leak test information into
    training and silently inflate every baseline below.
    """
    num = [c for c in X.columns if c not in categorical]
    steps = []
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                       list(categorical)))
    if num:
        steps.append(("num", StandardScaler(), num))
    pipe = Pipeline([("prep", ColumnTransformer(steps)), ("model", Ridge(alpha=1.0))])
    return cross_val_predict(pipe, X, y, cv=CV)


def main():
    banner("1. SCORED SIGNATURES")
    X, y, scores = load_screen(C.TRAIN)
    print(f"  {C.TRAIN}: {X.shape[0]:,} lines x {X.shape[1]:,} genes")
    print(f"\n  score correlations (pan-cancer):")
    print(scores.corr().round(2).to_string())

    banner("2. BASELINES")
    target = y[C.TARGET].to_numpy()
    ceiling = C.CEILING_R[C.TARGET]
    lin = y[[C.LINEAGE_COL]].fillna("UNKNOWN")
    results = []

    rmse0 = np.sqrt(((target - target.mean()) ** 2).mean())
    print(f"  {'B1 mean only':<28} r= 0.000  rho= 0.000  R2=  0.000  "
          f"RMSE={rmse0:.4f}     0.0% of ceiling")
    results.append(dict(baseline="B1 mean only", pearson_r=0.0, spearman_rho=0.0,
                         r2=0.0, rmse=rmse0, pct_of_ceiling=0.0))

    def add(name, y_true, y_pred):
        m = report_metric(name, y_true, y_pred, ceiling)
        results.append(dict(baseline=name, pearson_r=m["pearson_r"], spearman_rho=m["spearman_rho"],
                             r2=m["r2"], rmse=m["rmse"], pct_of_ceiling=m["pct_of_ceiling"]))

    add("B2 lineage only", target, cv_predict(lin, target, (C.LINEAGE_COL,)))

    if "CellCycle" in scores.columns:
        add("B3 proliferation only", target, cv_predict(scores[["CellCycle"]], target))
        combo = lin.copy()
        combo["CellCycle"] = scores["CellCycle"].to_numpy()
        add("B4 lineage + proliferation", target, cv_predict(combo, target, (C.LINEAGE_COL,)))

    # B5: shuffle the target and rerun the pipeline unchanged. With no real
    # relationship left, an honest pipeline must score ~0.
    allf = lin.copy()
    for c in scores.columns:
        allf[c] = scores[c].to_numpy()
    shuffled = np.random.default_rng(C.SEED).permutation(target)
    add("B5 PERMUTED LABELS", shuffled, cv_predict(allf, shuffled, (C.LINEAGE_COL,)))

    modelled = [c for c in C.MODELED_MODULES if c in scores.columns]
    mf = lin.copy()
    for c in modelled:
        mf[c] = scores[c].to_numpy()
    add("   lineage + prolif + modules", target, cv_predict(mf, target, (C.LINEAGE_COL,)))
    print(f"\n  modelled: {modelled}")
    print(f"  excluded: {[c for c in scores.columns if c not in modelled]}  "
          f"(redundant with modelled features -- see config.REFERENCE_MODULES)")

    write_and_report(pd.DataFrame(results),
                      C.PROCESSED / "baseline_results.csv", "baseline_results.csv")

    banner("3. SHARED-GENE DIAGNOSTIC")
    # Signatures that share genes correlate partly by construction. This
    # quantifies how much. The trimmed lists are a DIAGNOSTIC, not the
    # primary analysis: trimming a published signature redefines it, and
    # the shared genes here are the YAP/TAZ wound-response core, not noise.
    sigs = load_signatures(X.columns, alias_map=load_alias_map(), verbose=False)
    ranks = rank_matrix(X)
    crc_mask = (y[C.LINEAGE_COL] == C.CRC).to_numpy()
    for a, b in C.OVERLAP_PAIRS:
        if a not in sigs or b not in sigs:
            continue
        ga, gb = set(sigs[a]), set(sigs[b])
        sh = ga & gb
        if not sh:
            print(f"  {a} vs {b}: no shared genes")
            continue
        fa, fb = background_score(ranks, sorted(ga)), background_score(ranks, sorted(gb))
        ta, tb = background_score(ranks, sorted(ga - sh)), background_score(ranks, sorted(gb - sh))
        print(f"  {a} vs {b}: {len(sh)} shared ({100 * len(sh) / min(len(ga), len(gb)):.0f}% of smaller)")
        print(f"    pan-cancer  full r={stats.pearsonr(fa, fb)[0]:+.3f} -> "
              f"trimmed r={stats.pearsonr(ta, tb)[0]:+.3f}")
        print(f"    COREAD      full r={stats.pearsonr(fa[crc_mask], fb[crc_mask])[0]:+.3f} -> "
              f"trimmed r={stats.pearsonr(ta[crc_mask], tb[crc_mask])[0]:+.3f}")

    banner("4. WITHIN-COREAD ASSOCIATION, AND REPLICATION IN GDSC2")
    _, y2, scores2 = load_screen(C.TEST)
    feats = list(scores.columns)

    print(f"\n  higher {C.TARGET} = more resistant. * = p < 0.05 uncorrected.")
    print(f"  {'feature':<12} {C.TRAIN + ' (n=' + str(int(crc_mask.sum())) + ')':>22}"
          f" {C.TEST + ' (n=' + str(int((y2[C.LINEAGE_COL] == C.CRC).sum())) + ')':>22}")
    crc2 = (y2[C.LINEAGE_COL] == C.CRC).to_numpy()
    for c in feats:
        r1, p1 = stats.pearsonr(scores.loc[crc_mask, c], y.loc[crc_mask, C.TARGET])
        r2_, p2 = stats.pearsonr(scores2.loc[crc2, c], y2.loc[crc2, C.TARGET])
        print(f"  {c:<12} r={r1:+.3f} p={p1:.3f}{'*' if p1 < .05 else ' '}"
              f"      r={r2_:+.3f} p={p2:.3f}{'*' if p2 < .05 else ' '}")
    print(f"\n  {len(feats)} features -> Bonferroni threshold p < {0.05 / len(feats):.4f}")
    print(f"  NOTE: {C.TEST} re-measures largely the SAME cell lines, so this tests")
    print(f"  robustness to experimental noise, not generalisation to new lines.")

    banner("5. READ BEFORE MODELLING")
    b5 = [r for r in results if "PERMUTED" in r["baseline"]][0]
    print(f"  B5 permuted-label R2 = {b5['r2']:+.4f}")
    print("  ** LEAKAGE - do not proceed **" if abs(b5["r2"]) > 0.05
          else "  OK - pipeline is not leaking.")
    b4 = [r for r in results if r["baseline"].startswith("B4")]
    if b4:
        print(f"\n  Bar to beat: B4 lineage + proliferation, r = {b4[0]['pearson_r']:.3f}")


if __name__ == "__main__":
    main()
