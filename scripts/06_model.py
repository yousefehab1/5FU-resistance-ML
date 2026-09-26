"""Predict 5-FU AUC from the transcriptome: ElasticNet, Ridge and random forest.

Nested CV chooses the ElasticNet hyperparameters used everywhere else
(alpha = 0.02, l1_ratio = 0.5). --quick skips the search; do not cite it.

Inputs:  data/processed/X_GDSC{1,2}.parquet, y_GDSC{1,2}.parquet
Outputs: data/processed/model_results.csv
Run:     python scripts/06_model.py [--quick]
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

PROJECT_ROOT = project_root()
PROCESSED = PROJECT_ROOT / "data" / "processed"
MODELS = PROJECT_ROOT / "models"

TRAIN, TEST = "GDSC1", "GDSC2"
TARGET = "AUC"
CRC = "COREAD"
BAR = 0.418               # B4 lineage + proliferation, from script 03

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import (CEILING_R, K_GENES, N_FOLDS, SEED,
                           MODELED_MODULES as MODULES)
from fivefu.io import LINEAGE, load_screen
from fivefu.modeling import gene_pipeline
from fivefu.report import banner

CEILING = CEILING_R[TARGET]

CV = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

# python scripts/06_model.py --quick  -> smaller grid, no RandomForest.
# For checking the pipeline runs end to end. Full run for real numbers.
QUICK = "--quick" in sys.argv


def metrics(name, y_true, y_pred, results, note=""):
    r, _ = stats.pearsonr(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    r2 = 1 - ss_res / ((y_true - y_true.mean()) ** 2).sum()
    beats = "beats bar" if r > BAR else ""
    print(f"  {name:<34} r={r:6.3f}  rho={rho:6.3f}  R2={r2:7.3f}  "
          f"{100 * r / CEILING:5.1f}% ceiling  {beats}{note}")
    results.append(dict(model=name, pearson_r=r, spearman_rho=rho, r2=r2,
                        pct_of_ceiling=100 * r / CEILING, beats_bar=r > BAR))
    return r


def main():

    if QUICK:
        print("\n  *** QUICK MODE: reduced grid, no RandomForest. ***")
        print("  *** Numbers are indicative only -- run without --quick for real results. ***")

    banner("1. DATA")
    # drop_haem=False: this script models the whole panel. 07 is the one
    # that restricts to solid tumours, and reports the difference.
    X1, y1, s1 = load_screen(TRAIN, drop_haem=False)
    X2, y2, s2 = load_screen(TEST, drop_haem=False)
    genes = sorted(set(X1.columns) & set(X2.columns))
    X1, X2 = X1[genes], X2[genes]
    print(f"  train {TRAIN}: {X1.shape[0]:,} lines | test {TEST}: {X2.shape[0]:,} lines")
    print(f"  shared genes: {len(genes):,}")
    print(f"  bar to beat: r = {BAR} (lineage + proliferation) | ceiling r = {CEILING}")

    t1 = y1[TARGET].to_numpy()
    t2 = y2[TARGET].to_numpy()
    lin1 = y1[[LINEAGE]].fillna("UNKNOWN")
    results = []

    banner("2. CROSS-VALIDATED ON GDSC1")

    # --- M1 lineage only -------------------------------------------------
    lin_pipe = Pipeline([
        ("prep", ColumnTransformer([("cat", OneHotEncoder(handle_unknown="ignore",
                                                          min_frequency=5), [LINEAGE])])),
        ("model", Ridge(alpha=1.0))])
    metrics("M1 lineage only (the bar)", t1,
            cross_val_predict(lin_pipe, lin1, t1, cv=CV), results)

    # --- M2 modules only -------------------------------------------------
    mod_pipe = Pipeline([("scale", StandardScaler()), ("model", Ridge(alpha=1.0))])
    metrics("M2 modules only", t1,
            cross_val_predict(mod_pipe, s1[MODULES], t1, cv=CV), results)

    # --- M3 transcriptome-wide -------------------------------------------
    # Hyperparameters chosen by inner CV, so the outer CV stays honest.
    # n_jobs=1 throughout: with a 887 x 36,000 matrix, parallel workers each
    # copy the data and exhaust memory. Sequential is slower but survives.
    #
    # THIS SEARCH IS THE PROVENANCE OF alpha=0.02, l1_ratio=0.5.
    # Those values appear in config/analysis.yaml because eight other scripts
    # assume them. This one must never read them from there. A search whose
    # grid came from its own previous answer is not a search, and the project
    # would lose its only evidence that the values were derived rather than
    # guessed. The grid below stays written out, here, on purpose.
    en = GridSearchCV(
        gene_pipeline(ElasticNet(max_iter=3000, random_state=SEED)),
        {"model__alpha": [0.02] if QUICK else [0.005, 0.02, 0.05],
         "model__l1_ratio": [0.5] if QUICK else [0.5, 0.9]},
        cv=3, scoring="r2", n_jobs=1)
    print("  fitting ElasticNet (inner CV for alpha / l1_ratio) ...")
    r_en = metrics("M3 transcriptome (ElasticNet)", t1,
                   cross_val_predict(en, X1, t1, cv=CV), results)

    metrics("M3b transcriptome (Ridge)", t1,
            cross_val_predict(gene_pipeline(Ridge(alpha=100.0)), X1, t1, cv=CV), results)

    if QUICK:
        print("  [quick] skipping RandomForest")
    else:
        print("  fitting RandomForest (non-linear comparator) ...")
        metrics("M3c transcriptome (RandomForest)", t1,
                cross_val_predict(gene_pipeline(RandomForestRegressor(
                    n_estimators=100, random_state=SEED, n_jobs=1)), X1, t1, cv=CV),
                results)

    # --- M4 transcriptome + modules + lineage ----------------------------
    combo = X1.copy()
    for m in MODULES:
        combo[m] = s1[m].to_numpy()
    metrics("M4 transcriptome + modules", t1,
            cross_val_predict(en, combo, t1, cv=CV), results)

    # --- M5 permuted labels ----------------------------------------------
    # The tripwire. Shuffle the target, rerun the pipeline unchanged. With no
    # real relationship left, an honest pipeline must score ~0. Anything above
    # zero means information is leaking from test folds into training.
    #
    # NOTE the estimator below is not the one the arms above use: alpha=0.01
    # rather than the searched 0.02, max_iter=5000 rather than 3000, and no
    # random_state. That is an inconsistency, not a design. It is preserved
    # rather than tidied because these are the published numbers and
    # normalising it would move them.
    # See docs/LIMITATIONS.md, item 6.
    # A weaker penalty and more iterations make this arm easier to fit, so the
    # tripwire is if anything more sensitive, not less.
    perm = np.random.default_rng(SEED).permutation(t1)
    metrics("M5 PERMUTED LABELS", perm,
            cross_val_predict(gene_pipeline(ElasticNet(alpha=0.01, max_iter=5000)),
                              X1, perm, cv=CV), results)

    banner("3. EXTERNAL TEST ON GDSC2")
    print(f"  D10: {TEST} re-measures largely the same cell lines, so this tests")
    print(f"  robustness to experimental noise, not generalisation to new lines.")
    print()
    print(f"  READ r, NOT R2, HERE. The screens have different AUC distributions")
    print(f"  ({TRAIN} mean {t1.mean():.3f}, {TEST} mean {t2.mean():.3f}) because GDSC2")
    print(f"  compresses the range for this drug (D3). A model trained on one")
    print(f"  predicts on the other's scale, so it is systematically offset and R2")
    print(f"  goes sharply negative even when the ranking transfers well. The model")
    print(f"  transfers in ORDER, not in CALIBRATION -- worth stating rather than")
    print(f"  hiding, and fixable with a simple intercept/slope recalibration.\n")
    en.fit(X1, t1)
    best = en.best_params_
    print(f"  chosen: alpha={best['model__alpha']}, l1_ratio={best['model__l1_ratio']}")
    n_nonzero = int((en.best_estimator_.named_steps["model"].coef_ != 0).sum())
    print(f"  non-zero coefficients: {n_nonzero:,} of {K_GENES:,} selected genes\n")
    metrics("ElasticNet -> GDSC2", t2, en.predict(X2), results)

    mod_pipe.fit(s1[MODULES], t1)
    metrics("modules only -> GDSC2", t2, mod_pipe.predict(s2[MODULES]), results)

    banner("4. HELD-OUT CELL LINES  (true generalisation test)")
    # Unlike the GDSC2 comparison, these lines were never seen during training.
    idx_tr, idx_te = train_test_split(np.arange(len(X1)), test_size=0.25,
                                      random_state=SEED)
    en.fit(X1.iloc[idx_tr], t1[idx_tr])
    metrics("ElasticNet -> held-out GDSC1 lines", t1[idx_te],
            en.predict(X1.iloc[idx_te]), results)

    banner("5. CRC-ONLY  (the dissertation question)")
    crc1 = (y1[LINEAGE] == CRC).to_numpy()
    crc2 = (y2[LINEAGE] == CRC).to_numpy()
    print(f"  n = {crc1.sum()} (train) / {crc2.sum()} (test). Lineage is constant here,")
    print(f"  so the dominant confound is gone -- but so is most of the sample.")
    print(f"  Modules only: transcriptome-wide is not defensible at this n (D1).\n")
    cv_small = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    metrics("CRC modules, CV within GDSC1", t1[crc1],
            cross_val_predict(mod_pipe, s1.loc[crc1, MODULES], t1[crc1], cv=cv_small),
            results)
    mod_pipe.fit(s1.loc[crc1, MODULES], t1[crc1])
    metrics("CRC modules -> GDSC2 COREAD", t2[crc2],
            mod_pipe.predict(s2.loc[crc2, MODULES]), results)

    banner("6. SAVING")
    MODELS.mkdir(exist_ok=True)
    try:
        import joblib
        en.fit(X1, t1)
        joblib.dump({"model": en.best_estimator_, "genes": genes,
                     "target": TARGET, "train": TRAIN}, MODELS / "elasticnet_auc.joblib")
        mod_pipe.fit(s1[MODULES], t1)
        joblib.dump({"model": mod_pipe, "features": MODULES},
                    MODELS / "modules_auc.joblib")
        print(f"  wrote models/elasticnet_auc.joblib and models/modules_auc.joblib")
    except ImportError:
        print("  joblib not installed -- skipping model export (pip install joblib)")

    pd.DataFrame(results).to_csv(PROCESSED / "model_results.csv", index=False)

    banner("7. READ THIS")
    res = pd.DataFrame(results)
    m5 = res[res.model.str.contains("PERMUTED")].iloc[0]
    # Leakage shows up as a POSITIVE score on permuted labels. A large NEGATIVE
    # R2 is expected and harmless: with the target shuffled, the model fits noise
    # and predicts worse than the mean. Testing abs would flag that as
    # leakage, which is a false alarm -- the check is one-sided on purpose.
    leaking = (m5.r2 > 0.05) or (abs(m5.pearson_r) > 0.15)
    print(f"  M5 permuted labels: R2 = {m5.r2:+.4f}, r = {m5.pearson_r:+.4f}")
    print(f"  {'** LEAKAGE - do not proceed **' if leaking else '  OK - not leaking.'}")
    print(f"  (a negative R2 here is expected: shuffled target, model fits noise)")
    best_cv = res.iloc[:6].sort_values("pearson_r", ascending=False).iloc[0]
    print(f"  best cross-validated model: {best_cv.model} at r = {best_cv.pearson_r:.3f}")
    print(f"  the bar was r = {BAR}; the ceiling is r = {CEILING}")
    print(f"\n  Results -> {PROCESSED / 'model_results.csv'}")


if __name__ == "__main__":
    main()
