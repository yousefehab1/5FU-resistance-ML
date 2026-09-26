"""Refine the model: solid tumours only, MSI/TP53 covariates, DTP variants.

Inputs:  data/processed/X_*.parquet, y_*.parquet, signature_scores_*.parquet
         data/raw/mutations_summary_20260724.csv
Outputs: data/processed/refined_model_results.csv, dtp_confounders.csv,
         dtp_variant_comparison.csv
Run:     python scripts/07_refined_model.py [--quick]
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import GridSearchCV, KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

PROJECT_ROOT = project_root()
PROCESSED = PROJECT_ROOT / "data" / "processed"
MUTATIONS = PROJECT_ROOT / "data" / "raw" / "mutations_summary_20260724.csv"

TRAIN, TEST = "GDSC1", "GDSC2"
TARGET, CRC = "AUC", "COREAD"

# Leukaemias, lymphomas and myelomas. NB (neuroblastoma) and MB (medulloblastoma)
# are solid tumours and are kept, though they are not epithelial either -- worth
# revisiting if the epithelial argument is taken strictly.

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import (CEILING_R, N_FOLDS, SEED, HAEM,
                           MODELED_MODULES as MODULES)
from fivefu.io import LINEAGE, load_screen
from fivefu.modeling import gene_pipeline
from fivefu.report import banner
from fivefu import signatures as S
from fivefu.stats import partial_corr

CEILING = CEILING_R[TARGET]

COVARIATES_CAT = ["msi_status", "growth_properties"]
COVARIATES_NUM = ["mutational_burden"]

CV = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
QUICK = "--quick" in sys.argv


def metrics(name, y_true, y_pred, results, bar=None):
    r, _ = stats.pearsonr(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    r2 = 1 - ((y_true - y_pred) ** 2).sum() / ((y_true - y_true.mean()) ** 2).sum()
    flag = ""
    if bar is not None:
        flag = "beats bar" if r > bar else ""
    print(f"  {name:<38} r={r:6.3f}  rho={rho:6.3f}  R2={r2:7.3f}  "
          f"{100 * r / CEILING:5.1f}% ceiling  {flag}")
    results.append(dict(model=name, pearson_r=r, spearman_rho=rho, r2=r2,
                        pct_of_ceiling=100 * r / CEILING))
    return r


def covariate_frame(y):
    """Covariates as a modelling frame; categorical ones stay as text."""
    cols = [c for c in COVARIATES_CAT + COVARIATES_NUM if c in y.columns]
    f = y[cols].copy()
    for c in COVARIATES_CAT:
        if c in f.columns:
            f[c] = f[c].fillna("Unknown").astype(str)
    return f, [c for c in COVARIATES_CAT if c in f.columns], \
              [c for c in COVARIATES_NUM if c in f.columns]


def mixed_pipe(cat, num, model=None):
    steps = []
    if cat:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5), cat))
    if num:
        steps.append(("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                                       ("sc", StandardScaler())]), num))
    return Pipeline([("prep", ColumnTransformer(steps)),
                     ("model", model or Ridge(alpha=1.0))])


def main():

    banner("1. COHORT  (refinement 1: haematological lines removed)")
    # The haem filter IS the first refinement this script tests, so the
    # unfiltered panel is loaded alongside the filtered one to measure it.
    Xa, ya, sa = load_screen(TRAIN, drop_haem=False)
    X1, y1, s1 = load_screen(TRAIN, drop_haem=True)
    X2, y2, s2 = load_screen(TEST, drop_haem=True)
    print(f"  {TRAIN}: {len(ya):,} lines -> {len(y1):,} solid "
          f"({len(ya) - len(y1)} haematological removed: {sorted(HAEM)})")
    print(f"  {TEST}:  {len(y2):,} solid lines")
    print(f"  COREAD retained: {(y1[LINEAGE] == CRC).sum()} / {(y2[LINEAGE] == CRC).sum()}")
    print()
    h = ya[LINEAGE].isin(HAEM)
    print(f"  Why this matters: haematological mean AUC {ya.AUC[h].mean():.3f} "
          f"(sd {ya.AUC[h].std():.3f})")
    print(f"                    solid            mean AUC {ya.AUC[~h].mean():.3f} "
          f"(sd {ya.AUC[~h].std():.3f})")
    print(f"  Blood lines are both far more sensitive and more variable, so they")
    print(f"  dominate the pan-cancer signal. Removing them lowers every number")
    print(f"  below -- that is expected, and the remaining comparison is fairer.")

    t1, t2 = y1[TARGET].to_numpy(), y2[TARGET].to_numpy()
    lin1 = y1[[LINEAGE]].fillna("UNKNOWN")
    results = []

    banner("2. NEW BASELINE  (the bar moves)")
    bar = metrics("lineage only, solid tumours", t1,
                  cross_val_predict(mixed_pipe([LINEAGE], []), lin1, t1, cv=CV), results)

    cov1, cat, num = covariate_frame(y1)
    print(f"  covariates: {cat + num}")
    metrics("covariates only (MSI etc)", t1,
            cross_val_predict(mixed_pipe(cat, num), cov1, t1, cv=CV), results, bar)

    lc = pd.concat([lin1, cov1], axis=1)
    metrics("lineage + covariates", t1,
            cross_val_predict(mixed_pipe([LINEAGE] + cat, num), lc, t1, cv=CV), results, bar)

    lcm = lc.copy()
    for m in MODULES:
        lcm[m] = s1[m].to_numpy()
    metrics("lineage + covariates + modules", t1,
            cross_val_predict(mixed_pipe([LINEAGE] + cat, num + MODULES), lcm, t1, cv=CV),
            results, bar)

    if not QUICK:
        banner("3. TRANSCRIPTOME MODEL, SOLID TUMOURS ONLY")
        gene_pipe = gene_pipeline(ElasticNet(max_iter=3000, random_state=SEED))
        en = GridSearchCV(gene_pipe, {"model__alpha": [0.005, 0.02, 0.05],
                                      "model__l1_ratio": [0.5, 0.9]},
                          cv=3, scoring="r2", n_jobs=1)
        print("  fitting ...")
        metrics("transcriptome (ElasticNet), solid", t1,
                cross_val_predict(en, X1, t1, cv=CV), results, bar)
        en.fit(X1, t1)
        genes = sorted(set(X1.columns) & set(X2.columns))
        metrics("transcriptome -> GDSC2 solid", t2, en.predict(X2[X1.columns]),
                results, bar)
        # NOTE this estimator is spelled out rather than taken from config.
        # It omits l1_ratio, which sklearn then defaults to 0.5, and omits
        # random_state, which ElasticNet only consults under
        # selection="random" and so does not use here. Both land on the
        # configured values; left as written to keep the published numbers.
        # See docs/LIMITATIONS.md, item 6.
        perm = np.random.default_rng(SEED).permutation(t1)
        metrics("PERMUTED LABELS", perm, cross_val_predict(
            gene_pipeline(ElasticNet(alpha=0.02, max_iter=3000)),
            X1, perm, cv=CV), results)

    banner("4. THE MSI CONFOUND  (refinement 2 -- the important result)")
    print("  MMR deficiency is mechanistically linked to 5-FU handling, so MSI is")
    print("  not just any covariate. In COREAD it is associated with BOTH the")
    print("  target and DTP, which is the definition of a confounder.\n")
    # TP53 is the other candidate confounder: p53 mediates the 5-FU
    # stem-cell response. Adjusting for it is the control that shows the
    # weakening is specific to MSI.
    mut = pd.read_csv(MUTATIONS, low_memory=False)
    tp53_mut = set(mut[mut.gene_symbol == "TP53"].model_id)
    conf = []
    for lab, y_, s_ in [(TRAIN, y1, s1), (TEST, y2, s2)]:
        m = (y_[LINEAGE] == CRC) & y_.msi_status.notna()
        yy, ss = y_[m], s_[m.to_numpy()]
        auc = yy[TARGET].to_numpy()
        msi = (yy.msi_status == "MSI").astype(float).to_numpy()
        tp53 = np.array([i in tp53_mut for i in yy.index], dtype=float)
        n, n_msi = int(m.sum()), int(msi.sum())
        print(f"  --- {lab} COREAD, n={n} (MSI {n_msi}, MSS {n - n_msi})")
        tests = [("MSI vs AUC", *stats.pearsonr(msi, auc)),
                 ("DTP vs AUC", *stats.pearsonr(ss.DTP.to_numpy(), auc)),
                 ("MSI vs DTP", *stats.pearsonr(msi, ss.DTP.to_numpy()))]
        for nm, r, p in tests:
            print(f"      {nm:<26} r={r:+.3f}  p={p:.4f}")
        r, p = partial_corr(ss.DTP.to_numpy(), auc, [msi])
        tests.append(("DTP vs AUC | MSI", r, p))
        print(f"      {'DTP vs AUC | MSI':<26} r={r:+.3f}  p={p:.4f}   <-- adjusted")
        oth = [c for c in ss.columns if c != "DTP"]
        r, p = partial_corr(ss.DTP.to_numpy(), auc,
                            [msi] + [ss[c].to_numpy() for c in oth])
        tests.append(("DTP vs AUC | MSI + modules", r, p))
        print(f"      {'DTP vs AUC | MSI + modules':<26} r={r:+.3f}  p={p:.4f}")
        r, p = partial_corr(ss.DTP.to_numpy(), auc, [tp53])
        tests.append(("DTP vs AUC | TP53", r, p))
        print(f"      {'DTP vs AUC | TP53':<26} r={r:+.3f}  p={p:.4f}   <-- control")
        print()
        conf += [dict(screen=lab, n=n, n_msi=n_msi, n_mss=n - n_msi,
                      test=nm, r=r, p=p) for nm, r, p in tests]
    pd.DataFrame(conf).to_csv(PROCESSED / "dtp_confounders.csv", index=False)

    banner("5. DTP VARIANTS  (refinement 3: up-only vs down-only vs bidirectional)")
    print("  D12 assumed bidirectional was better. Testing rather than assuming.\n")
    rows = []
    for lab, Xd, y_, s_ in [(TRAIN, X1, y1, s1), (TEST, X2, y2, s2)]:
        sigs = S.load_signatures(Xd.columns, S.load_alias_map(required=False), verbose=False)
        ranks = S.rank_matrix(Xd)
        up = S.background_score(ranks, sigs["DTP_up"])
        dn = S.background_score(ranks, sigs["DTP_down"])
        crc = (y_[LINEAGE] == CRC).to_numpy()
        auc = y_[TARGET].to_numpy()
        print(f"  --- {lab} COREAD (n={int(crc.sum())})")
        print(f"      corr(DTP_up, DTP_down) = {np.corrcoef(up, dn)[0, 1]:+.3f}  "
              f"| corr(DTP_down, CellCycle) = "
              f"{np.corrcoef(dn, s_.CellCycle.to_numpy())[0, 1]:+.3f}")
        for nm, v in [("DTP up-only", up), ("DTP down-only", dn),
                      ("DTP bidirectional", up - dn)]:
            r, p = stats.pearsonr(v[crc], auc[crc])
            print(f"      {nm:<20} r={r:+.3f}  p={p:.4f}")
            rows.append(dict(screen=lab, variant=nm, r=r, p=p))
        print()
    pd.DataFrame(rows).to_csv(PROCESSED / "dtp_variant_comparison.csv", index=False)

    pd.DataFrame(results).to_csv(PROCESSED / "refined_model_results.csv", index=False)
    banner("DONE")
    print(f"  -> {PROCESSED / 'refined_model_results.csv'}")
    print(f"  -> {PROCESSED / 'dtp_variant_comparison.csv'}")


if __name__ == "__main__":
    main()
