"""
lib/modeling.py
================

The model pipeline and the cross-validation logic every modelling stage shares.

  elasticnet_pipeline()     SelectKBest -> StandardScaler -> ElasticNet
  cv_repeats()              repeated k-fold CV, per-repeat Pearson r
  summarise()               (mean, lo, hi, sd) of per-repeat r
  repeated_cv()             cv_repeats + one printed line + a result row
  oof_predictions()         single-pass out-of-fold predictions
  deconfounded_gene_fit()   the de-confounded model fitted once, its genes
  permuted_label_check()    the pipeline on shuffled labels (leakage check)
  ridge_cv_predict()        cross-validated Ridge baseline predictions
  transfer_to_crc()         CRC-only vs pan-solid transfer comparison

WHY THE CV LOOP IS HAND-ROLLED RATHER THAN A SKLEARN PIPELINE
---------------------------------------------------------------
De-confounding the target means subtracting lineage means of y. Computing
those means over the whole dataset would use test-fold targets, a genuine
leak. An earlier version of this project did exactly that and reported
r=0.279 for the de-confounded association; the corrected, fold-internal
version reports r=0.182. That gap IS the leak.

So within each fold, lineage means for BOTH X and y are estimated on the
TRAINING rows only, then applied to the held-out rows. sklearn's
cross_val_predict cannot express a target transformation. Lineages present in
test but not train are dropped from that fold.

DTYPE
-----
The GDSC matrices are float32. Functions that take a DataFrame fit on it as
given; upcasting to float64 first moves published numbers in the 7th decimal.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

import config as C
from lib.stats import bootstrap_pearson_draws

# A de-confounded fold left with fewer usable test rows than this is skipped
# rather than scored on noise.
MIN_TEST_ROWS = 5


def elasticnet_pipeline():
    """The canonical model: SelectKBest(f_regression, k) -> scale -> ElasticNet.

    SelectKBest lives INSIDE the pipeline so it is refit on the training fold
    only. Selecting genes by correlation with the full target before splitting
    would leak.
    """
    return Pipeline([
        ("select", SelectKBest(f_regression, k=C.K_GENES)),
        ("scale", StandardScaler()),
        ("model", ElasticNet(**C.ELASTICNET_PARAMS)),
    ])


def cv_repeats(X, y, lineage=None, deconfound=False, model_factory=elasticnet_pipeline,
               n_repeats=C.N_REPEATS, n_folds=C.N_FOLDS, seed=C.SEED):
    """
    Repeated k-fold CV, returning the per-repeat Pearson r as an array.

    deconfound=True: within each fold, lineage means for X and y are estimated
    on TRAINING rows only, then subtracted from both train and test; y is also
    divided by the training lineage SD (making the target "sensitivity relative
    to others of the same tissue"). Requires `lineage`.
    """
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    lin = np.asarray(lineage) if lineage is not None else None
    if deconfound and lin is None:
        raise ValueError("deconfound=True requires `lineage`")

    rs = []
    for rep in range(n_repeats):
        kf = KFold(n_folds, shuffle=True, random_state=seed + rep)
        preds, truth = [], []
        for tr, te in kf.split(X_np):
            Xtr, Xte = X_np[tr].copy(), X_np[te].copy()
            ytr, yte = y_np[tr].copy(), y_np[te].copy()

            if deconfound:
                dtr = pd.DataFrame(Xtr)
                dtr["_l"] = lin[tr]
                gm = dtr.groupby("_l").mean()
                ys = pd.Series(ytr).groupby(lin[tr])
                ym, ysd = ys.mean(), ys.std(ddof=1)

                # Test rows whose lineage is unseen in training cannot be
                # centred without borrowing information, so drop them.
                seen = np.isin(lin[te], gm.index)
                if seen.sum() < MIN_TEST_ROWS:
                    continue
                Xte, yte, lte = Xte[seen], yte[seen], lin[te][seen]

                Xtr = Xtr - gm.loc[lin[tr]].to_numpy()
                Xte = Xte - gm.loc[lte].to_numpy()
                ytr = (ytr - ym[lin[tr]].to_numpy()) / ysd[lin[tr]].to_numpy()
                yte = (yte - ym[lte].to_numpy()) / ysd[lte].to_numpy()

                ok = np.isfinite(ytr)
                Xtr, ytr = Xtr[ok], ytr[ok]
                ok2 = np.isfinite(yte)
                Xte, yte = Xte[ok2], yte[ok2]

            m = model_factory().fit(Xtr, ytr)
            preds.append(m.predict(Xte))
            truth.append(yte)

        p, t = np.concatenate(preds), np.concatenate(truth)
        rs.append(stats.pearsonr(t, p)[0])

    return np.array(rs)


def summarise(rs):
    """(mean, lo, hi, sd) from per-repeat correlations; the CI is percentile."""
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return rs.mean(), lo, hi, rs.std()


def repeated_cv(X, y, lineage=None, deconfound=False, name="model",
                model_factory=elasticnet_pipeline,
                n_repeats=C.N_REPEATS, n_folds=C.N_FOLDS, seed=C.SEED, verbose=True):
    """cv_repeats, printed as one line and returned as a result row
    dict(model, r_mean, r_lo, r_hi, r_sd)."""
    rs = cv_repeats(X, y, lineage, deconfound, model_factory, n_repeats, n_folds, seed)
    mean, lo, hi, sd = summarise(rs)
    if verbose:
        print(f"  {name:<44} r={mean:.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"(sd {sd:.3f} over {n_repeats} repeats)")
    return dict(model=name, r_mean=mean, r_lo=lo, r_hi=hi, r_sd=sd)


def oof_predictions(X, y, model_factory=elasticnet_pipeline, n_folds=C.N_FOLDS, seed=C.SEED):
    """
    Single-pass out-of-fold predictions, where a concrete per-row prediction is
    needed (assay-resolution stratification, dashboard display) rather than a
    CI on r. X keeps its dtype (see module docstring).
    """
    X_np = np.asarray(X)
    y_np = np.asarray(y, dtype=float)
    oof = np.full(len(y_np), np.nan)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X_np):
        oof[te] = model_factory().fit(X_np[tr], y_np[tr]).predict(X_np[te])
    return oof


def deconfounded_gene_fit(X, t, lin, keep):
    """(genes, fitted pipeline) for the de-confounded model, fitted once on all
    the rows in `keep`: centre X on its lineage means, z-score the target within
    lineage, fit, and keep the non-zero coefficients sorted by value."""
    gm = pd.DataFrame(np.asarray(X[keep])).groupby(lin[keep]).transform("mean").to_numpy()
    Xd = np.asarray(X[keep]) - gm
    ys = pd.Series(t[keep]).groupby(lin[keep])
    yd = ((t[keep] - ys.transform("mean").to_numpy()) /
          ys.transform(lambda v: v.std(ddof=1)).to_numpy())
    ok = np.isfinite(yd)
    fin = elasticnet_pipeline().fit(Xd[ok], yd[ok])
    sel = np.array(X.columns)[fin.named_steps["select"].get_support()]
    co = fin.named_steps["model"].coef_
    nz = co != 0
    return pd.DataFrame({"gene": sel[nz], "coef": co[nz]}).sort_values("coef"), fin


def permuted_label_check(X, y, print_name, row_name):
    """Leakage check: the pipeline on one seeded permutation of the labels, one
    pass of CV, no de-confounding. A correlation well away from zero means
    something outside the pipeline has seen the labels."""
    y_np = np.asarray(y, dtype=float)
    perm = np.random.default_rng(C.SEED).permutation(y_np)
    cv = KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED)
    pred = cross_val_predict(elasticnet_pipeline(), X, perm, cv=cv)
    r, p = stats.pearsonr(perm, pred)
    print(f"  PERMUTED LABELS, {print_name}    r={r:+.3f}  p={p:.3f}  "
          f"(expect ~0 -- confirms no leakage into the pipeline)")
    return dict(model=f"PERMUTED LABELS ({row_name})", r_mean=r, r_lo=np.nan,
                r_hi=np.nan, r_sd=np.nan)


def ridge_cv_predict(X, y, categorical=(), cv=None):
    """Cross-validated Ridge predictions for the baselines.

    Scaling and encoding live INSIDE the Pipeline so they refit on the training
    part of each fold only; doing them before cross_val_predict would leak test
    information and inflate every baseline.
    """
    if cv is None:
        cv = KFold(n_splits=C.N_FOLDS, shuffle=True, random_state=C.SEED)
    num = [c for c in X.columns if c not in categorical]
    steps = []
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=5),
                      list(categorical)))
    if num:
        steps.append(("num", StandardScaler(), num))
    pipe = Pipeline([("prep", ColumnTransformer(steps)), ("model", Ridge(alpha=1.0))])
    return cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)


def _ridge_modules_cv(F, t, n_repeats, n_folds, seed):
    """Repeated CV of a scaled Ridge on a small feature table; per-repeat r."""
    rs = []
    for rep in range(n_repeats):
        kf = KFold(n_folds, shuffle=True, random_state=seed + rep)
        pp, tt = [], []
        for tr, te in kf.split(F):
            mm = Pipeline([("sc", StandardScaler()), ("m", Ridge(1.0))]).fit(F.iloc[tr], t[tr])
            pp.append(mm.predict(F.iloc[te]))
            tt.append(t[te])
        rs.append(stats.pearsonr(np.concatenate(tt), np.concatenate(pp))[0])
    return np.array(rs)


def transfer_to_crc(X, y, s, crc_mask, modules, n_repeats=C.N_REPEATS,
                    n_folds=C.N_FOLDS, seed=C.SEED):
    """
    The three-way CRC comparison: (a) train on CRC only, cross-validated, the
    naive approach; (b) transfer: fit the transcriptome model on every solid
    line EXCEPT CRC and predict CRC directly; (c) modules plus the transfer
    prediction as one extra feature, cross-validated on CRC.

    CRC (n~43-46) is far too small to support a transcriptome-wide model
    trained on its own; transfer consistently outperforms local training.
    """
    t = np.asarray(y, dtype=float)
    tc = t[crc_mask]
    results = []

    def row(model, rs):
        lo, hi = np.percentile(rs, 2.5), np.percentile(rs, 97.5)
        return dict(model=model, r_mean=rs.mean(), r_lo=lo, r_hi=hi, r_sd=rs.std())

    rs = _ridge_modules_cv(s[modules][crc_mask], tc, n_repeats, n_folds, seed)
    results.append(row("CRC-only modules", rs))
    print(f"  {'(a) modules trained on CRC only':<44} r={rs.mean():+.3f}  "
          f"[{results[-1]['r_lo']:+.3f}, {results[-1]['r_hi']:+.3f}]")

    pan = elasticnet_pipeline().fit(X[~crc_mask], t[~crc_mask])
    pred = pan.predict(X[crc_mask])
    r, p = stats.pearsonr(tc, pred)
    bs = bootstrap_pearson_draws(tc, pred)
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"  {'(b) pan-solid model applied to CRC':<44} r={r:+.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    results.append(dict(model="transfer pan-solid -> CRC", r_mean=r,
                        r_lo=lo, r_hi=hi, r_sd=np.std(bs)))

    F = s[modules][crc_mask].copy()
    F["pan_pred"] = pred
    rs = _ridge_modules_cv(F, tc, n_repeats, n_folds, seed)
    results.append(row("CRC modules + transfer feature", rs))
    print(f"  {'(c) modules + transfer prediction':<44} r={rs.mean():+.3f}  "
          f"[{results[-1]['r_lo']:+.3f}, {results[-1]['r_hi']:+.3f}]")

    return results, pred
