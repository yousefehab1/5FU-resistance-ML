"""The shared model pipeline and the repeated-CV loop.

model() is SelectKBest -> StandardScaler -> ElasticNet with the settings in
config/analysis.yaml. repeated_cv() returns the per-repeat correlations and
prints nothing; callers format their own output.

06 and 07 use gene_pipeline() with their own estimators. Their permuted-label
settings differ on purpose to keep published numbers (docs/LIMITATIONS.md, item 6).
"""

from __future__ import annotations

from typing import Callable

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.compose import ColumnTransformer
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .config import ELASTICNET, K_GENES, N_FOLDS, N_REPEATS, SEED

# The deconfounding branch drops test rows whose lineage was not seen in
# training, because centring them would have to borrow information from the
# test set. A fold left with fewer than this many usable rows is skipped
# rather than scored on noise. Was a bare `5` in both 08 and 13.
MIN_TEST_ROWS = 5


def gene_pipeline(model, k_genes: int = K_GENES) -> Pipeline:
    """
    Select k genes, standardise, fit `model`.

    Selection sits INSIDE the pipeline on purpose. Selecting genes before
    cross-validation is the leakage bug this project keeps testing for with
    permuted labels: it lets the test fold influence which features exist.

    The estimator is injected because 06 and 07 put five different ones through
    this same shape: the tuned ElasticNet, Ridge, a RandomForest comparator,
    and the two permuted-label arms. Leaving it open is what let them share the
    shape without their hyperparameters being centralised along with it.
    """
    return Pipeline([("select", SelectKBest(f_regression, k=k_genes)),
                     ("scale", StandardScaler()),
                     ("model", model)])


def elasticnet_pipeline(k_genes: int = K_GENES, seed: int = SEED) -> Pipeline:
    """The project's standard pipeline: `gene_pipeline` around the ElasticNet
    whose settings live in config/analysis.yaml."""
    return gene_pipeline(ElasticNet(random_state=seed, **ELASTICNET), k_genes)


def repeated_cv(
    X,
    y,
    lineage,
    deconfound: bool,
    *,
    model_factory: Callable[[], Pipeline] = elasticnet_pipeline,
    n_repeats: int = N_REPEATS,
    n_folds: int = N_FOLDS,
    seed: int = SEED,
) -> np.ndarray:
    """
    Repeated k-fold CV, returning the per-repeat Pearson r.

    deconfound=True: within each fold, lineage means for X and y are estimated
    on TRAINING rows only, then subtracted from both train and test. y is also
    divided by the training lineage SD, making the target "sensitivity relative
    to others of the same tissue".

    Prints nothing and formats nothing. Callers differ in how they report and
    in what they pack the result into, and keeping that at the call site is
    what let 08's and 13's copies be merged without their output moving.
    """
    X_np, y_np = np.asarray(X, dtype=float), np.asarray(y, dtype=float)
    lin = np.asarray(lineage)
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
                # centred without borrowing information -- drop them.
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


def summarise(rs: np.ndarray) -> tuple[float, float, float, float]:
    """(mean, lo, hi, sd) from per-repeat correlations, as both callers did."""
    lo, hi = np.percentile(rs, [2.5, 97.5])
    return rs.mean(), lo, hi, rs.std()


def report_cv(X, y, lineage, deconfound: bool, name: str) -> dict:
    """Repeated CV, printed in 08's format, returned as 08's result row.

    This is presentation, and presentation is normally the call site's business
    (see the note above about how 08's and 13's copies were merged). It lives
    here because two scripts need this exact line: 08 prints it, and 21 reuses
    it so its TF-activity arms are directly comparable to 08's baselines.
    21 previously got it by importing 08 through an importlib shim.

    13 formats differently and keeps its own wrapper.
    """
    rs = repeated_cv(X, y, lineage, deconfound)
    mean, lo, hi, sd = summarise(rs)
    print(f"  {name:<44} r={mean:.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"(sd {sd:.3f} over {N_REPEATS} repeats)")
    return dict(model=name, r_mean=mean, r_lo=lo, r_hi=hi, r_sd=sd)


def deconfounded_gene_fit(X, t, lin, keep) -> tuple[pd.DataFrame, Pipeline]:
    """(genes, fitted pipeline) for the de-confounded model, fitted once on all the data.

    08 and 13 each carried this block, identical but for indentation: centre
    X on its lineage means, z-score the target within lineage, fit the shared
    pipeline, and keep the non-zero coefficients sorted by sign and size.
    What each script does with the table stays at its call site: 08 writes it
    at once, `scripts/08_final_model.py`
    (`g.to_csv(PROC / "final_model_genes.csv", index=False)`), and 13 collects
    one per drug to write later, `scripts/13_multidrug_models.py`
    (`all_genes[drug] = g`). So does the line reporting how many genes
    survived, which differs between them by a trailing newline. 08 also saves
    the fitted pipeline to models/final_deconfounded.joblib, which is why it is
    returned; 13 discards it.

    `np.asarray(X[keep])` keeps X's dtype. For the GDSC matrices that is
    float32, which makes this fit sensitive to the scikit-learn build. Upcasting
    here would change the results.
    """
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


def permuted_label_check(X, y, print_name: str, row_name: str) -> dict:
    """Leakage check: the shared pipeline on shuffled labels.

    One seeded permutation, one pass of fold-matched CV, no deconfounding.
    It should find nothing; a correlation well away from zero means something
    outside the pipeline has seen the labels.

    Two names because 20 printed "methylation model" but labelled its result
    row "methylation", and both strings are in its outputs. 21 used one name
    for both and passes it twice.

    13's permuted-label check is not this one. It draws from a shared generator and runs
    the full repeated CV with deconfounding, and it stays in 13.
    """
    y_np = np.asarray(y, dtype=float)
    perm = np.random.default_rng(SEED).permutation(y_np)
    cv = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    pred = cross_val_predict(elasticnet_pipeline(), X, perm, cv=cv)
    r, p = stats.pearsonr(perm, pred)
    print(f"  PERMUTED LABELS, {print_name}    r={r:+.3f}  p={p:.3f}  "
          f"(expect ~0 -- confirms no leakage into the pipeline)")
    return dict(model=f"PERMUTED LABELS ({row_name})", r_mean=r, r_lo=np.nan,
                r_hi=np.nan, r_sd=np.nan)


def ridge_cv_predict(X, y, categorical=(), cv=None) -> np.ndarray:
    """Cross-validated predictions from the Ridge baselines (03 and 13).

    EVERYTHING that learns from the data lives inside the Pipeline, so it is
    refitted on the training part of each fold only. Scaling or encoding before
    cross_val_predict would leak test information into training and silently
    inflate every number the baselines report.

    `cv` defaults to the single seeded KFold both scripts built at module
    level. KFold with a fixed random_state holds no state between calls, so
    building it here gives the same folds on every call.

    `n_jobs=1` is 13's spelling; 03 left it at the default, which is also one
    process when no joblib backend is configured, and none is.
    """
    if cv is None:
        cv = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
    num = [c for c in X.columns if c not in categorical]
    steps = []
    if categorical:
        steps.append(("cat", OneHotEncoder(handle_unknown="ignore",
                                           min_frequency=5), list(categorical)))
    if num:
        steps.append(("num", StandardScaler(), num))
    pipe = Pipeline([("prep", ColumnTransformer(steps)), ("model", Ridge(alpha=1.0))])
    return cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)
