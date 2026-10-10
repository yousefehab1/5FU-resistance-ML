"""
lib/modeling.py
================

The model pipeline and the cross-validation logic every modelling stage shares.

  elasticnet_pipeline()     SelectKBest -> StandardScaler -> ElasticNet, alpha tuned
  fixed_elasticnet_pipeline()  the same with alpha fixed at config.ELASTICNET_PARAMS
  scaled_ridge()            StandardScaler -> RidgeCV, for small feature tables
  scaled_target()           a model factory refit with the target rescaled to a fixed SD
  cv_oof()                  repeated k-fold CV, every out-of-fold prediction kept
  per_repeat()              per-repeat Pearson r, Spearman rho or C-index
  cv_repeats()              cv_oof reduced to per-repeat Pearson r
  summarise()               a result row: r over repeats, cell-line bootstrap CI
  repeated_cv()             cv_oof + one printed line + a result row
  compare_models()          two models on the same splits, corrected t-test
  cv_leave_lineage_out()    every lineage predicted by a model that never saw it
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

NESTED TUNING
-------------
elasticnet_pipeline() chooses alpha by an inner config.INNER_FOLDS-fold CV
on the rows it is fitted on. Inside cv_oof those are an outer training
fold, so the tuning never sees the lines it is scored on and the reported
r stays an outer-CV estimate. cv_oof records each fold's alpha.

DTYPE
-----
The GDSC matrices are float32. Functions that take a DataFrame fit on it as
given; upcasting to float64 first moves published numbers in the 7th decimal.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet, ElasticNetCV, Ridge, RidgeCV
from sklearn.model_selection import KFold, LeaveOneGroupOut, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

import config as C
from lib.stats import (bootstrap_lines_r, bootstrap_pearson_draws, concordance_index,
                       corrected_resampled_ttest)

# A de-confounded fold left with fewer usable test rows than this is skipped
# rather than scored on noise.
MIN_TEST_ROWS = 5


class TunedElasticNet(RegressorMixin, BaseEstimator):
    """ElasticNetCV whose alpha grid depends on the table's width: tables of
    at most config.FEW_FEATURES columns get config.ELASTICNET_ALPHAS_FEW,
    wider ones config.ELASTICNET_ALPHAS. Exposes alpha_, alphas_, coef_ and
    intercept_ like the ElasticNetCV it wraps."""

    def fit(self, X, y):
        p = C.ELASTICNET_PARAMS
        few = np.shape(X)[1] <= C.FEW_FEATURES
        grid = C.ELASTICNET_ALPHAS_FEW if few else C.ELASTICNET_ALPHAS
        self.model_ = ElasticNetCV(
            l1_ratio=p["l1_ratio"], alphas=list(grid),
            cv=KFold(C.INNER_FOLDS, shuffle=True, random_state=C.SEED),
            max_iter=p["max_iter"], random_state=p["random_state"], n_jobs=-1).fit(X, y)
        for a in ("alpha_", "alphas_", "coef_", "intercept_"):
            setattr(self, a, getattr(self.model_, a))
        return self

    def predict(self, X):
        return self.model_.predict(X)


def elasticnet_pipeline(tuned=True):
    """The canonical model: SelectKBest(f_regression, k) -> scale -> ElasticNet.

    SelectKBest lives INSIDE the pipeline so it is refit on the training fold
    only. Selecting genes by correlation with the full target before splitting
    would leak.

    tuned=True (the default) picks alpha by an inner CV on the training rows
    (TunedElasticNet; see NESTED TUNING above); tuned=False is the fixed-alpha
    reference.
    """
    model = TunedElasticNet() if tuned else ElasticNet(**C.ELASTICNET_PARAMS)
    return Pipeline([
        ("select", SelectKBest(f_regression, k=C.K_GENES)),
        ("scale", StandardScaler()),
        ("model", model),
    ])


def fixed_elasticnet_pipeline():
    """The pipeline with alpha fixed at config.ELASTICNET_PARAMS, the model
    every result used before nested tuning; kept as the reference."""
    return elasticnet_pipeline(tuned=False)


def scaled_ridge():
    """StandardScaler -> RidgeCV over config.RIDGE_ALPHAS (efficient leave-one-
    out), for the small feature tables: baselines and CRC-only modules."""
    return Pipeline([("sc", StandardScaler()), ("m", RidgeCV(alphas=list(C.RIDGE_ALPHAS)))])


def chosen_alpha(model):
    """(alpha, at_grid_edge) of a fitted model that tuned its alpha, else
    (nan, nan). Unwraps pipelines and TransformedTargetRegressor."""
    while True:
        if isinstance(model, Pipeline):
            model = model.steps[-1][1]
        elif isinstance(model, TransformedTargetRegressor):
            model = model.regressor_
        else:
            break
    if isinstance(model, (TunedElasticNet, ElasticNetCV)):
        grid = np.asarray(model.alphas_)
    elif isinstance(model, RidgeCV):
        grid = np.asarray(model.alphas)
    else:
        return np.nan, np.nan
    a = float(model.alpha_)
    return a, float(np.isclose(a, grid.min()) or np.isclose(a, grid.max()))


def _times(v, k):
    return v * k


def _over(v, k):
    return v / k


def scaled_target(sd, model_factory=fixed_elasticnet_pipeline):
    """A model factory whose fits see the target z-scored on the training rows
    and multiplied by `sd`; predictions come back on the original scale.

    ElasticNet's penalty is in target units, so the same alpha shrinks a
    narrow-range target harder than a wide one. Fitting every target at one
    fixed SD makes the penalty mean the same thing across screens and drugs.
    Used by the penalty-scale diagnostic in stages/03_fu_confounds.py.
    """
    def factory():
        tf = Pipeline([("z", StandardScaler()),
                       ("sd", FunctionTransformer(_times, inverse_func=_over,
                                                  kw_args={"k": sd}, inv_kw_args={"k": sd},
                                                  check_inverse=False))])
        return TransformedTargetRegressor(regressor=model_factory(), transformer=tf,
                                          check_inverse=False)
    return factory


def cv_oof(X, y, lineage=None, deconfound=False, model_factory=elasticnet_pipeline,
           n_repeats=C.N_REPEATS, n_folds=C.N_FOLDS, seed=C.SEED):
    """
    Repeated k-fold CV, keeping every out-of-fold prediction.

    Returns dict(truth, pred, fold), each (n_lines, n_repeats), and alpha,
    (n_folds, n_repeats). `truth` is the target as it was scored
    (de-confounded targets differ per repeat, because the lineage means come
    from that repeat's training folds). Lines not scored in a repeat are NaN
    in truth and pred and -1 in fold. alpha is each fit's chosen alpha, and
    alpha_edge 1 where it sat on the grid's edge (both NaN for untuned
    models and skipped folds).

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

    n = len(y_np)
    truth = np.full((n, n_repeats), np.nan)
    pred = np.full((n, n_repeats), np.nan)
    fold = np.full((n, n_repeats), -1)
    alpha = np.full((n_folds, n_repeats), np.nan)
    edge = np.full((n_folds, n_repeats), np.nan)
    for rep in range(n_repeats):
        kf = KFold(n_folds, shuffle=True, random_state=seed + rep)
        for k, (tr, te) in enumerate(kf.split(X_np)):
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
                Xte, yte, lte, te = Xte[seen], yte[seen], lin[te][seen], te[seen]

                Xtr = Xtr - gm.loc[lin[tr]].to_numpy()
                Xte = Xte - gm.loc[lte].to_numpy()
                ytr = (ytr - ym[lin[tr]].to_numpy()) / ysd[lin[tr]].to_numpy()
                yte = (yte - ym[lte].to_numpy()) / ysd[lte].to_numpy()

                ok = np.isfinite(ytr)
                Xtr, ytr = Xtr[ok], ytr[ok]
                ok2 = np.isfinite(yte)
                Xte, yte, te = Xte[ok2], yte[ok2], te[ok2]

            m = model_factory().fit(Xtr, ytr)
            pred[te, rep] = m.predict(Xte)
            truth[te, rep] = yte
            fold[te, rep] = k
            alpha[k, rep], edge[k, rep] = chosen_alpha(m)

    return dict(truth=truth, pred=pred, fold=fold, alpha=alpha, alpha_edge=edge)


def _scored(o, rep):
    """Row order for one repeat: by fold, then by line. KFold returns sorted
    test indices, so this is the order the predictions were produced in, and
    scoring in it keeps r identical to the bit with the original loop."""
    f = o["fold"][:, rep]
    rows = np.flatnonzero(f >= 0)
    return rows[np.argsort(f[rows], kind="stable")]


def per_repeat(o, metric="pearson"):
    """Per-repeat Pearson r, Spearman rho or Harrell's C from a cv_oof result."""
    out = []
    for rep in range(o["truth"].shape[1]):
        i = _scored(o, rep)
        t, p = o["truth"][i, rep], o["pred"][i, rep]
        if metric == "pearson":
            out.append(stats.pearsonr(t, p)[0])
        elif metric == "spearman":
            out.append(stats.spearmanr(t, p)[0])
        else:
            out.append(concordance_index(t, p))
    return np.array(out)


def cv_repeats(X, y, lineage=None, deconfound=False, model_factory=elasticnet_pipeline,
               n_repeats=C.N_REPEATS, n_folds=C.N_FOLDS, seed=C.SEED):
    """Repeated k-fold CV, returning the per-repeat Pearson r as an array.
    See cv_oof for the de-confounding."""
    return per_repeat(cv_oof(X, y, lineage, deconfound, model_factory, n_repeats, n_folds, seed))


def summarise(o, name="model"):
    """A result row from a cv_oof result: dict(model, r_mean, r_lo, r_hi, r_sd,
    rho_mean, cindex_mean, alpha_median, alpha_edge_frac). r_mean and r_sd
    are over repeats; the interval is the 2.5th and 97.5th percentile of a
    cell-line bootstrap (lib.stats.bootstrap_lines_r). alpha_median and
    alpha_edge_frac summarise the tuned alphas over folds (NaN if untuned)."""
    rs = per_repeat(o)
    lo, hi = np.percentile(bootstrap_lines_r(o["truth"], o["pred"]), [2.5, 97.5])
    a, e = o.get("alpha"), o.get("alpha_edge")
    tuned = a is not None and np.isfinite(a).any()
    return dict(model=name, r_mean=rs.mean(), r_lo=lo, r_hi=hi, r_sd=rs.std(),
                rho_mean=per_repeat(o, "spearman").mean(),
                cindex_mean=per_repeat(o, "cindex").mean(),
                alpha_median=np.nanmedian(a) if tuned else np.nan,
                alpha_edge_frac=np.nanmean(e) if tuned else np.nan)


def repeated_cv(X, y, lineage=None, deconfound=False, name="model",
                model_factory=elasticnet_pipeline,
                n_repeats=C.N_REPEATS, n_folds=C.N_FOLDS, seed=C.SEED, verbose=True,
                return_oof=False):
    """cv_oof, printed as one line and returned as a summarise() row; with
    return_oof=True, (row, cv_oof result) so models can be compared."""
    o = cv_oof(X, y, lineage, deconfound, model_factory, n_repeats, n_folds, seed)
    row = summarise(o, name)
    if verbose:
        print(f"  {name:<44} r={row['r_mean']:.3f}  [{row['r_lo']:.3f}, {row['r_hi']:.3f}]  "
              f"(sd {row['r_sd']:.3f} over {n_repeats} repeats)  "
              f"rho={row['rho_mean']:.3f}  C={row['cindex_mean']:.3f}")
    return (row, o) if return_oof else row


def compare_models(oa, ob, name_a, name_b):
    """
    Two models scored on the SAME lines and splits: per-split Pearson r
    differences (a - b), tested with the corrected resampled t-test. Returns
    a result row. Raises if the two runs did not score the same targets.
    """
    if not (np.array_equal(oa["fold"], ob["fold"])
            and np.allclose(oa["truth"], ob["truth"], equal_nan=True)):
        raise ValueError(f"{name_a} and {name_b} were not scored on the same splits")
    diffs, n_test, n_used = [], [], []
    for rep in range(oa["fold"].shape[1]):
        f = oa["fold"][:, rep]
        n_used.append((f >= 0).sum())
        for k in np.unique(f[f >= 0]):
            i = f == k
            t = oa["truth"][i, rep]
            diffs.append(stats.pearsonr(t, oa["pred"][i, rep])[0]
                         - stats.pearsonr(t, ob["pred"][i, rep])[0])
            n_test.append(i.sum())
    n_te = float(np.mean(n_test))
    mean, t, p = corrected_resampled_ttest(diffs, np.mean(n_used) - n_te, n_te)
    return dict(model_a=name_a, model_b=name_b, n_splits=len(diffs),
                r_a=per_repeat(oa).mean(), r_b=per_repeat(ob).mean(),
                mean_fold_r_diff=mean, t=t, p=p)


def cv_leave_lineage_out(X, y, lineage, name, model_factory=elasticnet_pipeline):
    """
    Leave-one-lineage-out CV: every tissue is predicted by a model that never
    saw a line of it. Returns a summarise()-style row plus:
      r_within_median     median within-lineage r, over held-out lineages
                          with at least config.MIN_LINEAGE_N lines
      n_lineages_within   how many lineages that median is over
    r_mean is the pooled r over all lines, so it still rewards getting a new
    tissue's average right; r_within_median does not.
    """
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    lin = np.asarray(lineage)
    pred = np.full(len(y_np), np.nan)
    alphas = []
    for tr, te in LeaveOneGroupOut().split(X_np, groups=lin):
        m = model_factory().fit(X_np[tr], y_np[tr])
        pred[te] = m.predict(X_np[te])
        alphas.append(chosen_alpha(m))
    a, e = np.array(alphas).T
    lo, hi = np.percentile(bootstrap_lines_r(y_np, pred), [2.5, 97.5])
    within = [stats.pearsonr(y_np[lin == g], pred[lin == g])[0]
              for g, c in pd.Series(lin).value_counts().items() if c >= C.MIN_LINEAGE_N]
    row = dict(model=name, r_mean=stats.pearsonr(y_np, pred)[0], r_lo=lo, r_hi=hi,
               r_sd=np.nan, rho_mean=stats.spearmanr(y_np, pred)[0],
               cindex_mean=concordance_index(y_np, pred),
               alpha_median=np.nanmedian(a) if np.isfinite(a).any() else np.nan,
               alpha_edge_frac=np.nanmean(e) if np.isfinite(e).any() else np.nan,
               r_within_median=float(np.median(within)) if within else np.nan,
               n_lineages_within=len(within))
    print(f"  {name:<44} r={row['r_mean']:.3f}  [{lo:.3f}, {hi:.3f}]  "
          f"within-lineage median r={row['r_within_median']:+.3f} "
          f"over {len(within)} lineages")
    return row


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
    pipe = Pipeline([("prep", ColumnTransformer(steps)),
                     ("model", RidgeCV(alphas=list(C.RIDGE_ALPHAS)))])
    return cross_val_predict(pipe, X, y, cv=cv, n_jobs=1)


def _ridge_modules_cv(F, t, n_repeats, n_folds, seed):
    """Repeated CV of a scaled Ridge on a small feature table; a cv_oof result."""
    return cv_oof(F, t, model_factory=scaled_ridge,
                  n_repeats=n_repeats, n_folds=n_folds, seed=seed)


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

    o = _ridge_modules_cv(s[modules][crc_mask], tc, n_repeats, n_folds, seed)
    results.append(summarise(o, "CRC-only modules"))
    print(f"  {'(a) modules trained on CRC only':<44} r={results[-1]['r_mean']:+.3f}  "
          f"[{results[-1]['r_lo']:+.3f}, {results[-1]['r_hi']:+.3f}]")

    pan = elasticnet_pipeline().fit(X[~crc_mask], t[~crc_mask])
    pred = pan.predict(X[crc_mask])
    r, p = stats.pearsonr(tc, pred)
    bs = bootstrap_pearson_draws(tc, pred)
    lo, hi = np.percentile(bs, 2.5), np.percentile(bs, 97.5)
    print(f"  {'(b) pan-solid model applied to CRC':<44} r={r:+.3f}  "
          f"[{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")
    a_pan, e_pan = chosen_alpha(pan)
    results.append(dict(model="transfer pan-solid -> CRC", r_mean=r,
                        r_lo=lo, r_hi=hi, r_sd=np.std(bs),
                        alpha_median=a_pan, alpha_edge_frac=e_pan,
                        rho_mean=stats.spearmanr(tc, pred)[0],
                        cindex_mean=concordance_index(tc, pred)))

    F = s[modules][crc_mask].copy()
    F["pan_pred"] = pred
    o = _ridge_modules_cv(F, tc, n_repeats, n_folds, seed)
    results.append(summarise(o, "CRC modules + transfer feature"))
    print(f"  {'(c) modules + transfer prediction':<44} r={results[-1]['r_mean']:+.3f}  "
          f"[{results[-1]['r_lo']:+.3f}, {results[-1]['r_hi']:+.3f}]")

    return results, pred
