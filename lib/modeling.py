"""
lib/modeling.py
================

The model pipeline factory and cross-validation logic, consolidating what
was six independent copies of the same `Pipeline([SelectKBest, StandardScaler,
ElasticNet])` construction, and porting the fold-internal lineage
de-confounding logic that is the single most safety-critical piece of code
in this project.

WHY repeated_cv IS HAND-ROLLED RATHER THAN A SKLEARN PIPELINE
---------------------------------------------------------------
De-confounding the target means subtracting lineage means of y. Computing
those means over the whole dataset would use test-fold targets -- a genuine
leak. An earlier version of this project's model did exactly that and
reported r=0.279 for the de-confounded association; the corrected,
fold-internal version reports r=0.182. That gap IS the leak.

So the fold loop is written out explicitly: within each fold, lineage means
for BOTH X and y are estimated on the TRAINING rows only, then applied to the
held-out rows. sklearn's cross_val_predict cannot express a target
transformation, which is why this is a manual loop rather than a Pipeline.
Lineages present in test but not train are dropped from that fold, since
their mean cannot be estimated without seeing them.
"""

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.linear_model import ElasticNet
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

import config as C


def elasticnet_pipeline():
    """The canonical model: SelectKBest(f_regression, k) -> scale -> ElasticNet.

    SelectKBest lives INSIDE the pipeline, not as a pre-filtering step, so it
    is refit on the training fold only every time -- this is the single most
    important structural choice in the model, since selecting genes by
    correlation with the full target before splitting would leak.
    """
    return Pipeline([
        ("select", SelectKBest(f_regression, k=C.K_GENES)),
        ("scale", StandardScaler()),
        ("model", ElasticNet(**C.ELASTICNET_PARAMS)),
    ])


def repeated_cv(X, y, lineage=None, deconfound=False, name="model",
                 model_factory=elasticnet_pipeline,
                 n_repeats=C.N_REPEATS, n_folds=C.N_FOLDS, seed=C.SEED, verbose=True):
    """
    Repeated k-fold CV returning per-repeat Pearson r as a distribution
    (mean + 95% CI via percentiles across repeats, not a single split).

    deconfound=True: within each fold, lineage means for X and y are
    estimated on TRAINING rows only, then subtracted from both train and
    test; y is also divided by the training lineage SD (making the target
    "sensitivity relative to others of the same tissue"). Requires `lineage`.
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
                # centred without borrowing information -- drop them.
                seen = np.isin(lin[te], gm.index)
                if seen.sum() < 5:
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

    rs = np.array(rs)
    lo, hi = np.percentile(rs, [2.5, 97.5])
    if verbose:
        print(f"  {name:<44} r={rs.mean():.3f}  [{lo:.3f}, {hi:.3f}]  "
              f"(sd {rs.std():.3f} over {n_repeats} repeats)")
    return dict(model=name, r_mean=rs.mean(), r_lo=lo, r_hi=hi, r_sd=rs.std())


def oof_predictions(X, y, model_factory=elasticnet_pipeline, n_folds=C.N_FOLDS, seed=C.SEED):
    """
    Single-pass out-of-fold predictions (not repeated -- used where a
    concrete per-row prediction is needed, e.g. dashboard display or
    assay-resolution stratification, as opposed to a CI on r).
    """
    X_np = np.asarray(X, dtype=float)
    y_np = np.asarray(y, dtype=float)
    oof = np.full(len(y_np), np.nan)
    for tr, te in KFold(n_folds, shuffle=True, random_state=seed).split(X_np):
        oof[te] = model_factory().fit(X_np[tr], y_np[tr]).predict(X_np[te])
    return oof


def transfer_to_crc(X, y, s, crc_mask, modules, n_repeats=C.N_REPEATS,
                     n_folds=C.N_FOLDS, seed=C.SEED):
    """
    The three-way CRC comparison: (a) train on CRC only, cross-validated --
    the naive approach; (b) transfer -- fit the transcriptome model on every
    solid line EXCEPT CRC, predict CRC directly; (c) modules + the transfer
    prediction as one extra feature, cross-validated on CRC.

    CRC (n~43-46) is far too small to support a transcriptome-wide model
    trained on its own; transfer consistently outperforms local training.
    """
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import Pipeline as SkPipeline
    from sklearn.preprocessing import StandardScaler as SkScaler

    t = np.asarray(y, dtype=float)
    results = []

    # (a) train on CRC only, cross-validated
    rs = []
    Xc = s[modules][crc_mask]
    for rep in range(n_repeats):
        kf = KFold(n_folds, shuffle=True, random_state=seed + rep)
        p, tt = [], []
        for tr, te in kf.split(Xc):
            mm = SkPipeline([("sc", SkScaler()), ("m", Ridge(1.0))]).fit(
                Xc.iloc[tr], t[crc_mask][tr])
            p.append(mm.predict(Xc.iloc[te])); tt.append(t[crc_mask][te])
        rs.append(stats.pearsonr(np.concatenate(tt), np.concatenate(p))[0])
    rs = np.array(rs)
    print(f"  {'(a) modules trained on CRC only':<44} r={rs.mean():+.3f}  "
          f"[{np.percentile(rs, 2.5):+.3f}, {np.percentile(rs, 97.5):+.3f}]")
    results.append(dict(model="CRC-only modules", r_mean=rs.mean(),
                         r_lo=np.percentile(rs, 2.5), r_hi=np.percentile(rs, 97.5),
                         r_sd=rs.std()))

    # (b) transfer: fit on every solid line EXCEPT CRC, predict CRC
    pan = elasticnet_pipeline().fit(X[~crc_mask], t[~crc_mask])
    pred = pan.predict(X[crc_mask])
    r, p = stats.pearsonr(t[crc_mask], pred)
    bs = [stats.pearsonr(t[crc_mask][i], pred[i])[0]
          for i in np.random.default_rng(seed).integers(0, crc_mask.sum(), (500, crc_mask.sum()))]
    print(f"  {'(b) pan-solid model applied to CRC':<44} r={r:+.3f}  "
          f"[{np.percentile(bs, 2.5):+.3f}, {np.percentile(bs, 97.5):+.3f}]  p={p:.3f}")
    results.append(dict(model="transfer pan-solid -> CRC", r_mean=r,
                         r_lo=np.percentile(bs, 2.5), r_hi=np.percentile(bs, 97.5),
                         r_sd=np.std(bs)))

    # (c) transfer prediction as a single feature alongside the modules
    F = s[modules][crc_mask].copy()
    F["pan_pred"] = pred
    rs = []
    for rep in range(n_repeats):
        kf = KFold(n_folds, shuffle=True, random_state=seed + rep)
        pp, tt = [], []
        for tr, te in kf.split(F):
            mm = SkPipeline([("sc", SkScaler()), ("m", Ridge(1.0))]).fit(
                F.iloc[tr], t[crc_mask][tr])
            pp.append(mm.predict(F.iloc[te])); tt.append(t[crc_mask][te])
        rs.append(stats.pearsonr(np.concatenate(tt), np.concatenate(pp))[0])
    rs = np.array(rs)
    print(f"  {'(c) modules + transfer prediction':<44} r={rs.mean():+.3f}  "
          f"[{np.percentile(rs, 2.5):+.3f}, {np.percentile(rs, 97.5):+.3f}]")
    results.append(dict(model="CRC modules + transfer feature", r_mean=rs.mean(),
                         r_lo=np.percentile(rs, 2.5), r_hi=np.percentile(rs, 97.5),
                         r_sd=rs.std()))

    return results, pred
