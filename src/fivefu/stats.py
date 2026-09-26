"""Resampling and correlation helpers.

The bootstrap draws its indices in a fixed way: a fresh default_rng(seed) per
call and one integers(0, n, (n_boot, n)) matrix. Drawing them any other way,
even an equivalent one, moves every published confidence interval.
bootstrap_r_difference draws pairs one at a time from a single generator with
2000 draws; that order is also fixed for the same reason.
"""

from __future__ import annotations

import numpy as np
from numpy.linalg import lstsq
from scipy import stats
from scipy.stats import hypergeom

from .config import SEED

N_BOOT = 500

# 14_drug_specificity.py resamples 2000 times where everything else resamples
# 500. That is a deliberate difference, not drift: it is estimating a difference
# of two correlations, which is noisier than either one, and it checks its own
# stability by re-running at SEED + 1 and requiring the bounds to agree.
N_BOOT_DIFF = 2000


def bootstrap_pearson_draws(x, y, n_boot: int = N_BOOT, seed: int = SEED) -> np.ndarray:
    """
    Per-resample Pearson r.

    `x` and `y` must be numpy arrays, not Series: the indexing below is
    positional, and pandas would resolve an integer array against the index
    instead. Every current call site already passes arrays.
    """
    n = len(x)
    rng = np.random.default_rng(seed)
    return np.array([stats.pearsonr(x[i], y[i])[0]
                     for i in rng.integers(0, n, (n_boot, n))])


def bootstrap_r(x, y, n_boot: int = N_BOOT, seed: int = SEED):
    """Return (r, p, lo, hi)."""
    r, p = stats.pearsonr(x, y)
    bs = bootstrap_pearson_draws(x, y, n_boot=n_boot, seed=seed)
    return r, p, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bootstrap_r_difference(x1, y1, x2, y2, n_boot: int = N_BOOT_DIFF,
                           seed: int = SEED) -> np.ndarray:
    """Per-resample (r1 - r2) for two independent, unequally sized samples.

    From `14_drug_specificity.py`, comparing 5-FU's correlation with
    oxaliplatin against its correlation with cisplatin. The two samples cover
    different cell lines and differ in size, so they are resampled
    independently.

    ORDER IS LOAD-BEARING. Both index draws come from one generator, alternating
    within the loop. Drawing sample 1's indices for all `n_boot` iterations and
    then sample 2's would be the same estimator and a different sequence, and
    the published interval would move.
    """
    x1, y1 = np.asarray(x1, float), np.asarray(y1, float)
    x2, y2 = np.asarray(x2, float), np.asarray(y2, float)
    rng = np.random.default_rng(seed)
    n1, n2 = len(x1), len(x2)
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        i1 = rng.integers(0, n1, n1)
        i2 = rng.integers(0, n2, n2)
        diffs[i] = (stats.pearsonr(x1[i1], y1[i1])[0]
                    - stats.pearsonr(x2[i2], y2[i2])[0])
    return diffs


def design(n: int, controls) -> np.ndarray:
    """`controls` as an OLS design matrix with an intercept column first.

    The intercept is added here rather than by callers because every call site
    in this project wants one, and one that forgot would silently residualise
    against an uncentred design.
    """
    return np.column_stack([np.ones(n)] + [np.asarray(z, float) for z in controls])


def residualize(y, controls) -> np.ndarray:
    """What is left of `y` once `controls` are regressed out by OLS.

    Three call sites had this written out: both halves of `partial_corr`, and
    `resid_auc` in `14_drug_specificity.py`, which removes a general
    chemosensitivity score from a drug's AUC before comparing drugs.
    """
    y = np.asarray(y, float)
    Z = design(len(y), controls)
    return y - Z @ lstsq(Z, y, rcond=None)[0]


def partial_corr(x, y, controls):
    """(r, p) between x and y after regressing `controls` out of both.

    `controls` is a list of 1-D arrays and may be empty, which short-circuits to
    the plain correlation.

    The short circuit is not only an optimisation. Regressing on an
    intercept-only design demeans both vectors, and Pearson r is invariant to
    that in exact arithmetic but not in the last bit in floating point, so
    taking the general path for `controls=[]` would move the raw correlations
    this project reports.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not controls:
        return stats.pearsonr(x, y)
    return stats.pearsonr(residualize(x, controls), residualize(y, controls))


def hypergeometric_overlap(n_a: int, n_b: int, observed: int, universe: int):
    """(expected, fold enrichment, upper-tail p) for an overlap of two gene sets.

    `13_multidrug_models.py` asks this of a selected gene list against each
    signature; `14_drug_specificity.py` asks it of two drugs' selected lists.
    Those are different questions with the same arithmetic, and only the
    arithmetic is here. Which sets are intersected with the universe before
    counting differs between the two call sites and stays there, because
    changing it would change what is being tested.

    Degenerate cases return NaN rather than raising, matching both originals: an
    empty universe has no expectation, and an empty gene set has no p-value.
    """
    expected = n_a * n_b / universe if universe else np.nan
    fold = observed / expected if expected > 0 else np.nan
    p = (hypergeom.sf(observed - 1, universe, n_a, n_b)
         if n_a > 0 and n_b > 0 else np.nan)
    return expected, fold, p
