"""
lib/stats.py
============

Statistical helpers shared by the stages.

  hypergeometric_overlap()    expected overlap, fold enrichment, upper-tail p
  hypergeometric_enrichment() the same test on actual gene sets, as a result row
  residualize()               what is left of y once controls are regressed out
  partial_corr()              Pearson r after regressing controls out of both
  bootstrap_pearson_draws()   per-resample Pearson r
  bootstrap_r()               (r, p, lo, hi)
  bootstrap_r_difference()    per-resample r1 - r2 for two independent samples
  compositional_null()        Arm B's random-gene-set trend null

The bootstrap draws its indices in a fixed way: a fresh default_rng(seed) per
call and one integers(0, n, (n_boot, n)) matrix. Drawing them any other way,
even an equivalent one, moves every published confidence interval.

`compositional_null` is a DIFFERENT test from `lib.signatures.background_score`'s
analytic null. That null was Monte Carlo once too, and was replaced because 200
draws proved far too few to calibrate an individual signature score. This one
only asks whether a TIME-TREND exceeds what random gene sets of the same size
give, where 200 draws is adequate. Do not merge the two nulls.
"""

import numpy as np
from numpy.linalg import lstsq
from scipy import stats

import config as C

N_BOOT = 500

# Stage 12 resamples 2000 times where everything else resamples 500. It is
# estimating a difference of two correlations, which is noisier than either one.
N_BOOT_DIFF = 2000


def hypergeometric_overlap(n_a, n_b, observed, universe):
    """(expected, fold enrichment, upper-tail p) for an overlap of two gene sets.

    Degenerate cases return NaN rather than raising: an empty universe has no
    expectation, and an empty gene set has no p-value.
    """
    expected = n_a * n_b / universe if universe else np.nan
    fold = observed / expected if expected > 0 else np.nan
    p = (stats.hypergeom.sf(observed - 1, universe, n_a, n_b)
         if n_a > 0 and n_b > 0 else np.nan)
    return expected, fold, p


def hypergeometric_enrichment(query_genes, signature_genes, universe, name=None):
    """
    Is `query_genes` enriched for `signature_genes`, against `universe`?

    universe is every gene the model COULD have selected from (the expression
    matrix's full gene set), not just the query. Both sets are intersected
    with it before counting.
    """
    universe = set(universe)
    query = set(query_genes) & universe
    sig = set(signature_genes) & universe
    N, K, n, k = len(universe), len(sig), len(query), len(query & sig)
    expected, fold, p = hypergeometric_overlap(n, K, k, N)
    return dict(signature=name, universe_size=N, signature_size=K,
                query_size=n, observed_overlap=k, expected_overlap=expected,
                fold_enrichment=fold, p_value=p)


def design(n, controls):
    """`controls` as an OLS design matrix with an intercept column first."""
    return np.column_stack([np.ones(n)] + [np.asarray(z, float) for z in controls])


def residualize(y, controls):
    """What is left of `y` once `controls` (a list of 1-D arrays) are regressed
    out by OLS with an intercept."""
    y = np.asarray(y, float)
    Z = design(len(y), controls)
    return y - Z @ lstsq(Z, y, rcond=None)[0]


def partial_corr(x, y, controls):
    """(r, p) between x and y after regressing `controls` out of both.

    `controls` may be empty, which short-circuits to the plain correlation.
    That is not only an optimisation: demeaning is exact in theory but not in
    the last bit of floating point, so the general path would move the raw
    correlations this project reports.
    """
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not controls:
        return stats.pearsonr(x, y)
    return stats.pearsonr(residualize(x, controls), residualize(y, controls))


def bootstrap_pearson_draws(x, y, n_boot=N_BOOT, seed=C.SEED):
    """Per-resample Pearson r. `x` and `y` must be numpy arrays: the indexing is
    positional, and pandas would resolve an integer array against the index."""
    n = len(x)
    rng = np.random.default_rng(seed)
    return np.array([stats.pearsonr(x[i], y[i])[0]
                     for i in rng.integers(0, n, (n_boot, n))])


def bootstrap_r(x, y, n_boot=N_BOOT, seed=C.SEED):
    """Return (r, p, lo, hi)."""
    r, p = stats.pearsonr(x, y)
    bs = bootstrap_pearson_draws(x, y, n_boot=n_boot, seed=seed)
    return r, p, np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def bootstrap_r_difference(x1, y1, x2, y2, n_boot=N_BOOT_DIFF, seed=C.SEED):
    """Per-resample (r1 - r2) for two independent, unequally sized samples.

    ORDER IS LOAD-BEARING: both index draws come from one generator,
    alternating within the loop. Drawing all of sample 1's indices first would
    be the same estimator but a different sequence, and the interval would move.
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


def compositional_null(observed_scores, time_hours, n_genes, ranks, rng,
                        feature=None, n_draws=C.COMPOSITIONAL_NULL_DRAWS):
    """
    Arm B compositional control: is a feature's correlation with time bigger
    than what a random gene set of the same size gives, drawn from the same
    expression matrix and scored the same way as the real signature?

    `n_genes` is the real signature's matched size. `ranks` and `rng` are
    passed in (computed and seeded ONCE by the caller) so the draws for
    different features consume one continuous stream. The empirical p measures
    deviation from the null's OWN mean, not from zero.
    """
    observed_r, _ = stats.pearsonr(observed_scores, time_hours)

    from lib.signatures import background_score

    genes = ranks.columns.to_numpy()
    null_rs = np.empty(n_draws)
    for i in range(n_draws):
        draw = rng.choice(genes, size=n_genes, replace=False)
        null_score = background_score(ranks, list(draw))
        null_rs[i] = stats.pearsonr(null_score, time_hours)[0]

    z = (observed_r - null_rs.mean()) / null_rs.std()
    p_empirical = (np.abs(null_rs - null_rs.mean()) >= abs(observed_r - null_rs.mean())).mean()

    return dict(feature=feature, observed_r=observed_r, null_mean=null_rs.mean(),
                null_sd=null_rs.std(), z=z, p_empirical=p_empirical, n_draws=n_draws)
