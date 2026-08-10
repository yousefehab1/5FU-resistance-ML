"""
lib/stats.py
============

Statistical tests that are either wholly new (`hypergeometric_enrichment`
does not exist anywhere in the original project -- its DTP_up 4.7x/p=1.1e-6
number lives only as a hand-typed literal and a raw console print of gene
overlaps) or were duplicated verbatim across scripts (`residualize` appeared
independently in the multidrug and Arm B scripts; `partial_correlation` was
buried in the now-archived `07_refined_model.py`).

`compositional_null` preserves the Arm B compositional-control Monte Carlo
test EXACTLY as it was, including the 200-draw count. This is a DIFFERENT
test from `lib.signatures.background_score`'s analytic null -- that null
was Monte Carlo once too, and was replaced because 200 draws proved far too
few to calibrate an individual signature score (rescoring with a different
seed gave r=0.74 for MYC against itself). This one only asks a coarser
question -- "does the observed TIME-TREND exceed what typical random gene
sets of this size give" -- where 200 draws is an adequate, still-standard
sample size. Do not merge these two nulls or "fix" this one to be analytic.
"""

import numpy as np
from scipy import stats

import config as C


def hypergeometric_enrichment(query_genes, signature_genes, universe, name=None):
    """
    Is `query_genes` enriched for `signature_genes`, against `universe`?

    query_genes   e.g. the ~413 non-zero-coefficient genes from the final
                  de-confounded model.
    signature_genes  e.g. the DTP_up gene set.
    universe      every gene the model COULD have selected from (the
                  expression matrix's full gene set), not just len(query).

    This is new code: the original project only ever printed the raw
    overlap between the model's genes and each signature, with no test of
    whether that overlap exceeds chance.
    """
    universe = set(universe)
    query = set(query_genes) & universe
    sig = set(signature_genes) & universe

    N = len(universe)
    K = len(sig)
    n = len(query)
    k = len(query & sig)

    expected = n * K / N if N else np.nan
    fold = k / expected if expected > 0 else np.nan
    # sf(k-1, ...) = P(X >= k) -- the enrichment (over-representation) tail.
    p = stats.hypergeom.sf(k - 1, N, K, n) if N else np.nan

    return dict(signature=name, universe_size=N, signature_size=K,
                query_size=n, observed_overlap=k, expected_overlap=expected,
                fold_enrichment=fold, p_value=p)


def residualize(y, x):
    """
    y with the linear effect of x removed: fit y ~ 1 + x by least squares,
    return the residuals. Used both for "5-FU-specific" AUC (residualized on
    general chemosensitivity) and for partialling CellCycle out of Arm B
    module trends.
    """
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    return y - design @ coef


def partial_correlation(x, y, Z):
    """
    Pearson r between x and y after regressing BOTH on covariates Z
    (n x p, no intercept column needed -- one is prepended here).

    Ported from the original's `07_refined_model.py::partial_r`, which is
    otherwise archived (superseded by fold-internal de-confounding for the
    main model) but whose MSI/TP53-adjustment logic is still the correct
    tool for a covariate the model itself does not use as a feature.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    Z = np.atleast_2d(np.asarray(Z, dtype=float))
    if Z.shape[0] != len(x):
        Z = Z.T
    design = np.column_stack([np.ones(len(x)), Z])

    bx, *_ = np.linalg.lstsq(design, x, rcond=None)
    by, *_ = np.linalg.lstsq(design, y, rcond=None)
    rx = x - design @ bx
    ry = y - design @ by
    return stats.pearsonr(rx, ry)


def ci_from_repeats(values, lo=2.5, hi=97.5):
    """95% CI (default) from a distribution of per-repeat/per-resample statistics."""
    values = np.asarray(values, dtype=float)
    return values.mean(), np.percentile(values, lo), np.percentile(values, hi)


def compositional_null(observed_scores, time_hours, feature, n_genes, expr,
                        n_draws=C.COMPOSITIONAL_NULL_DRAWS, seed=C.SEED):
    """
    Arm B compositional control: is `feature`'s correlation with time bigger
    than what a random gene set of the same size gives, drawn from the same
    expression matrix and scored the same way as the real signature?

    `n_genes` is the real signature's matched gene-set size (from
    `lib.signatures.load_signatures`) -- the null must draw sets of the same
    size, or the comparison is meaningless.

    Ported unchanged from `05_armB_induction.py` Section 5b, including the
    200-draw Monte Carlo count -- see module docstring for why this one
    stays Monte Carlo while the per-sample signature-scoring null does not.
    """
    from lib.signatures import background_score, rank_matrix

    observed_r, _ = stats.pearsonr(observed_scores, time_hours)

    ranks = rank_matrix(expr)
    rng = np.random.default_rng(seed)
    genes = expr.columns.to_numpy()
    null_rs = np.empty(n_draws)
    for i in range(n_draws):
        draw = rng.choice(genes, size=n_genes, replace=False)
        null_score = background_score(ranks, list(draw))
        null_rs[i] = stats.pearsonr(null_score, time_hours)[0]

    z = (observed_r - null_rs.mean()) / null_rs.std(ddof=1)
    p_empirical = (np.sum(np.abs(null_rs) >= np.abs(observed_r)) + 1) / (n_draws + 1)

    return dict(feature=feature, observed_r=observed_r, null_mean=null_rs.mean(),
                null_sd=null_rs.std(ddof=1), z=z, p_empirical=p_empirical,
                n_draws=n_draws)
