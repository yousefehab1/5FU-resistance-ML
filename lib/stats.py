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
  screen_agreement()          GDSC1 vs GDSC2 on the same lines: the noise ceiling
  bh_q()                      Benjamini-Hochberg q within one family, NaN-safe
  confirmatory_mask()         which rows of a table are config.CONFIRMATORY tests
  rowwise_r()                 Pearson r along each row of two matrices, NaN-aware
  bootstrap_lines_r()         CI for a repeated-CV r by resampling cell lines
  corrected_resampled_ttest() Nadeau-Bengio test for two models on the same splits
  concordance_index()         Harrell's C, censoring-aware
  signed_set_test()           direction-aware test of a gene set on a signed vector
  set_shift_test()            gene sets vs expression-matched genes, permutation null

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
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats
from statsmodels.stats.multitest import multipletests

import config as C

N_BOOT = 500

# stages/06_drug_specificity.py resamples 2000 times where everything else resamples 500. It is
# estimating a difference of two correlations, which is noisier than either one.
N_BOOT_DIFF = 2000

# Cell-line bootstrap of repeated-CV predictions. The draws are cheap (no
# refitting), so use as many as the difference bootstrap.
N_BOOT_LINES = 2000

# Target permutations for the genome-wide gene-set test.
N_PERM = 1000


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


def fisher_ci(r, n, k=0):
    """95% CI for a (partial) correlation from n observations with k
    controls, by Fisher's z: atanh(r) +- 1.96 / sqrt(n - 3 - k)."""
    se = 1 / np.sqrt(n - 3 - k)
    return np.tanh(np.arctanh(r) - 1.96 * se), np.tanh(np.arctanh(r) + 1.96 * se)


def fisher_z_difference_p(r1, n1, r2, n2):
    """Two-sided p that two correlations from independent samples differ."""
    z = (np.arctanh(r1) - np.arctanh(r2)) / np.sqrt(1 / (n1 - 3) + 1 / (n2 - 3))
    return 2 * stats.norm.sf(abs(z))


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


def screen_agreement(a, b, lines=None):
    """dict(n, r, p, rho) between two screens' values for the same lines.

    `a` and `b` are Series indexed by cell line; `lines` optionally restricts
    to a line set (the solid, expression-matched modelling cohort). Two
    independent measurements of the same drug on the same lines agree only
    this well, so no model predicting one of them from expression can be
    expected to do better: this is the ceiling `pct_of_ceiling` is read against.
    """
    both = pd.concat([a, b], axis=1, join="inner").dropna()
    if lines is not None:
        both = both[both.index.isin(lines)]
    if len(both) < 10:
        return dict(n=len(both), r=np.nan, p=np.nan, rho=np.nan)
    x, y = both.iloc[:, 0].to_numpy(float), both.iloc[:, 1].to_numpy(float)
    r, p = stats.pearsonr(x, y)
    return dict(n=len(both), r=r, p=p, rho=stats.spearmanr(x, y)[0])


def bh_q(p):
    """Benjamini-Hochberg q for one family of p-values. NaNs stay NaN and do
    not count towards the family size."""
    p = np.asarray(p, float)
    q = np.full(p.shape, np.nan)
    ok = np.isfinite(p)
    if ok.any():
        q[ok] = multipletests(p[ok], method="fdr_bh")[1]
    return q


def confirmatory_mask(df, table):
    """Boolean Series: the rows of `df` that config.CONFIRMATORY declares as
    confirmatory tests for `table`. Every other row is exploratory."""
    mask = pd.Series(False, index=df.index)
    for spec in C.CONFIRMATORY.get(table, []):
        m = pd.Series(True, index=df.index)
        for col, val in spec.items():
            m &= df[col] == val
        mask |= m
    return mask


def rowwise_r(a, b):
    """Pearson r between matching rows of two 2-D arrays, over the entries
    finite in both. Rows with fewer than 3 such entries give NaN."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    m = np.isfinite(a) & np.isfinite(b)
    k = m.sum(1)
    with np.errstate(invalid="ignore", divide="ignore"):
        ma = np.where(m, a, 0).sum(1) / k
        mb = np.where(m, b, 0).sum(1) / k
        da = np.where(m, a - ma[:, None], 0)
        db = np.where(m, b - mb[:, None], 0)
        r = (da * db).sum(1) / np.sqrt((da ** 2).sum(1) * (db ** 2).sum(1))
    r[k < 3] = np.nan
    return r


def bootstrap_lines_r(truth, pred, n_boot=N_BOOT_LINES, seed=C.SEED):
    """
    Per-resample mean-over-repeats r, resampling CELL LINES.

    `truth` and `pred` are (n_lines, n_repeats) out-of-fold matrices from
    lib.modeling.cv_oof (NaN where a line was not scored in a repeat). Each
    draw resamples lines once and scores every repeat on that same draw, so
    the interval reflects which lines happened to be screened. The
    predictions are held fixed, so training-set variability is not in it;
    the corrected resampled t-test covers that for model comparisons.

    This replaces a percentile over the five per-repeat r values, which with
    five numbers is close to their min and max and says nothing about lines.
    """
    truth, pred = np.asarray(truth, float), np.asarray(pred, float)
    if truth.ndim == 1:
        truth, pred = truth[:, None], pred[:, None]
    idx = np.random.default_rng(seed).integers(0, len(truth), (n_boot, len(truth)))
    return np.nanmean(np.column_stack([rowwise_r(truth[idx, j], pred[idx, j])
                                       for j in range(truth.shape[1])]), axis=1)


def corrected_resampled_ttest(diffs, n_train, n_test):
    """
    (mean difference, t, two-sided p) for per-split score differences
    between two models evaluated on the SAME splits (Nadeau and Bengio 2003).

    Splits of repeated k-fold CV share training rows, so the naive t-test
    is anticonservative. The correction inflates the variance by
    (1/J + n_test/n_train) and keeps J-1 degrees of freedom.
    """
    d = np.asarray(diffs, float)
    d = d[np.isfinite(d)]
    J = len(d)
    if J < 2:
        return np.nan, np.nan, np.nan
    var = d.var(ddof=1) * (1 / J + n_test / n_train)
    if var == 0:
        return d.mean(), np.nan, np.nan
    t = d.mean() / np.sqrt(var)
    return d.mean(), t, 2 * stats.t.sf(abs(t), J - 1)


def concordance_index(y, pred, event=None):
    """
    Harrell's C: the fraction of comparable pairs the prediction orders
    correctly, ties in the prediction counting one half.

    A pair (i, j) with y_i < y_j is comparable when i's value is observed
    (event_i True). With `event=None` every value is observed and C equals
    (Kendall's tau_a + 1) / 2. For a right-censored response (IC50 beyond
    the highest dose tested), event marks the lines whose IC50 was resolved.
    """
    y, pred = np.asarray(y, float), np.asarray(pred, float)
    ok = np.isfinite(y) & np.isfinite(pred)
    ev = np.ones(len(y), bool) if event is None else np.asarray(event, bool)
    y, pred, ev = y[ok], pred[ok], ev[ok]
    comparable = (y[:, None] < y[None, :]) & ev[:, None]
    n = comparable.sum()
    if n == 0:
        return np.nan
    dp = pred[None, :] - pred[:, None]
    return ((comparable & (dp > 0)).sum() + 0.5 * (comparable & (dp == 0)).sum()) / n


def signed_set_test(values, set_genes):
    """
    Direction-aware test of one gene set on a signed per-gene vector, here
    the model's non-zero coefficients (positive = higher AUC = resistant).
    Returns dict(n, n_positive, p_sign, mean_value, p_rank):
      p_sign   binomial test of the share of set genes with positive values
               against one half
      p_rank   Mann-Whitney test of the set's values against the other genes
    Both treat genes as independent. For a genome-wide vector use
    set_shift_test instead.
    """
    values = values.dropna()
    inset = values.index.isin(list(set_genes))
    v_in, v_out = values[inset].to_numpy(float), values[~inset].to_numpy(float)
    n = len(v_in)
    out = dict(n=n, n_positive=int((v_in > 0).sum()), p_sign=np.nan,
               mean_value=v_in.mean() if n else np.nan, p_rank=np.nan)
    if n == 0:
        return out
    out["p_sign"] = stats.binomtest(out["n_positive"], n, 0.5).pvalue
    if len(v_out):
        out["p_rank"] = stats.mannwhitneyu(v_in, v_out, alternative="two-sided").pvalue
    return out


def set_shift_test(X, y, gene_sets, strata, groups=None, n_perm=N_PERM, seed=C.SEED):
    """
    Does each gene set correlate with `y` more, or less, than genes like it?

    For every gene, r = corr(X[:, gene], y). A set's SHIFT is its mean r
    minus an expression-matched background: the genome-wide mean r within
    each `strata` bin (e.g. expression decile), weighted by how many of the
    set's genes fall in that bin. Matching matters: curated genes are well
    expressed, and well-expressed genes carry higher r than the genome as a
    whole, so an unmatched comparison calls every curated set significant.

    The null permutes `y` (within `groups`, if given) and recomputes both
    the set mean and the background, so correlation among a set's genes is
    kept. A random-gene-set null would not keep it: co-regulated genes move
    together, and treating them as independent makes every set look
    significant.

    Returns a DataFrame indexed by set name: n, mean_r, background_r, shift,
    p_perm (two-sided, (1 + #|null| >= |shift|) / (1 + n_perm)).
    """
    genes = np.asarray(X.columns)
    Xv = np.asarray(X, float)
    Xv = Xv - Xv.mean(0)
    xn = np.sqrt((Xv ** 2).sum(0))
    xn[xn == 0] = np.nan
    lab = pd.Series(strata).reindex(genes).to_numpy()
    bins = [lab == b for b in np.unique(lab[pd.notna(lab)])]
    sets = {k: np.isin(genes, list(v)) for k, v in gene_sets.items()}
    weights = {k: np.array([(m & b).sum() for b in bins]) / max(m.sum(), 1)
               for k, m in sets.items()}

    def shifts(Y):
        Yc = Y - Y.mean(0)
        R = (Xv.T @ Yc) / (xn[:, None] * np.sqrt((Yc ** 2).sum(0))[None, :])
        bg = np.vstack([np.nanmean(R[b], 0) for b in bins])
        return {k: (np.nanmean(R[m], 0), w @ bg) for k, (m, w) in
                ((k, (sets[k], weights[k])) for k in sets)}

    y = np.asarray(y, float)
    obs = shifts(y[:, None])
    rng = np.random.default_rng(seed)
    grp = np.zeros(len(y), int) if groups is None else pd.factorize(np.asarray(groups))[0]
    members = [np.flatnonzero(grp == g) for g in np.unique(grp)]
    null = {k: [] for k in sets}
    for start in range(0, n_perm, 100):
        Y = np.empty((len(y), min(100, n_perm - start)))
        for j in range(Y.shape[1]):
            perm = np.arange(len(y))
            for idx in members:
                perm[idx] = rng.permutation(idx)
            Y[:, j] = y[perm]
        for k, (m, b) in shifts(Y).items():
            null[k].append(m - b)
    rows = {}
    for k in sets:
        nl = np.concatenate(null[k])
        m, b = obs[k][0][0], obs[k][1][0]
        rows[k] = dict(n=int(sets[k].sum()), mean_r=m, background_r=b, shift=m - b,
                       p_perm=(1 + (np.abs(nl) >= abs(m - b)).sum()) / (1 + len(nl)))
    return pd.DataFrame.from_dict(rows, orient="index")
