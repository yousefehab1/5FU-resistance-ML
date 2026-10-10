"""Behaviour of lib/ on small synthetic data. Needs no project data."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps
from sklearn.linear_model import LinearRegression, Ridge
from statsmodels.stats.multitest import multipletests

import statsmodels.formula.api as smf

from lib.cohorts import AdjustedLogitZ, detect_gene_axis, zscore
import config as C
from lib.modeling import (chosen_alpha, compare_models, cv_leave_lineage_out, cv_oof,
                          cv_repeats, elasticnet_pipeline, fixed_elasticnet_pipeline,
                          permuted_label_check, scaled_target, summarise)
from lib.signatures import background_score, rank_matrix
from lib.stats import (bh_q, bootstrap_lines_r, bootstrap_r, concordance_index,
                       confirmatory_mask, corrected_resampled_ttest, fisher_ci,
                       fisher_z_difference_p, hypergeometric_enrichment,
                       hypergeometric_overlap, partial_corr, residualize, screen_agreement,
                       set_shift_test, signed_set_test)


@pytest.fixture(scope="module")
def dataset():
    """60 lines x 40 genes, 3 lineages; the target depends on gene 0 and lineage."""
    rng = np.random.default_rng(0)
    lineage = pd.Series(np.repeat(["A", "B", "C"], 20))
    X = pd.DataFrame(rng.normal(size=(60, 40)), columns=[f"G{i}" for i in range(40)])
    X["G1"] += lineage.map({"A": 0, "B": 2, "C": 4})
    y = pd.Series(0.8 * X["G0"] + lineage.map({"A": 0, "B": 1, "C": 2}) + rng.normal(scale=0.5, size=60))
    return X, y, lineage


@pytest.fixture(scope="module")
def confounded():
    rng = np.random.default_rng(1)
    z = rng.normal(size=200)
    x = 0.8 * z + rng.normal(scale=0.5, size=200)
    y = 0.7 * z + rng.normal(scale=0.5, size=200)
    return x, y, z


# ---- modelling --------------------------------------------------------------

def test_cv_finds_a_real_signal(dataset):
    assert cv_repeats(*dataset, deconfound=False).mean() > 0.5


def test_cv_is_deterministic(dataset):
    assert (cv_repeats(*dataset, deconfound=True) == cv_repeats(*dataset, deconfound=True)).all()


def test_deconfounding_removes_the_lineage_signal(dataset):
    """The de-confounding mechanics, on the fixed-alpha model: a near-unpenalised
    tuned fit reads lineage from G1 too well for the raw target to lose."""
    f = fixed_elasticnet_pipeline
    assert (cv_repeats(*dataset, deconfound=True, model_factory=f).mean()
            < cv_repeats(*dataset, deconfound=False, model_factory=f).mean())


def test_deconfounded_cv_near_zero_on_pure_lineage_noise():
    """y depends ONLY on lineage and X is pure noise, so any de-confounded CV
    correlation is a leak: lineage statistics computed over train+test rather
    than the training fold only (the historical r=0.279 vs 0.182 bug)."""
    rng = np.random.default_rng(1)
    lineage = np.repeat([f"L{i}" for i in range(6)], 25)
    y = np.repeat(rng.normal(0, 2.0, 6), 25) + rng.normal(0, 0.5, len(lineage))
    X = rng.normal(size=(len(lineage), 20))
    rs = cv_repeats(X, y, lineage, deconfound=True, model_factory=lambda: Ridge(alpha=1.0))
    assert abs(rs.mean()) < 0.2


def test_deconfound_requires_lineage(dataset):
    X, y, _ = dataset
    with pytest.raises(ValueError):
        cv_repeats(X, y, lineage=None, deconfound=True)


def test_shuffled_labels_find_nothing(dataset):
    """The leakage tripwire: feature selection inside the fold means no signal."""
    X, y, _ = dataset
    assert abs(permuted_label_check(X, y, "test", "test")["r_mean"]) < 0.3



def test_scaled_target_leaves_an_unpenalised_fit_unchanged(dataset):
    """OLS is invariant to the target's scale, so rescaling must round-trip."""
    X, y, _ = dataset
    plain = LinearRegression().fit(X, y).predict(X)
    wrapped = scaled_target(0.15, LinearRegression)().fit(X, y).predict(X)
    assert np.allclose(plain, wrapped)


def test_tuning_shrinks_noise_harder_than_signal():
    """Inner CV must pick a larger alpha when the target is noise than when a
    few genes drive it."""
    rng = np.random.default_rng(7)
    X = rng.normal(size=(150, 60))
    signal = X[:, :3] @ np.array([1.0, -0.8, 0.6]) + rng.normal(scale=0.3, size=150)
    noise = rng.normal(size=150)
    a_sig, _ = chosen_alpha(elasticnet_pipeline().fit(X, signal))
    a_noise, _ = chosen_alpha(elasticnet_pipeline().fit(X, noise))
    assert np.isclose(a_sig, C.ELASTICNET_ALPHAS).any()
    assert a_noise > a_sig


def test_cv_oof_records_the_tuned_alphas(dataset):
    X, y, lineage = dataset
    tuned = cv_oof(X, y, n_repeats=2)
    fixed = cv_oof(X, y, model_factory=fixed_elasticnet_pipeline, n_repeats=2)
    assert np.isfinite(tuned["alpha"]).all()
    grid = C.ELASTICNET_ALPHAS_FEW        # 40 columns, at most C.FEW_FEATURES
    assert np.isclose(tuned["alpha"].ravel()[:, None], grid).any(1).all()
    assert np.isnan(fixed["alpha"]).all()
    row, ref = summarise(tuned), summarise(fixed)
    assert min(grid) <= row["alpha_median"] <= max(grid)
    assert 0 <= row["alpha_edge_frac"] <= 1
    assert np.isnan(ref["alpha_median"])


def test_cv_repeats_matches_the_concatenated_fold_loop(dataset):
    """cv_repeats is per-repeat r over the predictions in fold order, exactly
    as the loop it replaced computed it."""
    X, y, lineage = dataset
    o = cv_oof(X, y, lineage, deconfound=True)
    expected = []
    for rep in range(o["fold"].shape[1]):
        f = o["fold"][:, rep]
        order = np.concatenate([np.flatnonzero(f == k) for k in range(f.max() + 1)])
        expected.append(sps.pearsonr(o["truth"][order, rep], o["pred"][order, rep])[0])
    assert (cv_repeats(X, y, lineage, deconfound=True) == np.array(expected)).all()


def test_compare_models_refuses_different_splits(dataset):
    X, y, _ = dataset
    a = cv_oof(X, y, seed=0)
    with pytest.raises(ValueError):
        compare_models(a, cv_oof(X, y, seed=1), "a", "b")
    row = compare_models(a, cv_oof(X[["G0"]], y, seed=0), "all genes", "G0 only")
    assert row["n_splits"] == 25 and np.isfinite(row["p"])


def test_leave_lineage_out_scores_every_line(dataset):
    X, y, lineage = dataset
    row = cv_leave_lineage_out(X, y, lineage, "toy")
    assert row["n_lineages_within"] == 3
    assert row["r_lo"] < row["r_mean"] < row["r_hi"]

# ---- statistics -------------------------------------------------------------

def test_partial_corr_without_controls_is_pearson(confounded):
    x, y, _ = confounded
    assert partial_corr(x, y, []) == sps.pearsonr(x, y)


def test_partial_corr_removes_a_confounder(confounded):
    x, y, z = confounded
    raw, adjusted = partial_corr(x, y, [])[0], partial_corr(x, y, [z])[0]
    assert raw > 0.5 and abs(adjusted) < raw / 2


def test_residuals_are_orthogonal_to_controls(confounded):
    _, y, z = confounded
    assert abs(np.corrcoef(residualize(y, [z]), z)[0, 1]) < 1e-10


def test_bootstrap_interval_contains_the_estimate(confounded):
    x, y, _ = confounded
    r, _, lo, hi = bootstrap_r(x, y)
    assert lo < r < hi
    assert bootstrap_r(x, y) == bootstrap_r(x, y)


def test_enrichment_separates_real_overlap_from_chance():
    _, fold_hit, p_hit = hypergeometric_overlap(900, 900, 800, 4000)
    _, fold_null, p_null = hypergeometric_overlap(900, 900, 202, 4000)
    assert p_hit < 1e-10 and fold_hit > 3
    assert p_null > 0.4 and fold_null == pytest.approx(1.0, abs=0.02)


def test_enrichment_edge_cases_give_nan():
    assert np.isnan(hypergeometric_overlap(10, 10, 0, 0)[1])
    assert np.isnan(hypergeometric_overlap(0, 10, 0, 4000)[2])


def test_enrichment_on_gene_sets_matches_known_case():
    universe = [f"g{i}" for i in range(100)]
    signature = set(universe[:20])
    query = set(universe[:8]) | set(universe[50:54]) | {"not_in_universe"}
    r = hypergeometric_enrichment(query, signature, universe, name="toy")
    assert (r["observed_overlap"], r["signature_size"], r["query_size"]) == (8, 20, 12)
    assert r["fold_enrichment"] == pytest.approx(8 / (12 * 20 / 100))
    assert r["p_value"] == pytest.approx(sps.hypergeom.sf(7, 100, 20, 12))


def test_bh_q_matches_statsmodels_and_skips_nan():
    p = np.array([0.001, 0.02, np.nan, 0.04, 0.5])
    q = bh_q(p)
    assert np.isnan(q[2])
    finite = p[~np.isnan(p)]
    assert np.allclose(q[~np.isnan(p)], multipletests(finite, method="fdr_bh")[1])
    assert (q[~np.isnan(p)] >= finite).all()


def test_screen_agreement_respects_the_line_set():
    idx = [f"L{i}" for i in range(40)]
    rng = np.random.default_rng(3)
    a = pd.Series(rng.normal(size=40), index=idx)
    b = a + rng.normal(scale=0.3, size=40)
    b.iloc[:20] = rng.normal(size=20)            # first half: no agreement
    full = screen_agreement(a, b)
    second_half = screen_agreement(a, b, lines=idx[20:])
    assert full["n"] == 40 and second_half["n"] == 20
    assert second_half["r"] > full["r"]
    assert np.isnan(screen_agreement(a, b, lines=idx[:5])["r"])


def test_confirmatory_mask_selects_declared_rows(monkeypatch):
    import config
    monkeypatch.setitem(config.CONFIRMATORY, "toy", [{"feature": "DTP", "model": "adjusted"}])
    df = pd.DataFrame({"feature": ["DTP", "DTP", "MYC"], "model": ["adjusted", "unadjusted", "adjusted"]})
    assert confirmatory_mask(df, "toy").tolist() == [True, False, False]
    assert not confirmatory_mask(df, "undeclared").any()



def test_bootstrap_lines_r_is_reproducible_and_brackets_r(confounded):
    x, y, _ = confounded
    truth = np.column_stack([y, y])
    pred = np.column_stack([x, x + 0.01])
    draws = bootstrap_lines_r(truth, pred)
    assert (draws == bootstrap_lines_r(truth, pred)).all()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert lo < sps.pearsonr(x, y)[0] < hi


def test_fisher_ci_matches_hand_computation():
    lo, hi = fisher_ci(0.3, 43, k=1)
    se = 1 / np.sqrt(43 - 3 - 1)
    assert np.isclose(lo, np.tanh(np.arctanh(0.3) - 1.96 * se))
    assert np.isclose(hi, np.tanh(np.arctanh(0.3) + 1.96 * se))
    assert lo < 0.3 < hi and fisher_ci(0.3, 43, k=5)[0] < lo   # controls widen it


def test_fisher_z_difference_is_symmetric_and_null_at_equal_r():
    assert np.isclose(fisher_z_difference_p(0.3, 30, 0.3, 16), 1.0)
    assert np.isclose(fisher_z_difference_p(0.5, 30, 0.1, 16),
                      fisher_z_difference_p(0.1, 16, 0.5, 30))
    assert fisher_z_difference_p(0.6, 300, 0.0, 300) < 1e-6


def test_corrected_ttest_matches_hand_computation():
    d = np.array([0.02, 0.05, -0.01, 0.03, 0.04])
    mean, t, p = corrected_resampled_ttest(d, n_train=80, n_test=20)
    se = np.sqrt(d.var(ddof=1) * (1 / 5 + 20 / 80))
    assert mean == pytest.approx(0.026)
    assert t == pytest.approx(0.026 / se)
    assert p == pytest.approx(2 * sps.t.sf(0.026 / se, 4))
    naive_t = 0.026 / np.sqrt(d.var(ddof=1) / 5)
    assert t < naive_t


def test_concordance_is_kendall_without_ties_or_censoring(confounded):
    x, y, _ = confounded
    assert concordance_index(y, x) == pytest.approx((sps.kendalltau(y, x)[0] + 1) / 2)


def test_concordance_skips_pairs_led_by_a_censored_value():
    y = np.array([1.0, 2.0, 3.0])
    pred = np.array([3.0, 2.0, 1.0])          # every pair ordered wrongly
    assert concordance_index(y, pred) == 0.0
    event = np.array([False, True, True])     # line 0 censored: only (1, 2) counts
    assert concordance_index(y, pred, event) == 0.0
    assert concordance_index(y, np.array([3.0, 1.0, 2.0]), event) == 1.0


def test_signed_set_test_reads_direction():
    genes = [f"g{i}" for i in range(200)]
    rng = np.random.default_rng(4)
    v = pd.Series(rng.normal(size=200), index=genes)
    v.iloc[:15] = np.abs(v.iloc[:15]) + 1     # a set pushed to resistance
    up = signed_set_test(v, genes[:15])
    assert up["n"] == 15 and up["n_positive"] == 15
    assert up["p_sign"] < 0.001 and up["p_rank"] < 0.001
    assert signed_set_test(v, ["absent"])["n"] == 0


def test_set_shift_test_keeps_gene_correlation():
    """Two co-regulated sets: one tracks y, one shares only a latent factor
    unrelated to y. Only the first may be called. A random-gene-set null,
    which treats the factor's genes as independent, would call both."""
    rng = np.random.default_rng(6)
    n = 120
    y = rng.normal(size=n)
    driven = 0.5 * y[:, None] + rng.normal(size=(n, 20))
    factor = rng.normal(size=n)
    coreg = 1.5 * factor[:, None] + rng.normal(size=(n, 20))
    noise = rng.normal(size=(n, 160))
    X = pd.DataFrame(np.hstack([driven, coreg, noise]),
                     columns=[f"g{i}" for i in range(200)])
    strata = pd.Series(0, index=X.columns)
    res = set_shift_test(X, y, {"driven": X.columns[:20], "coreg": X.columns[20:40]},
                         strata, n_perm=200)
    assert res.loc["driven", "shift"] > 0 and res.loc["driven", "p_perm"] < 0.01
    assert res.loc["coreg", "p_perm"] > 0.05

# ---- signatures and cohorts -------------------------------------------------

def test_signature_scores_high_where_its_genes_are_high():
    rng = np.random.default_rng(2)
    expr = pd.DataFrame(rng.normal(size=(2, 200)), columns=[f"G{i}" for i in range(200)])
    genes = [f"G{i}" for i in range(20)]
    expr.loc[0, genes] += 5
    scores = background_score(rank_matrix(expr), genes)
    assert scores[0] > 2 > scores[1]


def test_adjusted_logit_z_matches_a_direct_statsmodels_fit():
    # One design built once, score column swapped per call: must equal a fresh
    # formula fit, and its C() must be patsy's even though config is C here.
    rng = np.random.default_rng(4)
    n, genes = 120, [f"G{i}" for i in range(300)]
    expr = pd.DataFrame(rng.normal(size=(n, 300)), columns=genes,
                        index=[f"s{i}" for i in range(n)])
    up, down = genes[:15], genes[15:30]
    expr[up] = expr[up].add(rng.normal(size=n), axis=0)
    ranks = rank_matrix(expr)
    cov = pd.DataFrame({"grp": rng.choice(["a", "b", "c"], n)}, index=expr.index)
    true = background_score(ranks, up) - background_score(ranks, down)
    cov["response"] = rng.random(n) < 1 / (1 + np.exp(-0.8 * zscore(pd.Series(true)).to_numpy()))

    wald_z = AdjustedLogitZ(ranks, cov, "+ C(grp)")
    d = cov.assign(z=zscore(pd.Series(true, index=expr.index)),
                   response=cov.response.astype(int))
    direct = smf.logit("response ~ z + grp", data=d).fit(disp=0).tvalues["z"]
    assert np.isclose(wald_z([up, down]), direct)
    assert not np.isclose(wald_z([up]), direct)   # a different score refits


def test_gene_axis_is_detected_and_transposed():
    alias = {"MLH1": "MLH1", "TP53": "TP53"}
    genes_on_rows = pd.DataFrame(np.ones((3, 2)), index=["MLH1", "TP53", "KRAS"], columns=["s1", "s2"])
    assert "MLH1" in detect_gene_axis(genes_on_rows, alias, verbose=False).columns


def test_ensembl_ids_are_rejected():
    alias = {"MLH1": "MLH1"}
    ensembl = pd.DataFrame(np.ones((2, 2)), index=["ENSG00000076242", "ENSG00000141510"], columns=["s1", "s2"])
    with pytest.raises(SystemExit):
        detect_gene_axis(ensembl, alias, verbose=False)
