"""Behaviour of the fivefu library on small synthetic data."""

import numpy as np
import pandas as pd
import pytest
from scipy import stats as sps

from fivefu import config
from fivefu.cohorts import detect_gene_axis
from fivefu.modeling import permuted_label_check, repeated_cv
from fivefu.signatures import background_score, rank_matrix
from fivefu.stats import bootstrap_r, hypergeometric_overlap, partial_corr, residualize


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
    rs = repeated_cv(*dataset, deconfound=False)
    assert rs.mean() > 0.5


def test_cv_is_deterministic(dataset):
    assert (repeated_cv(*dataset, deconfound=True) == repeated_cv(*dataset, deconfound=True)).all()


def test_deconfounding_removes_the_lineage_signal(dataset):
    raw = repeated_cv(*dataset, deconfound=False).mean()
    deconf = repeated_cv(*dataset, deconfound=True).mean()
    assert deconf < raw


def test_shuffled_labels_find_nothing(dataset):
    """The leakage tripwire: feature selection inside the fold means no signal."""
    X, y, _ = dataset
    assert abs(permuted_label_check(X, y, "test", "test")["r_mean"]) < 0.3


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


# ---- signatures and cohorts -------------------------------------------------

def test_signature_scores_high_where_its_genes_are_high():
    rng = np.random.default_rng(2)
    expr = pd.DataFrame(rng.normal(size=(2, 200)), columns=[f"G{i}" for i in range(200)])
    genes = [f"G{i}" for i in range(20)]
    expr.loc[0, genes] += 5
    scores = background_score(rank_matrix(expr), genes)
    assert scores[0] > 2 > scores[1]


def test_gene_axis_is_detected_and_transposed():
    alias = {"MLH1": "MLH1", "TP53": "TP53"}
    genes_on_rows = pd.DataFrame(np.ones((3, 2)), index=["MLH1", "TP53", "KRAS"], columns=["s1", "s2"])
    out = detect_gene_axis(genes_on_rows, alias, verbose=False)
    assert "MLH1" in out.columns


def test_ensembl_ids_are_rejected():
    alias = {"MLH1": "MLH1"}
    ensembl = pd.DataFrame(np.ones((2, 2)), index=["ENSG00000076242", "ENSG00000141510"], columns=["s1", "s2"])
    with pytest.raises(SystemExit):
        detect_gene_axis(ensembl, alias, verbose=False)


# ---- config -----------------------------------------------------------------

def test_missing_config_key_raises():
    with pytest.raises(config.ConfigError):
        config.require({"a": {}}, "a", "b", source="test")
