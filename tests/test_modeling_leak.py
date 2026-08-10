"""
tests/test_modeling_leak.py
============================

Synthetic-leak regression test for lib.modeling.repeated_cv's fold-internal
lineage de-confounding.

WHY THIS TEST EXISTS
---------------------
This project's own history includes a caught data leak: an earlier version
of the final model computed lineage means over the WHOLE dataset (train and
test together) rather than the training fold only, and reported a
de-confounded association of r=0.279. The corrected, fold-internal version
reports r=0.182 -- the gap between those two numbers IS the leak.

The synthetic data here has NO real relationship between X and y beyond
lineage identity: y is exactly a lineage mean plus noise, X is pure random
noise unrelated to either. After correct de-confounding, nothing should be
left for any model to predict, so a correctly-implemented `repeated_cv`
must return r_mean close to zero. If de-confounding regresses to using
whole-dataset statistics (or is silently skipped), finite-sample
idiosyncrasies of the held-out fold leak into its own residualization and
this test starts failing -- catching exactly the historical bug class
before it reaches a real result.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pytest
from sklearn.linear_model import Ridge

from lib.modeling import repeated_cv


def _synthetic_lineage_only_data(n_lineages=6, n_per_lineage=25, n_features=20, seed=1):
    """
    y depends ONLY on lineage identity (a group mean plus noise); X is
    independent random noise, unrelated to y or lineage. Any CV correlation
    a de-confounded model shows here is necessarily a leak, not signal.
    """
    rng = np.random.default_rng(seed)
    lineage = np.repeat([f"L{i}" for i in range(n_lineages)], n_per_lineage)
    lineage_means = rng.normal(0, 2.0, n_lineages)
    y = np.repeat(lineage_means, n_per_lineage) + rng.normal(0, 0.5, len(lineage))
    X = rng.normal(size=(len(lineage), n_features))
    return X, y, lineage


def test_deconfounded_cv_near_zero_on_pure_lineage_noise():
    X, y, lineage = _synthetic_lineage_only_data()
    result = repeated_cv(X, y, lineage=lineage, deconfound=True,
                          name="synthetic leak test",
                          model_factory=lambda: Ridge(alpha=1.0),
                          n_repeats=5, n_folds=5, seed=0, verbose=False)
    assert abs(result["r_mean"]) < 0.2, (
        f"De-confounded CV recovered r={result['r_mean']:.3f} on data with no real "
        "X-y relationship beyond lineage. This is the signature of the historical "
        "leak bug: lineage statistics computed over the whole dataset (train+test) "
        "rather than the training fold only."
    )


def test_deconfound_requires_lineage():
    X, y, _ = _synthetic_lineage_only_data()
    with pytest.raises(ValueError):
        repeated_cv(X, y, lineage=None, deconfound=True, verbose=False)
