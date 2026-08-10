"""
tests/test_enrichment.py
==========================

Deterministic unit test for lib.stats.hypergeometric_enrichment against a
hand-constructed toy case with a known answer -- this function is new code
(the original project never implemented a real enrichment test), so there
is no prior implementation to diff against; this pins its arithmetic
instead.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pytest
from scipy.stats import hypergeom

from lib.stats import hypergeometric_enrichment


def test_hypergeometric_enrichment_matches_known_case():
    universe = [f"g{i}" for i in range(100)]
    signature = set(universe[:20])                        # 20 of 100
    query = set(universe[:8]) | set(universe[50:54])       # 8 overlap, 4 not

    r = hypergeometric_enrichment(query, signature, universe, name="toy")

    assert r["observed_overlap"] == 8
    assert r["signature_size"] == 20
    assert r["query_size"] == 12
    assert r["universe_size"] == 100

    expected = 12 * 20 / 100
    assert r["expected_overlap"] == pytest.approx(expected)
    assert r["fold_enrichment"] == pytest.approx(8 / expected)
    assert r["p_value"] == pytest.approx(hypergeom.sf(7, 100, 20, 12))


def test_hypergeometric_enrichment_ignores_genes_outside_universe():
    universe = [f"g{i}" for i in range(50)]
    signature = set(universe[:10]) | {"not_in_universe_1"}
    query = set(universe[:5]) | {"not_in_universe_2"}

    r = hypergeometric_enrichment(query, signature, universe, name="toy")

    assert r["universe_size"] == 50
    assert r["signature_size"] == 10       # the out-of-universe gene dropped
    assert r["query_size"] == 5            # the out-of-universe gene dropped
