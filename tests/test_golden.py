"""Every result table must match its frozen copy in tests/golden/.

Skips only when data/processed/ has never been populated (data/raw/ is ~12GB
and not in the repository). Once any output exists, every golden file must be
present and match: same columns, same rows, identical text, numbers within
1e-9, NaN in the same places.

After an intentional change to a result, re-freeze with `make golden-update`
and say why in the commit.
"""

import pandas as pd
import pytest

import config as C

ATOL = 1e-9
GOLDEN = C.PROJECT_ROOT / "tests" / "golden"
NAMES = sorted(p.name for p in GOLDEN.glob("*.csv"))

pytestmark = pytest.mark.skipif(
    not any((C.PROCESSED / n).exists() for n in NAMES),
    reason="data/processed/ is empty: run `make run` first. The gate did NOT run.",
)


@pytest.mark.parametrize("name", NAMES)
def test_output_matches_golden(name):
    live_path = C.PROCESSED / name
    assert live_path.exists(), f"{name} was not produced"
    pd.testing.assert_frame_equal(
        pd.read_csv(live_path), pd.read_csv(GOLDEN / name),
        check_exact=False, rtol=0, atol=ATOL, check_dtype=False,
    )
