"""Every result table must match its frozen copy in tests/golden/.

Skips only when data/processed/ has never been populated. Once any output
exists, every golden file must be present and match: same columns, same rows,
identical text, numbers within 1e-9, NaN in the same places.

After an intentional change to a result, re-freeze with `make golden-update`
and say why in the commit.
"""

import pandas as pd
import pytest

from fivefu.paths import golden_dir, processed_dir

ATOL = 1e-9
NAMES = sorted(p.name for p in golden_dir().glob("*.csv"))
LIVE = processed_dir()

pytestmark = pytest.mark.skipif(
    not any((LIVE / n).exists() for n in NAMES),
    reason="data/processed/ is empty: run `make run` first. The gate did NOT run.",
)


@pytest.mark.parametrize("name", NAMES)
def test_output_matches_golden(name):
    live_path = LIVE / name
    assert live_path.exists(), f"{name} was not produced"
    golden = pd.read_csv(golden_dir() / name)
    live = pd.read_csv(live_path)
    pd.testing.assert_frame_equal(
        live, golden, check_exact=False, rtol=0, atol=ATOL, check_dtype=False
    )
