"""
tests/test_golden.py
======================

Golden-file regression tests against the original project's checked-in
final numbers. These only run if data/processed/ has already been
populated by a full pipeline run (data/raw/ -- ~12GB -- is gitignored and
not available in every environment), so they are skipped rather than
failed when that output doesn't exist.

Reference values:
  - final_model_results.csv: raw r=0.427, lineage-de-confounded r=0.182.
    These are EXACT-MATCH tier -- identical copied data + SEED=0 fully
    determines fold assignment, so tolerance is tight.
  - enrichment_results.csv, DTP_up row: fold~4.7x, p~1.1e-6, as recorded in
    the original project's docs/16_final_model.md. This table does not
    exist as original-project code output (only a hand-typed dashboard
    literal), so docs/16 is the best available ground truth and tolerance
    is looser.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
import pytest

import config as C

FINAL_MODEL = C.PROCESSED / "final_model_results.csv"
ENRICHMENT = C.PROCESSED / "enrichment_results.csv"

pytestmark = pytest.mark.skipif(
    not (FINAL_MODEL.exists() and ENRICHMENT.exists()),
    reason="data/processed/ not populated -- run the stages/ pipeline first",
)


def test_final_model_raw_and_deconfounded_r():
    res = pd.read_csv(FINAL_MODEL).set_index("model")
    raw = res.loc["transcriptome -> raw AUC", "r_mean"]
    deconf = res.loc["transcriptome -> AUC, lineage de-confounded", "r_mean"]
    assert raw == pytest.approx(0.427, abs=0.005)
    assert deconf == pytest.approx(0.182, abs=0.02)


def test_dtp_up_enrichment():
    res = pd.read_csv(ENRICHMENT).set_index("signature")
    row = res.loc["DTP_up"]
    assert row["fold_enrichment"] == pytest.approx(4.7, abs=0.3)
    assert row["p_value"] < 1e-5
