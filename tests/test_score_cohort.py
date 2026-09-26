"""lib.cohorts.score_cohort must reproduce stage 08's HCT116 time-course scores.

score_cohort is how every external cohort (stage 14's clinical studies) is
scored, so it has to agree with the scoring the time-course analysis used.
Needs data/raw/ and a populated data/processed/; skipped otherwise.
"""

import numpy as np
import pandas as pd
import pytest

import config as C
from lib.cohorts import score_cohort

STORED = C.PROCESSED / "timecourse_scores.csv"
CURATED = ["CBC", "CellCycle", "DTP", "Fetal", "MYC", "RSC"]

pytestmark = pytest.mark.skipif(
    not (C.TIMECOURSE_FILE.exists() and STORED.exists()
         and (C.PROCESSED / "final_model_genes.csv").exists()),
    reason="needs data/raw/ and a pipeline run",
)


def _timecourse():
    df = pd.read_csv(C.TIMECOURSE_FILE, sep="\t", low_memory=False)
    counts = (df.set_index(C.TIMECOURSE_GENE_COL)[list(C.TIMECOURSE_SAMPLES)]
              .apply(pd.to_numeric, errors="coerce"))
    if counts.index.duplicated().any():
        counts = counts.groupby(level=0).sum()
    counts = counts[(counts > 0).sum(axis=1) >= C.TIMECOURSE_MIN_DETECTED]
    return counts.T


def test_score_cohort_reproduces_timecourse_scores_in_either_orientation():
    expr = _timecourse()
    # The 413 model genes recover only ~64% on this platform, which would
    # (correctly) stop a real cohort; disable the stop to compare the rest.
    a, _, _ = score_cohort(expr, min_recovery=None, verbose=False)
    b, _, _ = score_cohort(expr.T, min_recovery=None, verbose=False)
    stored = pd.read_csv(STORED, index_col=0).loc[a.index]
    assert np.abs(a[CURATED].to_numpy() - stored[CURATED].to_numpy()).max() < 1e-8
    assert np.allclose(a[CURATED].to_numpy(), b[CURATED].to_numpy())
