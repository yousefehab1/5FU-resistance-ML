"""Load the modelling matrices.

Order matters: the median used to impute missing values is computed before
haematological lines are dropped. Reversing the two steps changes published
numbers (see docs/LIMITATIONS.md, item 7). `drop_haem` has no default so each
call site states which cohort it uses.
"""

from __future__ import annotations

import pandas as pd

from .config import HAEM, MAX_GENE_MISSING
from .paths import processed_dir

LINEAGE = "TCGA_DESC"


def surviving_genes(X: pd.DataFrame, max_missing: float = MAX_GENE_MISSING) -> pd.Index:
    """Columns whose missing fraction is at or below the threshold.

    Note `>` in the drop, so a column exactly at the threshold is KEPT. Every
    call site spells it `m[m > MAX_GENE_MISSING]`, so the boundary belongs to
    the survivors.
    """
    m = X.isna().mean()
    return X.columns[m <= max_missing]


def load_expression(
    label: str,
    *,
    impute: bool,
    max_missing: float = MAX_GENE_MISSING,
) -> pd.DataFrame:
    """
    The gene matrix for one screen, missingness-filtered.

    impute=False is not an optimisation. `scripts/14_drug_specificity.py`
    (`X1_cols = load_expression("GDSC1", impute=False).columns`) reads GDSC1
    purely to intersect column names, and imputing there would compute a
    36,000-column median that is then discarded.
    """
    X = pd.read_parquet(processed_dir() / f"X_{label}.parquet")
    X = X[surviving_genes(X, max_missing)]
    if impute and X.isna().any().any():
        X = X.fillna(X.median())
    return X


def load_screen(
    label: str,
    *,
    drop_haem: bool,
    impute: bool = True,
    max_missing: float = MAX_GENE_MISSING,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    (X, y, signature_scores) for one screen, row-aligned.

    `drop_haem` has no default on purpose: some stages keep haematological
    lines and some do not, so every caller must choose.
    """
    X = load_expression(label, impute=impute, max_missing=max_missing)
    y = pd.read_parquet(processed_dir() / f"y_{label}.parquet")
    s = pd.read_parquet(processed_dir() / f"signature_scores_{label}.parquet")

    if drop_haem:
        keep = (~y[LINEAGE].isin(HAEM)).to_numpy()
        X, y, s = X[keep], y[keep], s[keep]

    return X, y, s
