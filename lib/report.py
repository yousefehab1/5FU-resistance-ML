"""
lib/report.py
=============

Console output and file-persistence helpers shared by every stage.

The original project's biggest correctness risk was not a modelling bug --
it was that several headline numbers (the DTP enrichment test, the
MSI-adjusted association, the drug-similarity ranking) were computed once,
printed to a terminal, and then hand-typed as literals into the dashboard.
If the upstream analysis was ever rerun with different data, those literals
would silently go stale. `write_and_report()` exists to make that structurally
impossible: every stage that computes a reportable number must persist it to
a file before (or as part of) printing it.
"""

import numpy as np
import pandas as pd
from scipy import stats


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def report_metric(name, y_true, y_pred, ceiling=None):
    """
    Pearson r, Spearman rho, R2, RMSE for one prediction -- consolidates the
    three near-identical evaluate()/metrics()/report() functions that were
    reimplemented independently across the original's scripts 03/07/09.

    Returns a dict (for persistence) and also prints one line.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    r, _ = stats.pearsonr(y_true, y_pred)
    rho, _ = stats.spearmanr(y_true, y_pred)
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    rmse = np.sqrt(ss_res / len(y_true))
    bias = y_pred.mean() - y_true.mean()

    line = f"  {name:<38} r={r:6.3f}  rho={rho:6.3f}  R2={r2:7.3f}  RMSE={rmse:.4f}"
    result = dict(name=name, pearson_r=r, spearman_rho=rho, r2=r2, rmse=rmse, bias=bias)
    if ceiling:
        line += f"   {100 * r / ceiling:5.1f}% of ceiling"
        result["pct_of_ceiling"] = 100 * r / ceiling
    print(line)
    return result


def write_and_report(df, path, label=None):
    """
    Persist a DataFrame to `path` and print a one-line confirmation. Every
    stage's terminal, reportable output should go through this rather than
    being left as a console-only print.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    print(f"  wrote {label or path.name}  ({df.shape[0]:,} rows) -> {path}")
