"""
lib/report.py
=============

Console output and file-persistence helpers shared by every stage.

Numbers that are only printed to a terminal get hand-copied into documents
and go stale when the analysis is rerun. `write_and_report()` prevents
that: every stage that computes a reportable number persists it to a file
as part of printing it, and the dashboard reads only those files.
"""

import numpy as np
from scipy import stats

import config as C


def banner(text):
    print("\n" + "=" * 78 + f"\n{text}\n" + "=" * 78)


def report_metric(name, y_true, y_pred, ceiling=None):
    """
    Pearson r, Spearman rho, R2, RMSE for one prediction.

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


def write_report(name, text):
    """Write one generated markdown report to data/processed/reports/."""
    C.REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    path = C.REPORTS_DIR / name
    path.write_text(text)
    print(f"  -> {path}")
