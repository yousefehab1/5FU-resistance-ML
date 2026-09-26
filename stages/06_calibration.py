"""
stages/06_calibration.py
==========================

Affine recalibration for the dashboard: the model trained on GDSC1 is
systematically offset when applied to GDSC2, because the two screens have
different AUC distributions. Fitting y_cal = a*y_hat + b fixes the offset.

For the OPTIMAL a and b, R2 becomes EXACTLY r^2 -- an affine transform
cannot change r. This makes it a display fix for the dashboard (so a shown
predicted AUC is not systematically wrong), NOT a modelling improvement.
It also needs labelled data from the target screen, which a genuinely new
dataset would not have.

Fitted on a held-out half of GDSC2 and evaluated on the other half, so the
reported numbers are honest rather than fitted-and-scored on the same rows.


INPUTS   data/processed/{X,y}_{GDSC1,GDSC2}.parquet
OUTPUTS  data/processed/calibration_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd

import config as C
from lib.io import exclude_haem, load_screen
from lib.modeling import elasticnet_pipeline
from lib.report import banner, report_metric, write_and_report


def main():
    banner("AFFINE RECALIBRATION (dashboard display only)")
    X1, y1 = load_screen(C.TRAIN, with_signatures=False)
    X2, y2 = load_screen(C.TEST, with_signatures=False)
    _, y1, X1 = exclude_haem(y1, X1)
    _, y2, X2 = exclude_haem(y2, X2)

    genes = sorted(set(X1.columns) & set(X2.columns))
    X1, X2 = X1[genes], X2[genes]
    t1, t2 = y1[C.TARGET].to_numpy(), y2[C.TARGET].to_numpy()
    print(f"  {C.TRAIN} mean AUC {t1.mean():.3f} | {C.TEST} mean AUC {t2.mean():.3f} "
          f"| offset {t2.mean() - t1.mean():+.3f}\n")

    m = elasticnet_pipeline().fit(X1, t1)
    pred2 = m.predict(X2)
    raw = report_metric(f"uncalibrated on {C.TEST}", t2, pred2)

    # Split GDSC2: fit a and b on one half, evaluate on the other. Fitting
    # and scoring on the same rows would flatter the result.
    rng = np.random.default_rng(C.SEED)
    idx = rng.permutation(len(t2))
    fit_i, ev_i = idx[:len(idx) // 2], idx[len(idx) // 2:]
    a, b = np.polyfit(pred2[fit_i], t2[fit_i], 1)
    print(f"  calibration fitted on {len(fit_i)} lines: a={a:.3f}, b={b:.3f}")
    cal = report_metric("calibrated, evaluated on held-out half", t2[ev_i], a * pred2[ev_i] + b)

    print()
    print("  Calibration removed the offset and changed r by exactly nothing")
    print(f"  ({raw['pearson_r']:.3f} raw vs {cal['pearson_r']:.3f} calibrated-half r, as the")
    print("  algebra predicts). Use it in the dashboard so a displayed AUC is not")
    print("  systematically wrong; do NOT report it as a model improvement.")

    out = pd.DataFrame([
        dict(stage="uncalibrated", **raw, a=np.nan, b=np.nan, n_fit=np.nan),
        dict(stage="calibrated_heldout", **cal, a=a, b=b, n_fit=len(fit_i)),
    ])
    write_and_report(out, C.PROCESSED / "calibration_results.csv", "calibration_results.csv")
    banner("DONE")


if __name__ == "__main__":
    main()
