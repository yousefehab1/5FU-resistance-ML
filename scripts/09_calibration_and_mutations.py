"""Affine recalibration for the dashboard, and mutations as features. Prints only.

GDSC1 and GDSC2 have different AUC distributions, so a GDSC1 model is offset on
GDSC2. Fitting y = a*y_hat + b fixes the offset but cannot change r: it is a
display fix. The a and b printed here are hardcoded in 11.

Inputs:  data/processed/X_*.parquet, y_*.parquet; data/raw/mutations_summary_*.csv
Outputs: none (printed)
Run:     python scripts/09_calibration_and_mutations.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
RAW = ROOT / "data" / "raw"

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import SEED
from fivefu.io import load_screen
from fivefu.modeling import elasticnet_pipeline
from fivefu.report import banner

DRIVERS = ["TP53", "KRAS", "BRAF", "PIK3CA", "APC", "SMAD4", "PTEN", "NRAS"]


def report(name, t, p):
    r = stats.pearsonr(t, p)[0]
    r2 = 1 - ((t - p) ** 2).sum() / ((t - t.mean()) ** 2).sum()
    bias = p.mean() - t.mean()
    print(f"  {name:<42} r={r:.3f}  R2={r2:+.3f}  mean bias={bias:+.4f}")
    return r, r2


def main():

    banner("ITEM 5 -- AFFINE RECALIBRATION (dashboard only)")
    X1, y1, _ = load_screen("GDSC1", drop_haem=True)
    X2, y2, _ = load_screen("GDSC2", drop_haem=True)
    genes = sorted(set(X1.columns) & set(X2.columns))
    X1, X2 = X1[genes], X2[genes]
    t1, t2 = y1.AUC.to_numpy(), y2.AUC.to_numpy()

    print(f"  GDSC1 mean AUC {t1.mean():.3f} | GDSC2 mean AUC {t2.mean():.3f} "
          f"| offset {t2.mean() - t1.mean():+.3f}\n")

    m = elasticnet_pipeline().fit(X1, t1)
    pred2 = m.predict(X2)
    r_raw, r2_raw = report("uncalibrated on GDSC2", t2, pred2)

    # Split GDSC2: fit a and b on one half, evaluate on the other. Fitting and
    # scoring on the same rows would flatter the result.
    rng = np.random.default_rng(SEED)
    idx = rng.permutation(len(t2))
    fit_i, ev_i = idx[:len(idx) // 2], idx[len(idx) // 2:]
    a, b = np.polyfit(pred2[fit_i], t2[fit_i], 1)
    print(f"  calibration fitted on {len(fit_i)} lines: a={a:.3f}, b={b:.3f}")
    r_cal, r2_cal = report("calibrated, evaluated on held-out half",
                           t2[ev_i], a * pred2[ev_i] + b)

    r_ev = stats.pearsonr(t2[ev_i], pred2[ev_i])[0]
    print(f"\n  r on the evaluation half: {r_ev:.3f}  ->  r^2 = {r_ev ** 2:.3f}")
    print(f"  calibrated R2 = {r2_cal:+.3f}, i.e. r^2 as predicted by the algebra.")
    print("  Calibration removed the offset and changed r by exactly nothing.")
    print("  Use it in the dashboard so a displayed AUC is not systematically")
    print("  wrong; do NOT report it as a model improvement.")

    banner("ITEM 4 -- SOMATIC MUTATIONS")
    hits = sorted(RAW.glob("mutations*.csv")) + sorted(RAW.glob("*mutation*.csv"))
    if not hits:
        print("  PENDING -- no mutation file found in data/raw/")
        print()
        print("  Download from https://cellmodelpassports.sanger.ac.uk/downloads")
        print("  under 'Mutations' (mutations_summary_<date>.csv or")
        print("  mutations_all_<date>.csv), put it in data/raw/, and rerun.")
        print()
        print("  It is keyed by model_id = SANGER_MODEL_ID, so no mapping needed.")
        print(f"  Driver genes this script will test: {DRIVERS}")
        print()
        print("  Expected value: MSI status alone outperformed every expression")
        print("  module (D19), so genomic features are the most under-used source")
        print("  of signal here. TP53 is the strongest prior: HCT116 is TP53")
        print("  wild-type and p53 mediates the 5-FU stem-cell response (Cho 2020).")
        return

    path = hits[-1]
    print(f"  found {path.name}")
    mut = pd.read_csv(path, low_memory=False)
    print(f"  {mut.shape[0]:,} rows x {mut.shape[1]} columns")
    print(f"  columns: {list(mut.columns)[:12]}")

    idc = next((c for c in ["model_id", "model_name", "SANGER_MODEL_ID"]
                if c in mut.columns), None)
    gc = next((c for c in ["gene_symbol", "gene", "symbol"] if c in mut.columns), None)
    if not idc or not gc:
        print(f"  !! Could not find model-id and gene columns. Tell Claude the")
        print(f"     column list above and the script will be adjusted.")
        return

    # Binary gene x model mutation matrix for the driver panel.
    mut = mut[mut[gc].isin(DRIVERS)]
    flags = (mut.assign(v=1).pivot_table(index=idc, columns=gc, values="v",
                                         aggfunc="max").fillna(0))
    print(f"  driver mutation matrix: {flags.shape[0]:,} models x {flags.shape[1]} genes")

    for label, y_, s_ in [("GDSC1", y1, None), ("GDSC2", y2, None)]:
        common = y_.index.intersection(flags.index)
        f = flags.reindex(y_.index).fillna(0)
        auc = y_.AUC.to_numpy()
        crc = (y_.TCGA_DESC == "COREAD").to_numpy()
        print(f"\n  --- {label}: {len(common):,} models with mutation data")
        print(f"      {'gene':<8} {'n mut':>6}  {'all solid':>18}  {'COREAD':>18}")
        for g in [c for c in DRIVERS if c in f.columns]:
            v = f[g].to_numpy()
            r, p = stats.pearsonr(v, auc)
            if crc.sum() > 10 and v[crc].std() > 0:
                rc, pc = stats.pearsonr(v[crc], auc[crc])
                crc_s = f"r={rc:+.3f} p={pc:.3f}"
            else:
                crc_s = "n/a"
            print(f"      {g:<8} {int(v.sum()):>6}  r={r:+.3f} p={p:.3f}  {crc_s:>18}")

    banner("DONE")


if __name__ == "__main__":
    main()
