"""
stages/04_final_model.py
=========================

The final model: repeated 5x5 CV with confidence intervals (not a single
split), fold-internal lineage de-confounding applied to BOTH features and
target, transfer of a pan-solid model to CRC (in preference to training on
just the ~43-46 CRC lines directly), and assay-resolution stratification.

The de-confounding logic (lib.modeling.repeated_cv) is the single most
safety-critical piece of code in this project -- see its docstring for why
it is a hand-rolled loop and what leak it fixes.

Curated-signature overlap for the selected genes is NOT computed here. The
original project only ever printed raw gene-name overlaps with no
statistical test; that is now stages/05_enrichment.py's job, done properly
with a hypergeometric test against the model's actual gene universe.

INPUTS   data/processed/{X,y,signature_scores}_GDSC1.parquet
OUTPUTS  data/processed/final_model_results.csv
         data/processed/final_model_genes.csv
         models/final_deconfounded.joblib
"""

import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import joblib
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

import config as C
from lib.io import exclude_haem, load_screen
from lib.modeling import elasticnet_pipeline, oof_predictions, repeated_cv, transfer_to_crc
from lib.report import banner, write_and_report

warnings.filterwarnings("ignore")


def main():
    banner("1. DATA")
    X, y, s = load_screen(C.TRAIN)
    _, y, X, s = exclude_haem(y, X, s)
    lin = y[C.LINEAGE_COL].to_numpy()
    t = y[C.TARGET].to_numpy()

    cnt = pd.Series(lin).value_counts()
    big = set(cnt[cnt >= C.MIN_LINEAGE_N].index)
    keep = np.isin(lin, list(big))
    print(f"  solid lines: {len(y)}  |  in lineages with >= {C.MIN_LINEAGE_N} lines: {keep.sum()}")
    print(f"  de-confounded analyses use the {keep.sum()} lines in {len(big)} lineages")

    res = []

    banner("2. REPEATED CV WITH CIs, RAW vs DE-CONFOUNDED")
    print(f"  {C.N_REPEATS} repeats x {C.N_FOLDS}-fold. CI is across repeats.\n")
    res.append(repeated_cv(X, t, lin, False, "transcriptome -> raw AUC"))
    res.append(repeated_cv(X[keep], t[keep], lin[keep], True,
                            "transcriptome -> AUC, lineage de-confounded"))
    res.append(repeated_cv(s[C.MODELED_MODULES], t, lin, False, "modules -> raw AUC"))
    res.append(repeated_cv(s[C.MODELED_MODULES][keep], t[keep], lin[keep], True,
                            "modules -> AUC, lineage de-confounded"))

    banner("3. TRANSFER TO CRC")
    crc = (y[C.LINEAGE_COL] == C.CRC).to_numpy()
    print(f"  CRC n={crc.sum()}, pan-solid training n={(~crc).sum()}\n")
    crc_res, _ = transfer_to_crc(np.asarray(X, dtype=float), t, s, crc, C.MODELED_MODULES)
    res.extend(crc_res)

    banner("3b. STRATIFIED BY ASSAY RESOLUTION")
    # Switching the target from LN_IC50 to AUC was meant to escape censoring
    # (53-70% of IC50s are curve-fit extrapolations beyond the highest dose
    # tested). It did not escape it: a line that is never killed has AUC
    # pushed against 1.0, so "not resolved by the assay" simply moved
    # coordinates. Pooling those lines with responsive ones hides that the
    # model cannot order them at all -- so report the two ranges separately.
    p_all = oof_predictions(X, t, model_factory=elasticnet_pipeline,
                             n_folds=C.N_FOLDS, seed=C.SEED)

    print(f"  IC50 extrapolated beyond max dose: {100 * y.ic50_extrapolated.mean():.1f}% "
          f"of solid lines")
    print(f"  {'AUC band':<22} {'n':>5} {'% of lines':>11} {'CV r':>8}")
    for lab, msk in [("<= 0.90 (responsive)", t <= 0.90),
                     ("0.90 - 0.95", (t > 0.90) & (t <= 0.95)),
                     ("> 0.95 (assay ceiling)", t > 0.95)]:
        if msk.sum() > 10:
            rr = stats.pearsonr(t[msk], p_all[msk])[0]
            print(f"  {lab:<22} {int(msk.sum()):>5} {100 * msk.mean():>10.1f}% {rr:>8.3f}")

    resp = t <= 0.95
    r_resp = stats.pearsonr(t[resp], p_all[resp])[0]
    r_ceil = stats.pearsonr(t[~resp], p_all[~resp])[0]
    r_pool = stats.pearsonr(t, p_all)[0]
    print(f"\n  pooled r = {r_pool:.3f}")
    print(f"  responsive lines only (AUC <= 0.95, n={int(resp.sum())}): r = {r_resp:.3f}")
    print(f"  ceiling lines only    (AUC >  0.95, n={int((~resp).sum())}): r = {r_ceil:+.3f}")
    print()
    print("  All of the model's discriminative power sits in the responsive range.")
    print("  Among lines at the assay ceiling it is indistinguishable from noise --")
    print("  expected, since the experiment never resolved them. Report the pooled")
    print("  figure as ranking responsive lines and flagging the rest as 'resistant")
    print("  beyond resolution', not as a single unqualified correlation.")
    res.append(dict(model="stratified: responsive (AUC<=0.95)", r_mean=r_resp,
                     r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
    res.append(dict(model="stratified: assay ceiling (AUC>0.95)", r_mean=r_ceil,
                     r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))

    banner("4. WHAT THE DE-CONFOUNDED MODEL SELECTS")
    gm = pd.DataFrame(np.asarray(X[keep])).groupby(lin[keep]).transform("mean").to_numpy()
    Xd = np.asarray(X[keep]) - gm
    ys = pd.Series(t[keep]).groupby(lin[keep])
    yd = ((t[keep] - ys.transform("mean").to_numpy()) /
          ys.transform(lambda v: v.std(ddof=1)).to_numpy())
    ok = np.isfinite(yd)
    fin = elasticnet_pipeline().fit(Xd[ok], yd[ok])
    sel = np.array(X.columns)[fin.named_steps["select"].get_support()]
    co = fin.named_steps["model"].coef_
    nz = co != 0
    g = pd.DataFrame({"gene": sel[nz], "coef": co[nz]}).sort_values("coef")
    write_and_report(g, C.PROCESSED / "final_model_genes.csv", "final_model_genes.csv")
    print(f"  {nz.sum()} non-zero genes of {C.K_GENES} selected\n")
    print("  RESISTANCE-associated (top 12):")
    print("   ", ", ".join(g.tail(12).iloc[::-1].gene))
    print("\n  SENSITIVITY-associated (top 12):")
    print("   ", ", ".join(g.head(12).gene))

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fin, "genes": list(X.columns), "deconfounded": True,
                 "lineages": sorted(big)}, C.MODELS_DIR / "final_deconfounded.joblib")
    print(f"\n  saved models/final_deconfounded.joblib")

    write_and_report(pd.DataFrame(res), C.PROCESSED / "final_model_results.csv",
                      "final_model_results.csv")
    banner("DONE")


if __name__ == "__main__":
    main()
