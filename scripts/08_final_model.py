"""The reported 5-FU model: repeated 5x5 CV with confidence intervals.

Adds lineage de-confounding (tissue means from training rows only), the
pan-solid to colorectal transfer, and results split by assay resolution.

Inputs:  data/processed/X_GDSC1.parquet, y_GDSC1.parquet, signature_scores_GDSC1.parquet
Outputs: data/processed/final_model_results.csv, final_model_genes.csv
Run:     python scripts/08_final_model.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
PROC = ROOT / "data" / "processed"
MODELS = ROOT / "models"

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import (CEILING_AUC, K_GENES, SEED, N_REPEATS, N_FOLDS,
                           MIN_LINEAGE_N, MODELED_MODULES as MODULES)
from fivefu.io import load_screen
from fivefu.modeling import deconfounded_gene_fit, elasticnet_pipeline, report_cv
from fivefu.report import banner
from fivefu.stats import bootstrap_pearson_draws


def main():
    banner("1. DATA")
    X, y, s = load_screen("GDSC1", drop_haem=True)
    lin = y.TCGA_DESC.to_numpy()
    t = y.AUC.to_numpy()
    cnt = pd.Series(lin).value_counts()
    big = set(cnt[cnt >= MIN_LINEAGE_N].index)
    keep = np.isin(lin, list(big))
    print(f"  solid lines: {len(y)}  |  in lineages with >= {MIN_LINEAGE_N} lines: {keep.sum()}")
    print(f"  de-confounded analyses use the {keep.sum()} lines in {len(big)} lineages")

    res = []

    banner("2. ITEM 1+2 -- REPEATED CV WITH CIs, RAW vs DE-CONFOUNDED")
    print(f"  {N_REPEATS} repeats x {N_FOLDS}-fold. CI is across repeats.\n")
    res.append(report_cv(X, t, lin, False, "transcriptome -> raw AUC"))
    res.append(report_cv(X[keep], t[keep], lin[keep], True,
                         "transcriptome -> AUC, lineage de-confounded"))
    res.append(report_cv(s[MODULES], t, lin, False, "modules -> raw AUC"))
    res.append(report_cv(s[MODULES][keep], t[keep], lin[keep], True,
                         "modules -> AUC, lineage de-confounded"))

    banner("3. ITEM 3 -- TRANSFER TO CRC")
    crc = (y.TCGA_DESC == "COREAD").to_numpy()
    print(f"  CRC n={crc.sum()}, pan-solid training n={(~crc).sum()}\n")

    # (a) train on CRC only, cross-validated -- the approach being replaced
    rs = []
    for rep in range(N_REPEATS):
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        p, tt = [], []
        for tr, te in kf.split(s[MODULES][crc]):
            mm = Pipeline([("sc", StandardScaler()), ("m", Ridge(1.0))]) \
                .fit(s[MODULES][crc].iloc[tr], t[crc][tr])
            p.append(mm.predict(s[MODULES][crc].iloc[te])); tt.append(t[crc][te])
        rs.append(stats.pearsonr(np.concatenate(tt), np.concatenate(p))[0])
    rs = np.array(rs)
    print(f"  {'(a) modules trained on CRC only':<44} r={rs.mean():+.3f}  "
          f"[{np.percentile(rs,2.5):+.3f}, {np.percentile(rs,97.5):+.3f}]")
    res.append(dict(model="CRC-only modules", r_mean=rs.mean(),
                    r_lo=np.percentile(rs, 2.5), r_hi=np.percentile(rs, 97.5), r_sd=rs.std()))

    # (b) transfer: fit on every solid line EXCEPT colorectal, predict CRC
    pan = elasticnet_pipeline().fit(X[~crc], t[~crc])
    pred = pan.predict(X[crc])
    r, p = stats.pearsonr(t[crc], pred)
    bs = bootstrap_pearson_draws(t[crc], pred)
    print(f"  {'(b) pan-solid model applied to CRC':<44} r={r:+.3f}  "
          f"[{np.percentile(bs,2.5):+.3f}, {np.percentile(bs,97.5):+.3f}]  p={p:.3f}")
    res.append(dict(model="transfer pan-solid -> CRC", r_mean=r,
                    r_lo=np.percentile(bs, 2.5), r_hi=np.percentile(bs, 97.5), r_sd=np.std(bs)))

    # (c) transfer prediction as a single feature alongside the modules
    F = s[MODULES][crc].copy()
    F["pan_pred"] = pred
    rs = []
    for rep in range(N_REPEATS):
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        pp, tt = [], []
        for tr, te in kf.split(F):
            mm = Pipeline([("sc", StandardScaler()), ("m", Ridge(1.0))]).fit(F.iloc[tr], t[crc][tr])
            pp.append(mm.predict(F.iloc[te])); tt.append(t[crc][te])
        rs.append(stats.pearsonr(np.concatenate(tt), np.concatenate(pp))[0])
    rs = np.array(rs)
    print(f"  {'(c) modules + transfer prediction':<44} r={rs.mean():+.3f}  "
          f"[{np.percentile(rs,2.5):+.3f}, {np.percentile(rs,97.5):+.3f}]")
    res.append(dict(model="CRC modules + transfer feature", r_mean=rs.mean(),
                    r_lo=np.percentile(rs, 2.5), r_hi=np.percentile(rs, 97.5), r_sd=rs.std()))

    banner("3b. STRATIFIED BY ASSAY RESOLUTION  (D6)")
    # Switching the target from LN_IC50 to AUC (D2) was meant to escape the
    # censoring problem: 53-70% of IC50s sit above the highest dose tested and
    # are curve-fit extrapolations rather than measurements.
    #
    # It did not escape it. AUC has its own ceiling: a line that is never killed
    # has AUC pushed against 1.0, so "not resolved by the assay" simply moved
    # coordinates. Pooling those lines with the responsive ones hides the fact
    # that the model cannot order them at all.
    #
    # So performance is reported separately either side of the ceiling. This is
    # the honest version of the headline number, and it is free.
    p_all = np.full(len(t), np.nan)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
        p_all[te] = elasticnet_pipeline().fit(X.iloc[tr], t[tr]).predict(X.iloc[te])

    print(f"  IC50 extrapolated beyond max dose: {100 * y.ic50_extrapolated.mean():.1f}% "
          f"of solid lines")
    print(f"  {'AUC band':<22} {'n':>5} {'% of lines':>11} {'CV r':>8}")
    for lab, msk in [("<= 0.90 (responsive)", t <= 0.90),
                     (f"0.90 - {CEILING_AUC}", (t > 0.90) & (t <= CEILING_AUC)),
                     (f"> {CEILING_AUC} (assay ceiling)", t > CEILING_AUC)]:
        if msk.sum() > 10:
            rr = stats.pearsonr(t[msk], p_all[msk])[0]
            print(f"  {lab:<22} {int(msk.sum()):>5} {100 * msk.mean():>10.1f}% {rr:>8.3f}")

    resp = t <= CEILING_AUC
    r_resp = stats.pearsonr(t[resp], p_all[resp])[0]
    r_ceil = stats.pearsonr(t[~resp], p_all[~resp])[0]
    r_pool = stats.pearsonr(t, p_all)[0]
    print(f"\n  pooled r = {r_pool:.3f}")
    print(f"  responsive lines only (AUC <= {CEILING_AUC}, n={int(resp.sum())}): r = {r_resp:.3f}")
    print(f"  ceiling lines only    (AUC >  {CEILING_AUC}, n={int((~resp).sum())}): r = {r_ceil:+.3f}")
    print()
    print("  READ THIS. All of the model's discriminative power sits in the")
    print("  responsive range. Among the lines at the assay ceiling it is")
    print("  indistinguishable from noise -- which is expected, because the")
    print("  experiment never resolved them. The pooled figure is not wrong, but")
    print("  it describes a model that ranks responsive lines and flags the rest")
    print("  as 'resistant beyond resolution'. Report it that way.")
    print()
    print("  This is also the case for the censored-regression extension (D6):")
    print("  a Tobit / AFT model would stop the fit chasing noise in ~29% of the")
    print("  data. Expect better-conditioned coefficients, not a higher r.")
    res.append(dict(model=f"stratified: responsive (AUC<={CEILING_AUC})", r_mean=r_resp,
                    r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
    res.append(dict(model=f"stratified: assay ceiling (AUC>{CEILING_AUC})", r_mean=r_ceil,
                    r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))

    banner("4. WHAT THE DE-CONFOUNDED MODEL SELECTS")
    g, fin = deconfounded_gene_fit(X, t, lin, keep)
    g.to_csv(PROC / "final_model_genes.csv", index=False)
    print(f"  {len(g)} non-zero genes of {K_GENES} selected\n")
    print("  RESISTANCE-associated (top 12):")
    print("   ", ", ".join(g.tail(12).iloc[::-1].gene))
    print("\n  SENSITIVITY-associated (top 12):")
    print("   ", ", ".join(g.head(12).gene))

    try:
        from fivefu import signatures as S
        sigs = S.load_signatures(X.columns, S.load_alias_map(required=False), verbose=False)
        print("\n  overlap with curated signatures:")
        mg = set(g.gene)
        for k_, v in sigs.items():
            o = mg & set(v)
            if o:
                print(f"    {k_:10} {len(o)}: {sorted(o)}")
    except Exception as e:
        print(f"  (signature overlap skipped: {e})")

    try:
        import joblib
        MODELS.mkdir(exist_ok=True)
        joblib.dump({"model": fin, "genes": list(X.columns),
                     "deconfounded": True, "lineages": sorted(big)},
                    MODELS / "final_deconfounded.joblib")
        print(f"\n  saved models/final_deconfounded.joblib")
    except ImportError:
        pass

    pd.DataFrame(res).to_csv(PROC / "final_model_results.csv", index=False)
    banner("DONE")
    print(f"  -> {PROC/'final_model_results.csv'}")
    print(f"  -> {PROC/'final_model_genes.csv'}")


if __name__ == "__main__":
    main()
