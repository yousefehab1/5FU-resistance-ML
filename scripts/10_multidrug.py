"""Split 5-FU response into general chemosensitivity and a 5-FU-specific part.

General chemosensitivity is the mean z-scored AUC across all GDSC1 drugs. The
model and the DTP association are rerun on the residual.

Inputs:  data/raw/GDSC1_fitted_dose_response_24Jul22.csv
         data/processed/X_GDSC1.parquet, y_GDSC1.parquet, signature_scores_GDSC1.parquet
Outputs: data/processed/multidrug_results.csv, fu_drug_similarity.csv, dtp_specificity.csv
Run:     python scripts/10_multidrug.py
"""

from pathlib import Path
import sys
import warnings

import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
GDSC1 = RAW / "GDSC1_fitted_dose_response_24Jul22.csv"

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import HAEM, MIN_DRUG_LINES
from fivefu.io import load_screen
from fivefu.modeling import repeated_cv, summarise
from fivefu.report import banner
from fivefu.stats import residualize

FU = "5-Fluorouracil"


def cv_row(X, y, name, res):
    """Repeated k-fold CV with no lineage de-confounding, printed at 10's width.

    This was a third copy of fivefu.modeling.repeated_cv, the library's
    deconfound=False path. The library prints nothing, so 10's line format and
    its result row stay here, as 13's do.
    """
    mean, lo, hi, _ = summarise(repeated_cv(X, y, None, deconfound=False))
    print(f"  {name:<46} r={mean:.3f}  [{lo:.3f}, {hi:.3f}]")
    res.append(dict(model=name, r=mean, lo=lo, hi=hi))


def main():

    banner("1. BUILDING THE CELL LINE x DRUG MATRIX")
    d = pd.read_csv(GDSC1, low_memory=False,
                    usecols=["SANGER_MODEL_ID", "DRUG_NAME", "AUC", "TCGA_DESC"])
    print(f"  {len(d):,} dose-response curves, {d.DRUG_NAME.nunique()} drugs, "
          f"{d.SANGER_MODEL_ID.nunique():,} cell lines")

    d = d[~d.TCGA_DESC.isin(HAEM)]
    mat = d.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_NAME",
                        values="AUC", aggfunc="mean")
    keep = mat.columns[mat.notna().sum() >= MIN_DRUG_LINES]
    mat = mat[keep]
    print(f"  solid lines only, drugs screened on >= {MIN_DRUG_LINES} lines: "
          f"{mat.shape[0]:,} x {mat.shape[1]}")

    banner("2. GENERAL CHEMOSENSITIVITY")
    # z-score each drug so drugs on different scales contribute equally, then
    # average across drugs per cell line. Low = sensitive to everything.
    z = (mat - mat.mean()) / mat.std()
    general = z.drop(columns=[FU], errors="ignore").mean(axis=1)
    print(f"  defined as mean z-scored AUC across {z.shape[1] - 1} other drugs")
    print(f"  computed for {general.notna().sum():,} cell lines")

    fu = mat[FU]
    both = pd.concat([fu, general], axis=1, keys=["fu", "gen"]).dropna()
    r, p = stats.pearsonr(both.fu, both.gen)
    print(f"\n  5-FU AUC vs general chemosensitivity: r = {r:.3f}  "
          f"(r2 = {r ** 2:.3f}, p = {p:.1e}, n = {len(both):,})")
    print(f"  -> {100 * r ** 2:.0f}% of 5-FU AUC variance is the generic axis")

    banner("3. WHICH DRUGS DOES 5-FU MOST RESEMBLE?")
    cor = z.corrwith(z[FU]).drop(FU).sort_values(ascending=False)
    print("  most similar:")
    for k, v in cor.head(12).items():
        print(f"    {k:<28} r={v:+.3f}")
    print("  least similar:")
    for k, v in cor.tail(5).items():
        print(f"    {k:<28} r={v:+.3f}")
    print(f"\n  median correlation with all other drugs: {cor.median():+.3f}")
    cor.rename_axis("drug").rename("r").reset_index().to_csv(
        PROC / "fu_drug_similarity.csv", index=False)

    banner("4. MODELLING THE 5-FU-SPECIFIC COMPONENT")
    X, y, s = load_screen("GDSC1", drop_haem=True)

    idx = y.index.intersection(both.index)
    X, y, s = X.loc[idx], y.loc[idx], s.loc[idx]
    fu_v = both.loc[idx, "fu"].to_numpy()
    gen_v = both.loc[idx, "gen"].to_numpy()
    print(f"  {len(idx):,} solid lines with expression and multi-drug data\n")

    # Residualise 5-FU on general sensitivity: what is left is the part of
    # 5-FU response NOT shared with every other drug.
    fu_res = residualize(fu_v, [gen_v])

    res = []
    cv_row(X, fu_v, "expression -> raw 5-FU AUC", res)
    cv_row(X, gen_v, "expression -> GENERAL chemosensitivity", res)
    cv_row(X, fu_res, "expression -> 5-FU SPECIFIC (general removed)", res)

    banner("5. DOES THE DTP ASSOCIATION SURVIVE IN THE SPECIFIC COMPONENT?")
    crc = (y.TCGA_DESC == "COREAD").to_numpy()
    print(f"  COREAD n={crc.sum()}  (higher = more resistant)\n")
    spec = []
    for nm, tgt in [("raw 5-FU AUC", fu_v),
                    ("general chemosensitivity", gen_v),
                    ("5-FU specific (general removed)", fu_res)]:
        rr, pp = stats.pearsonr(s.loc[crc, "DTP"], tgt[crc])
        print(f"    DTP vs {nm:<34} r={rr:+.3f}  p={pp:.4f}")
        spec.append(dict(target=nm, n=int(crc.sum()), r=rr, p=pp))

    pd.DataFrame(res).to_csv(PROC / "multidrug_results.csv", index=False)
    pd.DataFrame(spec).to_csv(PROC / "dtp_specificity.csv", index=False)
    banner("DONE")
    print(f"  -> {PROC / 'multidrug_results.csv'}")


if __name__ == "__main__":
    main()
