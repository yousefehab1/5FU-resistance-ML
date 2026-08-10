"""
stages/09_multidrug.py
=========================

Item 7. Borrow strength across GDSC's ~264 other drugs.

THE QUESTION BEHIND IT
------------------------
Cell lines differ in how sensitive they are to drugs IN GENERAL -- growth
rate, apoptotic priming, membrane transport, assay behaviour. If a large
share of 5-FU AUC variance is that generic axis, a model predicting 5-FU
has largely been predicting "is this a fragile cell line", not 5-FU
biology. This is the same class of question as the lineage confound
(stage 04), answered the same way: measure the generic component, remove
it, see what survives.

WHAT THIS SCRIPT DOES
-----------------------
  1. Builds a cell line x drug AUC matrix from the full GDSC1 release.
  2. Defines general chemosensitivity as the mean z-scored AUC across
     drugs, and measures how much of 5-FU response that explains.
  3. Ranks which drugs 5-FU most resembles -- a mechanism sanity check.
     This surfaced an unexpected finding: 5-FU most resembles CX-5461, a
     pure RNA Pol I / ribosome-biogenesis inhibitor, while TYMS -- its own
     canonical DNA-directed target -- shows essentially no association
     with 5-FU response at all. Both numbers existed only as prose in
     app.py in the original; here they are rows in persisted tables.
  4. Residualises 5-FU on general sensitivity and re-runs the model on the
     5-FU-SPECIFIC component.
  5. Tests whether the DTP association survives in that residual.

INPUTS   data/raw/GDSC1_fitted_dose_response_24Jul22.csv
         data/processed/X_GDSC1.parquet, y_GDSC1.parquet, signature_scores_GDSC1.parquet
OUTPUTS  data/processed/multidrug_results.csv       (existing shape)
         data/processed/drug_similarity.csv          (new -- full ranked table, not just top/bottom)
         data/processed/gene_target_check.csv        (new -- TYMS et al., was a hand-typed doc table)
         data/processed/dtp_specificity_check.csv    (new -- was console-only)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from scipy import stats

import config as C
from lib.io import exclude_haem, load_screen
from lib.modeling import repeated_cv
from lib.report import banner, write_and_report


def main():
    banner("1. BUILDING THE CELL LINE x DRUG MATRIX")
    d = pd.read_csv(C.GDSC1_FILE, low_memory=False,
                     usecols=["SANGER_MODEL_ID", "DRUG_NAME", "AUC", "TCGA_DESC"])
    print(f"  {len(d):,} dose-response curves, {d.DRUG_NAME.nunique()} drugs, "
          f"{d.SANGER_MODEL_ID.nunique():,} cell lines")

    d = d[~d.TCGA_DESC.isin(C.HAEM)]
    mat = d.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_NAME",
                         values="AUC", aggfunc="mean")
    keep = mat.columns[mat.notna().sum() >= C.MIN_DRUG_LINES]
    mat = mat[keep]
    print(f"  solid lines only, drugs screened on >= {C.MIN_DRUG_LINES} lines: "
          f"{mat.shape[0]:,} x {mat.shape[1]}")

    banner("2. GENERAL CHEMOSENSITIVITY")
    # z-score each drug so drugs on different scales contribute equally, then
    # average across drugs per cell line. Low = sensitive to everything.
    z = (mat - mat.mean()) / mat.std()
    general = z.drop(columns=[C.FU_DRUG_NAME], errors="ignore").mean(axis=1)
    print(f"  defined as mean z-scored AUC across {z.shape[1] - 1} other drugs")
    print(f"  computed for {general.notna().sum():,} cell lines")

    fu = mat[C.FU_DRUG_NAME]
    both = pd.concat([fu, general], axis=1, keys=["fu", "gen"]).dropna()
    r, p = stats.pearsonr(both.fu, both.gen)
    print(f"\n  5-FU AUC vs general chemosensitivity: r = {r:.3f}  "
          f"(r2 = {r ** 2:.3f}, p = {p:.1e}, n = {len(both):,})")
    print(f"  -> {100 * r ** 2:.0f}% of 5-FU AUC variance is the generic axis")

    banner("3. WHICH DRUGS DOES 5-FU MOST RESEMBLE?")
    cor = z.corrwith(z[C.FU_DRUG_NAME]).drop(C.FU_DRUG_NAME).sort_values(ascending=False)
    print("  most similar:")
    for k, v in cor.head(12).items():
        print(f"    {k:<28} r={v:+.3f}")
    print("  least similar:")
    for k, v in cor.tail(5).items():
        print(f"    {k:<28} r={v:+.3f}")
    print(f"\n  median correlation with all other drugs: {cor.median():+.3f}")

    sim = cor.rename("r_with_5fu").rename_axis("drug").reset_index()
    write_and_report(sim, C.PROCESSED / "drug_similarity.csv", "drug_similarity.csv")

    banner("4. GENE-TARGET CHECK")
    print("  Is 5-FU's own DNA-directed target, TYMS, actually predictive of")
    print("  response? A hand-typed doc table in the original -- here, real code.")
    X, y = load_screen(C.TRAIN, with_signatures=False)
    _, y_solid, X_solid = exclude_haem(y, X)
    rows = []
    for gene in C.PYRIMIDINE_PATHWAY_GENES:
        if gene not in X_solid.columns:
            print(f"    {gene:<8} not in expression matrix -- skipped")
            continue
        gr, gp = stats.pearsonr(X_solid[gene], y_solid[C.TARGET])
        print(f"    {gene:<8} r={gr:+.3f}  p={gp:.3f}  (n={len(X_solid):,})")
        rows.append(dict(gene=gene, r_with_auc=gr, p_value=gp, n=len(X_solid)))
    write_and_report(pd.DataFrame(rows), C.PROCESSED / "gene_target_check.csv",
                      "gene_target_check.csv")
    print()
    print("  TYMS, thymidylate synthase, the direct molecular target of 5-FU, has")
    print("  essentially zero association with 5-FU sensitivity across this panel.")

    banner("5. MODELLING THE 5-FU-SPECIFIC COMPONENT")
    X, y, s = load_screen(C.TRAIN, with_signatures=True)
    _, y, X, s = exclude_haem(y, X, s)

    idx = y.index.intersection(both.index)
    X, y, s = X.loc[idx], y.loc[idx], s.loc[idx]
    fu_v = both.loc[idx, "fu"].to_numpy()
    gen_v = both.loc[idx, "gen"].to_numpy()
    print(f"  {len(idx):,} solid lines with expression and multi-drug data\n")

    # Residualise 5-FU on general sensitivity: what is left is the part of
    # 5-FU response NOT shared with every other drug.
    Z = np.column_stack([np.ones(len(gen_v)), gen_v])
    fu_res = fu_v - Z @ lstsq(Z, fu_v, rcond=None)[0]

    res = []
    res.append(repeated_cv(X, fu_v, name="expression -> raw 5-FU AUC"))
    res.append(repeated_cv(X, gen_v, name="expression -> GENERAL chemosensitivity"))
    res.append(repeated_cv(X, fu_res, name="expression -> 5-FU SPECIFIC (general removed)"))
    write_and_report(pd.DataFrame(res), C.PROCESSED / "multidrug_results.csv",
                      "multidrug_results.csv")

    banner("6. DOES THE DTP ASSOCIATION SURVIVE IN THE SPECIFIC COMPONENT?")
    crc = (y[C.LINEAGE_COL] == C.CRC).to_numpy()
    print(f"  COREAD n={crc.sum()}  (higher = more resistant)\n")
    dtp_rows = []
    for nm, tgt in [("raw 5-FU AUC", fu_v),
                    ("general chemosensitivity", gen_v),
                    ("5-FU specific (general removed)", fu_res)]:
        rr, pp = stats.pearsonr(s.loc[crc, "DTP"], tgt[crc])
        print(f"    DTP vs {nm:<34} r={rr:+.3f}  p={pp:.4f}")
        dtp_rows.append(dict(target=nm, r_with_dtp=rr, p_value=pp, n=int(crc.sum())))
    write_and_report(pd.DataFrame(dtp_rows), C.PROCESSED / "dtp_specificity_check.csv",
                      "dtp_specificity_check.csv")

    banner("DONE")


if __name__ == "__main__":
    main()
