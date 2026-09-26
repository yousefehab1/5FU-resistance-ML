"""Is the DTP association specific to 5-FU?

Tests DTP against each drug after removing general chemosensitivity, compares
the drugs' selected genes, and tests the ribosome-biogenesis hypothesis
(5-FU tracking oxaliplatin more than cisplatin).

Inputs:  data/raw/GDSC2_fitted_dose_response_24Jul22.csv
         data/processed/targets/, signature_scores_GDSC2.parquet, multidrug_genes_*.csv
Outputs: data/processed/drug_specificity.csv, drug_specificity_gene_overlap.csv,
         regimen_scores.csv, reports/14_drug_specificity.md
Run:     python scripts/14_drug_specificity.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
RAW, PROC, DOCS = ROOT / "data" / "raw", ROOT / "data" / "processed", ROOT / "data" / "processed" / "reports"
TARGETS_DIR = PROC / "targets"

GDSC2_FILE = RAW / "GDSC2_fitted_dose_response_24Jul22.csv"

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import (N_FOLDS, SEED, HAEM, MIN_DRUG_LINES,
                           DRUGS as FOCAL)
from fivefu.io import load_expression
from fivefu.modeling import elasticnet_pipeline
from fivefu.report import banner
from fivefu.stats import (bootstrap_r_difference, hypergeometric_overlap,
                          partial_corr, residualize)

CRC = "COREAD"


def general_chemosensitivity(d_gdsc2, exclude_drug):
    """Mean z-scored AUC across GDSC2 drugs (>=300 solid lines), excluding one drug."""
    mat = d_gdsc2.pivot_table(index="SANGER_MODEL_ID", columns="DRUG_NAME",
                              values="AUC", aggfunc="mean")
    keep = mat.columns[mat.notna().sum() >= MIN_DRUG_LINES]
    mat = mat[keep]
    z = (mat - mat.mean()) / mat.std()
    return z.drop(columns=[exclude_drug], errors="ignore").mean(axis=1)


def main():
    banner("1. LOADING GDSC2 (uniform screen -- only one with all five drugs)")
    d2 = pd.read_csv(GDSC2_FILE, low_memory=False,
                     usecols=["SANGER_MODEL_ID", "DRUG_NAME", "AUC", "TCGA_DESC"])
    d2 = d2[~d2.TCGA_DESC.isin(HAEM)]
    print(f"  {len(d2):,} rows, {d2.DRUG_NAME.nunique()} drugs, solid lines only")

    # No haematological filter at load: those lines were already dropped from
    # the response table three lines above, and every X here is indexed through
    # that table.
    X2 = load_expression("GDSC2", impute=True)
    scores2 = pd.read_parquet(PROC / "signature_scores_GDSC2.parquet")
    module_cols = list(scores2.columns)
    print(f"  module scores available: {module_cols}")

    targets = {}
    for drug in FOCAL:
        p = TARGETS_DIR / f"{drug.replace(' ', '_')}_GDSC2.parquet"
        targets[drug] = pd.read_parquet(p)
        print(f"  {drug:16} GDSC2 target table n={len(targets[drug])}")

    banner("2. GENERAL CHEMOSENSITIVITY, PER FOCAL DRUG (self-excluded)")
    general = {drug: general_chemosensitivity(d2, drug) for drug in FOCAL}
    for drug in FOCAL:
        print(f"  {drug:16} n={general[drug].notna().sum():,} lines with the generic axis")

    banner("3a. IS DTP (AND EVERY OTHER MODULE) DRUG-SPECIFIC?  (COREAD only)")
    spec_rows = []
    for drug in FOCAL:
        y = targets[drug]
        crc = y[y.TCGA_DESC == CRC]
        both = crc.join(scores2, how="inner").join(general[drug].rename("general"), how="inner")
        msi = both.msi_status.isin(["MSI", "MSS"])
        print(f"\n  {drug}: COREAD n={len(both)}, with known MSI n={int(msi.sum())}")

        for mod in module_cols:
            m_all = both["general"].notna() & both[mod].notna() & both.AUC.notna()
            sub = both[m_all]
            if len(sub) < 15:
                continue
            r_raw, p_raw = partial_corr(sub[mod], sub.AUC, [])

            sub_msi = sub[sub.msi_status.isin(["MSI", "MSS"])]
            msi_dummy = (sub_msi.msi_status == "MSI").astype(float)
            if len(sub_msi) >= 15:
                r_msi, p_msi = partial_corr(sub_msi[mod], sub_msi.AUC, [msi_dummy])
            else:
                r_msi = p_msi = np.nan

            r_gen, p_gen = partial_corr(sub[mod], sub.AUC, [sub["general"]])

            if len(sub_msi) >= 15:
                r_both, p_both = partial_corr(sub_msi[mod], sub_msi.AUC,
                                              [sub_msi["general"], msi_dummy])
            else:
                r_both = p_both = np.nan

            spec_rows.append(dict(drug=drug, module=mod, n=len(sub), n_msi_known=len(sub_msi),
                                  r_raw=r_raw, p_raw=p_raw, r_msi_adj=r_msi, p_msi_adj=p_msi,
                                  r_general_adj=r_gen, p_general_adj=p_gen,
                                  r_both_adj=r_both, p_both_adj=p_both))
            flag = "*" if p_gen < 0.05 else " "
            print(f"    {mod:10} raw r={r_raw:+.3f}  MSI-adj r={r_msi:+.3f}  "
                  f"general-adj r={r_gen:+.3f}{flag}  both-adj r={r_both:+.3f}")

    spec_df = pd.DataFrame(spec_rows)

    dtp_gen = spec_df[(spec_df.module == "DTP")]
    dtp_sig = dtp_gen[dtp_gen.p_general_adj < 0.05]
    n_focal_with_dtp = dtp_sig.drug.nunique()
    outcome = ("general drug-tolerance programme" if n_focal_with_dtp >= 3
              else "mechanism-specific (5-FU only or fewer than 3 drugs)" if n_focal_with_dtp <= 1
              else "partial: some but not all drugs")
    print(f"\n  DTP survives general-chemosensitivity adjustment (p<0.05) for "
          f"{n_focal_with_dtp} of {len(FOCAL)} drugs: {sorted(dtp_sig.drug)}")
    print(f"  -> {outcome}")

    banner("3b. RIBOSOME-BIOGENESIS TEST: 5-FU vs OXALIPLATIN/CISPLATIN, RESIDUALISED")
    fu_y, ox_y, cis_y = targets["5-Fluorouracil"], targets["Oxaliplatin"], targets["Cisplatin"]

    def resid_auc(y, drug):
        """This drug's AUC with the general chemosensitivity score removed."""
        g = general[drug]
        both = pd.concat([y.AUC, g.rename("g")], axis=1).dropna()
        r = residualize(both.AUC.to_numpy(), [both.g.to_numpy()])
        return pd.Series(r, index=both.index)

    fu_res = resid_auc(fu_y, "5-Fluorouracil")
    ox_res = resid_auc(ox_y, "Oxaliplatin")
    cis_res = resid_auc(cis_y, "Cisplatin")

    fu_ox_raw = pd.concat([fu_y.AUC, ox_y.AUC], axis=1, join="inner", keys=["fu", "ox"]).dropna()
    fu_cis_raw = pd.concat([fu_y.AUC, cis_y.AUC], axis=1, join="inner", keys=["fu", "cis"]).dropna()
    r_raw_ox = stats.pearsonr(fu_ox_raw.fu, fu_ox_raw.ox)[0]
    r_raw_cis = stats.pearsonr(fu_cis_raw.fu, fu_cis_raw.cis)[0]
    print(f"  RAW      5-FU vs Oxaliplatin r={r_raw_ox:+.3f} (n={len(fu_ox_raw)})   "
          f"5-FU vs Cisplatin r={r_raw_cis:+.3f} (n={len(fu_cis_raw)})   "
          f"diff={r_raw_ox - r_raw_cis:+.3f}")

    fu_ox = pd.concat([fu_res, ox_res], axis=1, join="inner", keys=["fu", "ox"]).dropna()
    fu_cis = pd.concat([fu_res, cis_res], axis=1, join="inner", keys=["fu", "cis"]).dropna()
    r_adj_ox = stats.pearsonr(fu_ox.fu, fu_ox.ox)[0]
    r_adj_cis = stats.pearsonr(fu_cis.fu, fu_cis.cis)[0]
    diff = r_adj_ox - r_adj_cis
    print(f"  ADJUSTED 5-FU vs Oxaliplatin r={r_adj_ox:+.3f} (n={len(fu_ox)})   "
          f"5-FU vs Cisplatin r={r_adj_cis:+.3f} (n={len(fu_cis)})   diff={diff:+.3f}")

    def bootstrap_diff(seed):
        """The two arms resampled independently, at 2000 draws (fivefu.stats)."""
        return bootstrap_r_difference(fu_ox.fu, fu_ox.ox,
                                      fu_cis.fu, fu_cis.cis, seed=seed)

    d1 = bootstrap_diff(SEED)
    lo1, hi1 = np.percentile(d1, [2.5, 97.5])
    d2b = bootstrap_diff(SEED + 1)
    lo2, hi2 = np.percentile(d2b, [2.5, 97.5])
    print(f"\n  bootstrap CI on the difference (seed {SEED}):     [{lo1:+.3f}, {hi1:+.3f}]")
    print(f"  bootstrap CI on the difference (seed {SEED+1}, reproducibility check): [{lo2:+.3f}, {hi2:+.3f}]")
    agree = abs(lo1 - lo2) < 0.03 and abs(hi1 - hi2) < 0.03
    print(f"  reproducibility: {'OK, bounds agree within 0.03' if agree else 'DISAGREE -- CI is estimator noise, not signal'}")
    print(f"\n  Direction: {'confirms' if diff > 0 else 'contradicts'} the prediction "
          f"(5-FU should track oxaliplatin more than cisplatin). "
          f"{'CI excludes 0, difference is real.' if lo1 > 0 or hi1 < 0 else 'CI includes 0, not distinguishable from no difference.'}")

    banner("3c. GENE OVERLAP BETWEEN DRUG MODELS")
    # impute=False: only the surviving column names are wanted here, and
    # imputing would compute a 36,000-column median that is thrown away.
    X1_cols = load_expression("GDSC1", impute=False).columns
    universe = sorted(set(X1_cols) & set(X2.columns))
    print(f"  shared gene universe (both screens' post-filter columns): {len(universe):,}")

    gene_sets = {}
    for drug in FOCAL:
        g = pd.read_csv(PROC / f"multidrug_genes_{drug.replace(' ', '_')}.csv")
        genes = set(g.gene) & set(universe)
        gene_sets[drug] = genes
        print(f"  {drug:16} {len(g)} selected, {len(genes)} in shared universe")

    overlap_rows = []
    M = len(universe)
    for i, a in enumerate(FOCAL):
        for b in FOCAL[i + 1:]:
            Ka, Kb = len(gene_sets[a]), len(gene_sets[b])
            obs = len(gene_sets[a] & gene_sets[b])
            expected, fold, p = hypergeometric_overlap(Ka, Kb, obs, M)
            overlap_rows.append(dict(drug_a=a, drug_b=b, n_a=Ka, n_b=Kb, observed=obs,
                                     expected=expected, fold=fold, p_hypergeom=p))
            print(f"  {a:16} x {b:16} observed={obs:3d}  expected={expected:6.2f}  p={p:.2e}")
    overlap_df = pd.DataFrame(overlap_rows)

    banner("3d. FOLFOX / FOLFIRI REGIMEN COMPOSITES (GDSC2, raw AUC CV predictions)")
    kf = KFold(N_FOLDS, shuffle=True, random_state=SEED)
    preds = {}
    for drug in ["5-Fluorouracil", "Oxaliplatin", "SN-38"]:
        y = targets[drug]
        shared = y.index.intersection(X2.index)
        Xd, yd = X2.loc[shared], y.loc[shared, "AUC"]
        p_all = pd.Series(np.nan, index=shared)
        for tr, te in kf.split(Xd):
            m = elasticnet_pipeline().fit(Xd.iloc[tr], yd.iloc[tr])
            p_all.iloc[te] = m.predict(Xd.iloc[te])
        z = (p_all - p_all.mean()) / p_all.std()
        preds[drug] = z
        print(f"  {drug:16} n={len(z)} out-of-fold predictions z-scored")

    folfox = pd.concat([preds["5-Fluorouracil"], preds["Oxaliplatin"]], axis=1,
                       join="inner", keys=["z_5FU", "z_Ox"]).dropna()
    folfox["FOLFOX_score"] = folfox.mean(axis=1)
    folfiri = pd.concat([preds["5-Fluorouracil"], preds["SN-38"]], axis=1,
                        join="inner", keys=["z_5FU", "z_SN38"]).dropna()
    folfiri["FOLFIRI_score"] = folfiri.mean(axis=1)
    print(f"  FOLFOX  composite: n={len(folfox)}")
    print(f"  FOLFIRI composite: n={len(folfiri)}")

    regimen = folfox.join(folfiri, how="outer", lsuffix="_folfox", rsuffix="_folfiri")
    regimen.index.name = "SANGER_MODEL_ID"
    regimen.to_csv(PROC / "regimen_scores.csv")
    print(f"  -> {PROC / 'regimen_scores.csv'}")

    banner("WRITING drug_specificity.csv AND FINDINGS DOC")
    spec_df.to_csv(PROC / "drug_specificity.csv", index=False)
    overlap_df.to_csv(PROC / "drug_specificity_gene_overlap.csv", index=False)
    print(f"  -> {PROC / 'drug_specificity.csv'}")
    print(f"  -> {PROC / 'drug_specificity_gene_overlap.csv'}")

    lines = ["# Drug specificity and mechanism test\n",
             "All cross-drug work here runs on GDSC2, the only screen with all five "
             "drugs (matches doc 25's preliminary check). Full numbers in "
             "`data/processed/drug_specificity.csv`, "
             "`drug_specificity_gene_overlap.csv` and `regimen_scores.csv`.\n"]

    lines.append("## (a) Is DTP drug-specific?\n")
    lines.append("General-chemosensitivity-adjusted DTP-vs-AUC correlation, within COREAD:\n")
    lines.append("| Drug | n | r (raw) | r (MSI-adj) | r (general-adj) | r (both-adj) | p (general-adj) |")
    lines.append("|---|---|---|---|---|---|---|")
    for _, r in dtp_gen.iterrows():
        lines.append(f"| {r.drug} | {r.n} | {r.r_raw:+.3f} | {r.r_msi_adj:+.3f} | "
                     f"{r.r_general_adj:+.3f} | {r.r_both_adj:+.3f} | {r.p_general_adj:.4f} |")
    lines.append("")
    lines.append(f"**Outcome (pre-specified, both reportable): {outcome}.** "
                 f"DTP survives general-chemosensitivity adjustment (p<0.05) for "
                 f"{n_focal_with_dtp} of {len(FOCAL)} drugs: {sorted(dtp_sig.drug)}.\n")
    lines.append("Every other module's numbers are in the CSV; this section reports DTP "
                 "specifically since that is the project's central claim.\n")

    lines.append("## (b) The ribosome-biogenesis test\n")
    lines.append(f"Raw: 5-FU vs Oxaliplatin r={r_raw_ox:+.3f}, 5-FU vs Cisplatin r={r_raw_cis:+.3f}, "
                 f"difference {r_raw_ox - r_raw_cis:+.3f} (inflated by general chemosensitivity, not the test).\n")
    lines.append(f"**Adjusted (the real test): 5-FU vs Oxaliplatin r={r_adj_ox:+.3f}, "
                 f"5-FU vs Cisplatin r={r_adj_cis:+.3f}, difference {diff:+.3f}.**\n")
    lines.append(f"Bootstrap 95% CI on the difference: [{lo1:+.3f}, {hi1:+.3f}] "
                 f"(seed {SEED}), [{lo2:+.3f}, {hi2:+.3f}] (seed {SEED+1}, reproducibility "
                 f"check, {'agrees' if agree else 'DISAGREES -- treat with caution'}).\n")
    ci_verdict = ("excludes zero: the gap is real" if (lo1 > 0 or hi1 < 0)
                  else "includes zero: not distinguishable from no difference")
    lines.append(f"The CI {ci_verdict}. Direction "
                 f"{'confirms' if diff > 0 else 'contradicts'} the ribosome-biogenesis "
                 f"prediction (5-FU should track oxaliplatin more than cisplatin, since "
                 f"oxaliplatin kills via ribosome biogenesis stress and cisplatin via "
                 f"conventional DNA damage, Bruno and Ebert 2017).\n")

    lines.append("## (c) Gene overlap between drug models\n")
    lines.append("| Drug A | Drug B | n(A) | n(B) | Observed | Expected | Fold | p |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for _, r in overlap_df.iterrows():
        lines.append(f"| {r.drug_a} | {r.drug_b} | {r.n_a} | {r.n_b} | {r.observed} | "
                     f"{r.expected:.2f} | {r.fold:.2f} | {r.p_hypergeom:.2e} |")
    lines.append("")

    lines.append("## (d) FOLFOX / FOLFIRI regimen composites\n")
    lines.append(f"FOLFOX (5-FU + oxaliplatin): n={len(folfox)} cell lines scored, "
                 f"mean of z-scored out-of-fold CV predictions.\n")
    lines.append(f"FOLFIRI (5-FU + SN-38): n={len(folfiri)} cell lines scored, same method.\n")
    lines.append("Per-line scores in `data/processed/regimen_scores.csv`. These are what "
                 "the patient cohorts in 17_clinical_validation.py can actually test, since patients never "
                 "receive 5-FU alone.\n")

    lines.append("## Traps checked\n")
    lines.append("- Every cross-drug correlation reported both raw and residualised on "
                 "general chemosensitivity (not raw alone).")
    lines.append("- The ribosome-biogenesis comparison used a bootstrap CI on the "
                 "difference, with a reproducibility check against a second seed, "
                 "not just the two point estimates.")
    lines.append("- Outcome (a) is stated plainly per the pre-specification, without "
                 "hedging toward whichever answer seemed more interesting.")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "14_drug_specificity.md").write_text("\n".join(lines) + "\n")
    print(f"  -> {DOCS / '14_drug_specificity.md'}")

    banner("DONE")


if __name__ == "__main__":
    main()
