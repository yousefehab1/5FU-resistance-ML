"""
Two confounds on the 5-FU result, each measured and then removed.

MUTATIONS AND MSI
-----------------
Microsatellite instability is mechanistically tied to 5-FU handling
(mismatch-repair deficiency changes how cells process 5-FU-induced DNA
damage), and in COREAD it is associated with BOTH DTP and AUC -- the
definition of a confounder. This computes how much of the raw within-COREAD
DTP-resistance association (r=+0.344, stages/02_fu_model.py) survives after
adjusting for it, and runs the same adjustment for TP53 mutation status (the
strongest remaining mechanistic prior: HCT116 is TP53 wild-type and p53
mediates the 5-FU stem-cell response). MSI and mutation status are clinical
covariates, not features the final model uses, so the adjustment is done here
rather than in the model.

GENERAL CHEMOSENSITIVITY
------------------------
Cell lines differ in how sensitive they are to drugs IN GENERAL -- growth
rate, apoptotic priming, membrane transport, assay behaviour. If a large
share of 5-FU AUC variance is that generic axis, a model predicting 5-FU has
largely been predicting "is this a fragile cell line", not 5-FU biology. Same
class of question as the lineage confound, answered the same way: measure the
generic component, remove it, see what survives.

  1. Builds a cell line x drug AUC matrix from the full GDSC1 release.
  2. Defines general chemosensitivity as the mean z-scored AUC across
     drugs, and measures how much of 5-FU response that explains.
  3. Ranks which drugs 5-FU most resembles -- a mechanism sanity check.
     This surfaced an unexpected finding: 5-FU most resembles CX-5461, a
     pure RNA Pol I / ribosome-biogenesis inhibitor, while TYMS -- its own
     canonical DNA-directed target -- shows essentially no association
     with 5-FU response at all. Both are rows in persisted tables.
  4. Residualises 5-FU on general sensitivity and re-runs the model on the
     5-FU-SPECIFIC component.
  5. Tests whether the DTP association survives in that residual.

INPUTS   data/processed/{X,y,signature_scores}_{GDSC1,GDSC2}.parquet
         data/raw/mutations_summary_20260724.csv
         data/raw/GDSC1_fitted_dose_response_24Jul22.csv
OUTPUTS  data/processed/driver_mutation_correlations.csv
         data/processed/mutation_covariate_results.csv
         data/processed/multidrug_results.csv
         data/processed/drug_similarity.csv
         data/processed/gene_target_check.csv
         data/processed/dtp_specificity_check.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

import config as C
from lib.io import exclude_haem, load_mutation_flags, load_screen
from lib.modeling import repeated_cv
from lib.report import banner, write_and_report
from lib.stats import partial_corr, residualize


def mutation_covariates(scores):
    """Driver mutations vs AUC, then DTP adjusted for MSI and for TP53."""
    banner("MUTATIONS 1. DRIVER MUTATIONS vs AUC")
    driver_rows = []
    for label in [C.TRAIN, C.TEST]:
        y, _ = scores[label]
        flags = load_mutation_flags(C.DRIVER_GENES, model_ids=y.index)
        auc = y[C.TARGET].to_numpy()
        crc = (y[C.LINEAGE_COL] == C.CRC).to_numpy()

        print(f"\n  --- {label}: {int(flags.to_numpy().sum())} mutation calls "
              f"across {len(C.DRIVER_GENES)} driver genes")
        print(f"      {'gene':<8} {'n mut':>6}  {'all solid':>18}  {'COREAD':>18}")
        for gene in [g for g in C.DRIVER_GENES if g in flags.columns]:
            v = flags[gene].to_numpy()
            r, p = stats.pearsonr(v, auc)
            row = dict(screen=label, gene=gene, n_mutant=int(v.sum()),
                       r_all_solid=r, p_all_solid=p, r_coread=float("nan"), p_coread=float("nan"))
            if crc.sum() > 10 and v[crc].std() > 0:
                rc, pc = stats.pearsonr(v[crc], auc[crc])
                row["r_coread"], row["p_coread"] = rc, pc
                crc_s = f"r={rc:+.3f} p={pc:.3f}"
            else:
                crc_s = "n/a"
            print(f"      {gene:<8} {int(v.sum()):>6}  r={r:+.3f} p={p:.3f}  {crc_s:>18}")
            driver_rows.append(row)
    write_and_report(pd.DataFrame(driver_rows), C.PROCESSED / "driver_mutation_correlations.csv",
                      "driver_mutation_correlations.csv")

    banner("MUTATIONS 2. THE MSI CONFOUND")
    print("  MMR deficiency is mechanistically linked to 5-FU handling. In COREAD")
    print("  MSI status is associated with both AUC and DTP -- the definition of")
    print("  a confounder. This is the adjusted version of the r=+0.344 headline")
    print("  DTP-resistance association (stages/02_fu_model.py).\n")

    rows = []
    for label in [C.TRAIN, C.TEST]:
        y, s = scores[label]
        m = (y[C.LINEAGE_COL] == C.CRC) & y["msi_status"].notna()
        yy, ss = y[m], s[m.to_numpy()]
        auc = yy[C.TARGET].to_numpy()
        msi = (yy["msi_status"] == "MSI").astype(float).to_numpy()

        print(f"  --- {label} COREAD, n={int(m.sum())} "
              f"(MSI {int(msi.sum())}, MSS {int(len(msi) - msi.sum())})")
        for name, a, b in [("MSI vs AUC", msi, auc),
                            ("DTP vs AUC", ss["DTP"].to_numpy(), auc),
                            ("MSI vs DTP", msi, ss["DTP"].to_numpy())]:
            r, p = stats.pearsonr(a, b)
            print(f"      {name:<26} r={r:+.3f}  p={p:.4f}")
            rows.append(dict(screen=label, test=name, r=r, p=p))

        r, p = partial_corr(ss["DTP"].to_numpy(), auc, [msi])
        print(f"      {'DTP vs AUC | MSI':<26} r={r:+.3f}  p={p:.4f}   <-- adjusted")
        rows.append(dict(screen=label, test="DTP vs AUC | MSI", r=r, p=p))

        other_modules = [c for c in ss.columns if c != "DTP"]
        r, p = partial_corr(ss["DTP"].to_numpy(), auc,
                            [msi] + [ss[c].to_numpy() for c in other_modules])
        print(f"      {'DTP vs AUC | MSI + modules':<26} r={r:+.3f}  p={p:.4f}")
        rows.append(dict(screen=label, test="DTP vs AUC | MSI + modules", r=r, p=p))

        tp53 = load_mutation_flags(["TP53"], model_ids=yy.index)["TP53"].to_numpy()
        r, p = partial_corr(ss["DTP"].to_numpy(), auc, [tp53])
        print(f"      {'DTP vs AUC | TP53':<26} r={r:+.3f}  p={p:.4f}   <-- adjusted")
        rows.append(dict(screen=label, test="DTP vs AUC | TP53", r=r, p=p))
        print()

    write_and_report(pd.DataFrame(rows), C.PROCESSED / "mutation_covariate_results.csv",
                      "mutation_covariate_results.csv")


def general_chemosensitivity(X, y, s):
    """How much of 5-FU response is the generic drug-sensitivity axis, and
    what survives once it is removed. Solid-tumour GDSC1 inputs."""
    banner("CHEMOSENSITIVITY 1. BUILDING THE CELL LINE x DRUG MATRIX")
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

    banner("CHEMOSENSITIVITY 2. GENERAL CHEMOSENSITIVITY")
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

    banner("CHEMOSENSITIVITY 3. WHICH DRUGS DOES 5-FU MOST RESEMBLE?")
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

    banner("CHEMOSENSITIVITY 4. GENE-TARGET CHECK")
    print("  Is 5-FU's own DNA-directed target, TYMS, actually predictive of")
    print("  response?")
    rows = []
    for gene in C.PYRIMIDINE_PATHWAY_GENES:
        if gene not in X.columns:
            print(f"    {gene:<8} not in expression matrix -- skipped")
            continue
        gr, gp = stats.pearsonr(X[gene], y[C.TARGET])
        print(f"    {gene:<8} r={gr:+.3f}  p={gp:.3f}  (n={len(X):,})")
        rows.append(dict(gene=gene, r_with_auc=gr, p_value=gp, n=len(X)))
    write_and_report(pd.DataFrame(rows), C.PROCESSED / "gene_target_check.csv",
                      "gene_target_check.csv")
    print()
    print("  TYMS, thymidylate synthase, the direct molecular target of 5-FU, has")
    print("  essentially zero association with 5-FU sensitivity across this panel.")

    banner("CHEMOSENSITIVITY 5. MODELLING THE 5-FU-SPECIFIC COMPONENT")
    idx = y.index.intersection(both.index)
    X, y, s = X.loc[idx], y.loc[idx], s.loc[idx]
    fu_v = both.loc[idx, "fu"].to_numpy()
    gen_v = both.loc[idx, "gen"].to_numpy()
    print(f"  {len(idx):,} solid lines with expression and multi-drug data\n")

    # Residualise 5-FU on general sensitivity: what is left is the part of
    # 5-FU response NOT shared with every other drug.
    fu_res = residualize(fu_v, [gen_v])

    res = []
    res.append(repeated_cv(X, fu_v, name="expression -> raw 5-FU AUC"))
    res.append(repeated_cv(X, gen_v, name="expression -> GENERAL chemosensitivity"))
    res.append(repeated_cv(X, fu_res, name="expression -> 5-FU SPECIFIC (general removed)"))
    write_and_report(pd.DataFrame(res), C.PROCESSED / "multidrug_results.csv",
                      "multidrug_results.csv")

    banner("CHEMOSENSITIVITY 6. DOES THE DTP ASSOCIATION SURVIVE IN THE SPECIFIC COMPONENT?")
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


def main():
    X, y, s = load_screen(C.TRAIN)
    # The replication screen is needed for its response and scores only.
    y2, s2 = (pd.read_parquet(C.PROCESSED / f"{name}_{C.TEST}.parquet")
              for name in ("y", "signature_scores"))
    mutation_covariates({C.TRAIN: (y, s), C.TEST: (y2, s2)})

    _, y, X, s = exclude_haem(y, X, s)
    general_chemosensitivity(X, y, s)
    banner("DONE")


if __name__ == "__main__":
    main()
