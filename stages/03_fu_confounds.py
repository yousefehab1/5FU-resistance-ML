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

MSI AS AN EFFECT MODIFIER, AND MMR COUPLING (post hoc)
------------------------------------------------------
Adjusting for MSI asks whether MSI explains the DTP association. These ask
whether the association DIFFERS by MSI status: DTP vs AUC within MSS and
within MSI lines, an AUC ~ DTP * MSI interaction (COREAD, and pan-solid with
lineage fixed effects, where 40 more MSI lines add power), and the pan-solid
transcriptome model scored on MSS colorectal lines alone. Then: is DTP tied
to mismatch-repair expression or mutational burden? MSI drives both, so only
the within-MSS and MSI-adjusted rows can separate a coupling from MSI again.

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
         data/processed/msi_effect_modification.csv
         data/processed/dtp_mmr_coupling.csv
         data/processed/multidrug_results.csv
         data/processed/drug_similarity.csv
         data/processed/gene_target_check.csv
         data/processed/dtp_specificity_check.csv
         data/processed/module_screen_discrepancy.csv
         data/processed/penalty_scale_sensitivity.csv
"""

import sys
from pathlib import Path
import warnings

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf
from scipy import stats

import config as C
from lib.io import exclude_haem, load_mutation_flags, load_screen, solid_screen
from lib.modeling import (cv_repeats, elasticnet_pipeline, fixed_elasticnet_pipeline,
                          oof_predictions, repeated_cv, scaled_target)
from lib.report import banner, write_and_report
from lib.stats import (bh_q, bootstrap_r, bootstrap_r_difference, fisher_ci,
                       fisher_z_difference_p, partial_corr, residualize)


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


def _zs(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / v.std(ddof=0)


def _lineage_dummies(lin):
    d = pd.get_dummies(pd.Series(lin), drop_first=True, dtype=float)
    return [d[c].to_numpy() for c in d.columns]


def _r_row(x, y, controls=()):
    """r, its 95% CI and p: a line bootstrap for a plain correlation, Fisher's
    z (with one degree of freedom per control) for a partial one."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    if not controls:
        r, p, lo, hi = bootstrap_r(x, y)
        return dict(n=len(x), r=r, lo=lo, hi=hi, p=p, ci="line bootstrap")
    r, p = partial_corr(x, y, list(controls))
    lo, hi = fisher_ci(r, len(x), len(controls))
    return dict(n=len(x), r=r, lo=lo, hi=hi, p=p, ci="Fisher z")


def msi_effect_modification(data):
    """
    Does the DTP association differ between MSI and MSS lines? `data` is
    {screen: (X, y, s)}, solid lines. Every row is post hoc.
    """
    banner("MSI AS AN EFFECT MODIFIER (post hoc)")
    rows = []
    for label, (X, y, s) in data.items():
        auc, lin = y[C.TARGET].to_numpy(), y[C.LINEAGE_COL].to_numpy()
        # Pan-solid rows need the lineage label too (2 lines lack one).
        known = (y["msi_status"].notna() & y[C.LINEAGE_COL].notna()).to_numpy()
        msi = (y["msi_status"] == "MSI").to_numpy()
        crc = lin == C.CRC
        dtp = s["DTP"].to_numpy()
        mss_c, msi_c = crc & known & ~msi, crc & msi
        print(f"  --- {label}: COREAD MSS {mss_c.sum()}, MSI {msi_c.sum()}; "
              f"solid MSI {msi[known].sum()} of {known.sum()}")

        def add(analysis, subset, metric, est, lo, hi, p, n, ci):
            rows.append(dict(screen=label, analysis=analysis, subset=subset, metric=metric,
                             n=int(n), estimate=est, lo=lo, hi=hi, p=p, ci=ci))
            print(f"      {analysis:<34} {subset:<28} {metric:<16} {est:+.3f} "
                  f"[{lo:+.3f}, {hi:+.3f}]  p={p:.3f}")

        for subset, m in [("COREAD", crc & known), ("COREAD MSS", mss_c), ("COREAD MSI", msi_c)]:
            o = _r_row(dtp[m], auc[m])
            add("DTP vs AUC", subset, "r", o["r"], o["lo"], o["hi"], o["p"], o["n"], o["ci"])
        r1, r2 = stats.pearsonr(dtp[mss_c], auc[mss_c])[0], stats.pearsonr(dtp[msi_c], auc[msi_c])[0]
        d = bootstrap_r_difference(dtp[mss_c], auc[mss_c], dtp[msi_c], auc[msi_c])
        add("DTP vs AUC", "COREAD MSS minus MSI", "r difference", r1 - r2,
            np.percentile(d, 2.5), np.percentile(d, 97.5),
            fisher_z_difference_p(r1, mss_c.sum(), r2, msi_c.sum()),
            mss_c.sum() + msi_c.sum(), "bootstrap; Fisher z p")

        # AUC per SD of DTP, with an MSI interaction. Lineage enters as a
        # string column, which patsy treats as categorical (this module's C is
        # config, not patsy's C()).
        for subset, m, extra in [("COREAD", crc & known, ""),
                                 ("solid, lineage fixed effects", known, " + lineage")]:
            df = pd.DataFrame(dict(auc=auc[m], dtp=_zs(dtp[m]), msi=msi[m].astype(float),
                                   lineage=lin[m].astype(str)))
            fit = smf.ols(f"auc ~ dtp * msi{extra}", data=df).fit()
            ci = fit.conf_int()
            for term, metric in [("dtp", "DTP slope in MSS"), ("dtp:msi", "interaction")]:
                add("AUC ~ DTP * MSI (OLS, AUC per SD)", subset, metric, fit.params[term],
                    ci.loc[term, 0], ci.loc[term, 1], fit.pvalues[term], fit.nobs, "OLS")

        pred = elasticnet_pipeline().fit(X[~crc], auc[~crc]).predict(X[crc])
        for subset, m in [("COREAD", known[crc]), ("COREAD MSS", ~msi[crc] & known[crc]),
                          ("COREAD MSI", msi[crc])]:
            o = _r_row(pred[m], auc[crc][m])
            add("pan-solid model -> COREAD", subset, "r", o["r"], o["lo"], o["hi"], o["p"],
                o["n"], o["ci"])

    out = pd.DataFrame(rows)
    out["q"] = out.groupby("screen")["p"].transform(bh_q)
    out["declared"] = "post hoc"
    write_and_report(out, C.PROCESSED / "msi_effect_modification.csv",
                     "msi_effect_modification.csv")


def dtp_mmr_coupling(data):
    """
    Is DTP tied to mismatch repair? DTP against an MMR expression score (mean
    z of config.MMR_GENES) and log10 mutational burden, then the DTP-AUC
    association with burden held fixed. MSI lowers MMR expression and raises
    burden, so the plain COREAD rows partly restate MSI; the within-MSS and
    MSI-adjusted rows are the test. Every row is post hoc.
    """
    banner("PERSISTER-MUTABILITY COUPLING: DTP vs MMR AND BURDEN (post hoc)")
    burden_col = C.COVARIATES_NUM[0]
    rows = []
    for label, (X, y, s) in data.items():
        auc, lin = y[C.TARGET].to_numpy(), y[C.LINEAGE_COL].to_numpy()
        burden = np.log10(y[burden_col].to_numpy(float))
        ok = ((y["msi_status"].notna() & y[C.LINEAGE_COL].notna()).to_numpy()
              & np.isfinite(burden))
        msi = (y["msi_status"] == "MSI").to_numpy().astype(float)
        crc = lin == C.CRC
        dtp = s["DTP"].to_numpy()
        mmr_expr = X[C.MMR_GENES].to_numpy(float)
        print(f"  --- {label}")

        sets = [("COREAD", crc & ok, lambda m: []),
                ("COREAD MSS", crc & ok & (msi == 0), lambda m: []),
                ("COREAD, MSI-adjusted", crc & ok, lambda m: [msi[m]]),
                ("solid, lineage-adjusted", ok, lambda m: _lineage_dummies(lin[m])),
                ("solid, lineage- and MSI-adjusted", ok,
                 lambda m: _lineage_dummies(lin[m]) + [msi[m]])]
        for subset, m, controls in sets:
            mmr = _zs(((mmr_expr[m] - mmr_expr[m].mean(0)) / mmr_expr[m].std(0)).mean(1))
            tests = [("DTP vs MMR score", dtp[m], mmr),
                     ("DTP vs log10 burden", dtp[m], burden[m]),
                     ("DTP vs AUC | log10 burden", dtp[m], auc[m])]
            if subset == "COREAD":
                tests.insert(0, ("MMR score vs MSI (positive control)", mmr, msi[m]))
            for test, a, b in tests:
                ctl = controls(m) + ([burden[m]] if "burden" in test.split("|")[-1]
                                     and "|" in test else [])
                o = _r_row(a, b, ctl)
                rows.append(dict(screen=label, test=test, subset=subset, n_controls=len(ctl), **o))
                print(f"      {test:<38} {subset:<34} r={o['r']:+.3f} "
                      f"[{o['lo']:+.3f}, {o['hi']:+.3f}]  p={o['p']:.3f}  n={o['n']}")

    out = pd.DataFrame(rows)
    out["q"] = out.groupby("screen")["p"].transform(bh_q)
    out["declared"] = "post hoc"
    write_and_report(out, C.PROCESSED / "dtp_mmr_coupling.csv", "dtp_mmr_coupling.csv")


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


def lineage_split_r(t, pred, lin):
    """(between-lineage r, within-lineage r) for one set of predictions, over
    lineages with at least MIN_LINEAGE_N lines. Raw r mixes the two; a model
    can rank tissues well and lines within a tissue badly, or the reverse."""
    df = pd.DataFrame(dict(t=t, p=pred, lin=lin)).dropna()
    df = df.groupby("lin").filter(lambda g: len(g) >= C.MIN_LINEAGE_N)
    gm = df.groupby("lin")[["t", "p"]].mean()
    w = df[["t", "p"]] - df.groupby("lin")[["t", "p"]].transform("mean")
    rb = stats.pearsonr(gm.t, gm.p)[0] if len(gm) >= 3 else np.nan
    rw = stats.pearsonr(w.t, w.p)[0] if len(w) >= 10 else np.nan
    return rb, rw


def module_screen_discrepancy():
    """
    The modules model reaches r~0.25 on GDSC1 but under 0.1 on GDSC2. Each row
    below changes one thing, so the explanation is read off the table:
      responsive only   drop lines at the assay ceiling (AUC > CEILING_AUC);
                        if this closes the gap, the ceiling fraction drives it
      shared lines      both screens on the lines they both measured; if this
                        closes the gap, line composition drives it
      transfer          fit on GDSC1, predict GDSC2; if this holds up, the
                        model transfers and the GDSC2 refit is the problem
      unit-SD target    refit GDSC2 with AUC divided by its SD; ElasticNet's
                        penalty is on the target's scale, so if this closes
                        the gap the fixed alpha is over-shrinking GDSC2
    Every row is split into between- and within-lineage r.
    """
    banner("MODULES MODEL: WHY GDSC1 AND GDSC2 DISAGREE")
    # Response and module scores only; the expression matrix is not needed.
    data = {}
    for lbl in [C.TRAIN, C.TEST]:
        y, s = (pd.read_parquet(C.PROCESSED / f"{name}_{lbl}.parquet")
                for name in ("y", "signature_scores"))
        _, y, s = exclude_haem(y, s)
        data[lbl] = (y, s)
    shared = set(data[C.TRAIN][0].index) & set(data[C.TEST][0].index)
    rows = []

    def describe(screen, line_set, t, lin, pred, r_row):
        rb, rw = lineage_split_r(t, pred, lin)
        rows.append(dict(screen=screen, line_set=line_set, n=len(t),
                         n_lineages=int(pd.Series(lin).nunique()),
                         pct_auc_ceiling=100 * float((t > C.CEILING_AUC).mean()),
                         auc_sd=float(np.std(t, ddof=1)),
                         r_mean=r_row["r_mean"], r_lo=r_row["r_lo"], r_hi=r_row["r_hi"],
                         r_between_lineage=rb, r_within_lineage=rw))
        print(f"    between-lineage r={rb:+.3f}  within-lineage r={rw:+.3f}  "
              f"ceiling={rows[-1]['pct_auc_ceiling']:.1f}%  AUC sd={rows[-1]['auc_sd']:.3f}")

    with warnings.catch_warnings():
        # SelectKBest(k=1500) on four module columns keeps all four, and says so.
        warnings.filterwarnings("ignore", message="k=.*is greater than n_features")

        for screen in [C.TRAIN, C.TEST]:
            y, s = data[screen]
            t_all = y[C.TARGET].to_numpy()
            sets = [("all solid", np.ones(len(y), bool), 1.0),
                    (f"responsive only (AUC <= {C.CEILING_AUC})", t_all <= C.CEILING_AUC, 1.0),
                    ("lines in both screens", y.index.isin(shared), 1.0)]
            if screen == C.TEST:
                sets.append(("all solid, target scaled to unit SD",
                             np.ones(len(y), bool), np.std(t_all, ddof=1)))
            for line_set, m, scale in sets:
                X = s[C.MODELED_MODULES][m]
                t, lin = t_all[m] / scale, y[C.LINEAGE_COL].to_numpy()[m]
                r_row = repeated_cv(X, t, lin, False, f"{screen}: {line_set}")
                describe(screen, line_set, t * scale, lin, oof_predictions(X, t), r_row)

        y1, s1 = data[C.TRAIN]
        y2, s2 = data[C.TEST]
        fit = elasticnet_pipeline().fit(s1[C.MODELED_MODULES], y1[C.TARGET].to_numpy())
    new = ~y2.index.isin(shared)
    for line_set, m in [("transfer: GDSC1 fit, all GDSC2 lines", np.ones(len(y2), bool)),
                        ("transfer: GDSC1 fit, lines only in GDSC2", new)]:
        if m.sum() < 20:
            print(f"  {line_set}: only {int(m.sum())} lines, skipped")
            continue
        t = y2[C.TARGET].to_numpy()[m]
        pred = fit.predict(s2[C.MODELED_MODULES][m])
        r = stats.pearsonr(t, pred)[0]
        print(f"  {line_set:<44} r={r:.3f}  (n={int(m.sum())}, single fit)")
        describe(C.TEST, line_set, t, y2[C.LINEAGE_COL].to_numpy()[m], pred,
                 dict(r_mean=r, r_lo=np.nan, r_hi=np.nan))

    write_and_report(pd.DataFrame(rows), C.PROCESSED / "module_screen_discrepancy.csv")


def penalty_scale_sensitivity():
    """
    Does one fixed ElasticNet penalty suit every model? The penalty is in
    target units and alpha was tuned on GDSC1 5-FU raw AUC, so each model is
    refit three ways:
      current           the target as the pipeline sees it today
      unit SD           the target z-scored on the training rows
      tuning SD         z-scored, then rescaled to the SD of the target alpha
                        was tuned on (GDSC1 5-FU solid AUC)
    A rule that fixed the problem would help every row. It does not: the
    four-feature modules model and the 1,500-gene transcriptome model want
    different penalties, so the target scale alone is not the fix.
    """
    banner("PENALTY SCALE: ONE ALPHA ACROSS SCREENS AND MODEL SIZES")
    data = {}
    for lbl in [C.TRAIN, C.TEST]:
        X, y, s = load_screen(lbl)
        _, y, X, s = exclude_haem(y, X, s)
        data[lbl] = (X, y, s)
    tuning_sd = float(np.std(data[C.TRAIN][1][C.TARGET].to_numpy(), ddof=1))
    rules = [("fixed alpha", fixed_elasticnet_pipeline), ("unit SD", scaled_target(1.0)),
             ("tuning SD", scaled_target(tuning_sd)), ("nested tuning", elasticnet_pipeline)]
    print(f"  tuning SD ({C.TRAIN} {C.TARGET}, solid) = {tuning_sd:.4f}\n")
    rows = []
    with warnings.catch_warnings():
        # Four module columns under SelectKBest(k=1500); unit-SD transcriptome
        # fits that stop at max_iter. Neither changes what the table shows.
        warnings.filterwarnings("ignore", message="k=.*is greater than n_features")
        warnings.filterwarnings("ignore", message="Objective did not converge")
        for screen in [C.TRAIN, C.TEST]:
            X, y, s = data[screen]
            t, lin = y[C.TARGET].to_numpy(), y[C.LINEAGE_COL].to_numpy()
            cnt = pd.Series(lin).value_counts()
            keep = np.isin(lin, list(cnt[cnt >= C.MIN_LINEAGE_N].index))
            for features, F in [("transcriptome", X), ("modules", s[C.MODELED_MODULES])]:
                for deconf in (False, True):
                    m = keep if deconf else np.ones(len(t), bool)
                    for rule, factory in rules:
                        rs = cv_repeats(F[m], t[m], lin[m], deconf, model_factory=factory)
                        rows.append(dict(screen=screen, features=features,
                                         target="de-confounded" if deconf else "raw AUC",
                                         scaling=rule, auc_sd=float(np.std(t[m], ddof=1)),
                                         r_mean=rs.mean(), r_sd=rs.std()))
                        print(f"  {screen} {features:<14} {rows[-1]['target']:<14} "
                              f"{rule:<10} r={rs.mean():.3f}  (sd {rs.std():.3f})")
    write_and_report(pd.DataFrame(rows), C.PROCESSED / "penalty_scale_sensitivity.csv")


def main():
    X, y, s = load_screen(C.TRAIN)
    # The replication screen is needed for its response and scores only.
    y2, s2 = (pd.read_parquet(C.PROCESSED / f"{name}_{C.TEST}.parquet")
              for name in ("y", "signature_scores"))
    mutation_covariates({C.TRAIN: (y, s), C.TEST: (y2, s2)})

    _, y, X, s = exclude_haem(y, X, s)
    solid = {C.TRAIN: (X, y, s), C.TEST: solid_screen(C.TEST)}
    msi_effect_modification(solid)
    dtp_mmr_coupling(solid)
    general_chemosensitivity(X, y, s)
    module_screen_discrepancy()
    penalty_scale_sensitivity()
    banner("DONE")


if __name__ == "__main__":
    main()
