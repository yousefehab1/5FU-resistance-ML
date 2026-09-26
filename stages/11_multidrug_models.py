"""Run the stage 04 pipeline for each drug and test its genes for signature enrichment.

Inputs:  data/processed/targets/, multidrug_qc.csv, X_*.parquet, signature_scores_*.parquet
Outputs: data/processed/multidrug_model_results.csv, multidrug_enrichment.csv,
         multidrug_genes_<drug>.csv, reports/11_multidrug_models.md
Run:     python stages/11_multidrug_models.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

ROOT = C.PROJECT_ROOT
PROC, DOCS = ROOT / "data" / "processed", ROOT / "data" / "processed" / "reports"
TARGETS_DIR = PROC / "targets"

# Analysis parameters live in config.py; nothing here re-declares one.
from config import (DRUGS, K_GENES, SEED, N_FOLDS,
                    MIN_LINEAGE_N, CEILING_AUC)
from lib.io import load_expression
from lib.modeling import (deconfounded_gene_fit, elasticnet_pipeline,
                             cv_repeats, ridge_cv_predict, summarise)
from lib.report import banner
from lib import signatures as S
from lib.stats import hypergeometric_overlap


# n_jobs left at default (None = 1) throughout: nested parallelism with a
# 786 x 36,000 matrix exhausted memory before.


def report_cv(X, y, lineage, deconfound, name, res, drug, screen, n):
    """The shared repeated CV, reported this script's way.

    Numerics come from lib.modeling.cv_repeats; only the presentation is
    local: a wider name column, no standard deviation in the printed line,
    and a result row carrying drug, screen and n.
    """
    rs = cv_repeats(X, y, lineage, deconfound)
    mean, lo, hi, sd = summarise(rs)
    print(f"  {name:<48} r={mean:.3f}  [{lo:.3f}, {hi:.3f}]")
    res.append(dict(drug=drug, screen=screen, model=name, n=n,
                    r_mean=mean, r_lo=lo, r_hi=hi, r_sd=sd))
    return rs


def evaluate_baseline(name, y_true, y_pred, res, drug, screen, n):
    r, _ = stats.pearsonr(y_true, y_pred)
    print(f"  {name:<48} r={r:.3f}")
    res.append(dict(drug=drug, screen=screen, model=name, n=n,
                    r_mean=r, r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
    return r


def hypergeom_enrichment(selected, universe, sigs):
    """
    Observed vs expected overlap between `selected` genes and each signature
    in `sigs`, hypergeometric p-value (upper tail: P(X >= observed)).

    The arithmetic is lib.stats.hypergeometric_overlap, shared with
    12_drug_specificity.py. What stays here is which sets are counted: the
    signature is intersected with the universe and the selected list is not,
    because the selected genes are columns of the matrix the universe was taken
    from and are already in it. Changing that would change the test.
    """
    M = len(universe)
    N = len(selected)
    rows = []
    for name, genes in sigs.items():
        K = len(set(genes) & set(universe))
        obs = len(set(selected) & set(genes))
        expected, fold, p = hypergeometric_overlap(N, K, obs, M)
        rows.append(dict(signature=name, n_signature_in_universe=K,
                         n_selected=N, observed=obs, expected=expected,
                         fold_enrichment=fold, p_hypergeom=p))
    return pd.DataFrame(rows)


def main():
    banner("0. LOADING QC DECISIONS AND SIGNATURE UNIVERSE")
    qc = pd.read_csv(PROC / "multidrug_qc.csv")
    chosen = dict(zip(qc.drug, qc.chosen_screen))
    print("  chosen screens:", chosen)

    # Both screens up front: five drugs are modelled below and each one picks a
    # screen, so loading per drug would re-read the same matrix repeatedly.
    # No haematological filter here; the per-drug response tables decide the
    # cohort, and every one of them is intersected with X by index below.
    Xcache = {lbl: load_expression(lbl, impute=True) for lbl in ["GDSC1", "GDSC2"]}
    sigscore_cache = {lbl: pd.read_parquet(PROC / f"signature_scores_{lbl}.parquet")
                      for lbl in ["GDSC1", "GDSC2"]}

    alias_map = S.load_alias_map(required=False)
    all_results, all_genes, all_enrichment = [], {}, []

    for drug in DRUGS:
        screen = chosen[drug]
        banner(f"DRUG: {drug}  (screen={screen})")

        tpath = TARGETS_DIR / f"{drug.replace(' ', '_')}_{screen}.parquet"
        y_full = pd.read_parquet(tpath)
        X_all = Xcache[screen]
        scores_all = sigscore_cache[screen]

        shared = y_full.index.intersection(X_all.index)
        n_lost = len(y_full) - len(shared)
        print(f"  target table n={len(y_full)}, with cached expression n={len(shared)} "
              f"({n_lost} lost, see D-multi3 in docstring)")

        y = y_full.loc[shared]
        X = X_all.loc[shared]
        scores = scores_all.loc[shared]
        lin = y.TCGA_DESC.to_numpy()
        t = y.AUC.to_numpy()

        cnt = pd.Series(lin).value_counts()
        big = set(cnt[cnt >= MIN_LINEAGE_N].index)
        keep = np.isin(lin, list(big))
        print(f"  lineages with >= {MIN_LINEAGE_N} lines: {len(big)}, "
              f"{keep.sum()} of {len(y)} lines")

        banner(f"{drug}: REPEATED CV, RAW vs DE-CONFOUNDED")
        report_cv(X, t, lin, False, "transcriptome -> raw AUC",
                  all_results, drug, screen, len(y))
        report_cv(X[keep], t[keep], lin[keep], True,
                  "transcriptome -> AUC, lineage de-confounded",
                  all_results, drug, screen, int(keep.sum()))

        banner(f"{drug}: BASELINES (03_baselines.py convention)")
        lin_df = y[["TCGA_DESC"]].fillna("UNKNOWN")
        rmse0 = np.sqrt(((t - t.mean()) ** 2).mean())
        print(f"  {'B1 mean only':<48} r=0.000  (RMSE={rmse0:.4f})")
        all_results.append(dict(drug=drug, screen=screen, model="B1 mean only",
                                n=len(y), r_mean=0.0, r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))

        evaluate_baseline("B2 lineage only", t,
                          ridge_cv_predict(lin_df, t, ("TCGA_DESC",)),
                          all_results, drug, screen, len(y))
        evaluate_baseline("B3 proliferation only", t,
                          ridge_cv_predict(scores[["CellCycle"]], t),
                          all_results, drug, screen, len(y))
        combo = lin_df.copy()
        combo["CellCycle"] = scores["CellCycle"].to_numpy()
        evaluate_baseline("B4 lineage + proliferation", t,
                          ridge_cv_predict(combo, t, ("TCGA_DESC",)),
                          all_results, drug, screen, len(y))

        rng = np.random.default_rng(SEED)
        perm = rng.permutation(t)
        b5_pred = ridge_cv_predict(combo, perm, ("TCGA_DESC",))
        r5, _ = stats.pearsonr(perm, b5_pred)
        r2_5 = r5 ** 2
        print(f"  {'B5 PERMUTED LABELS':<48} r={r5:+.3f}  R2={r2_5:.4f}  "
              f"{'** LEAKAGE **' if abs(r2_5) > 0.05 else 'OK'}")
        all_results.append(dict(drug=drug, screen=screen, model="B5 PERMUTED LABELS",
                                n=len(y), r_mean=r5, r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
        if abs(r2_5) > 0.05:
            print(f"  !! permuted-label check FAILED for {drug}. Stopping.")
            sys.exit(1)

        # Full permuted-label check on the de-confounded
        # ElasticNet pipeline itself, not just the Ridge baseline above.
        perm_full = rng.permutation(t[keep])
        rs_perm = report_cv(X[keep], perm_full, lin[keep], True,
                            "R1 CHECK: de-confounded model, permuted target",
                            [], drug, screen, int(keep.sum()))
        r1_r, r1_r2 = rs_perm.mean(), rs_perm.mean() ** 2
        r1_ok = abs(r1_r) < 0.15 and r1_r2 <= 0.05
        print(f"  permuted-label check: |r|={abs(r1_r):.3f} (<0.15?) R2={r1_r2:.4f} (<=0.05?) "
              f"-> {'PASS' if r1_ok else 'FAIL'}")
        all_results.append(dict(drug=drug, screen=screen,
                                model="R1 CHECK: de-confounded model, permuted target",
                                n=int(keep.sum()), r_mean=r1_r, r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
        if not r1_ok:
            print(f"  !! permuted-label check FAILED for {drug}'s de-confounded model. "
                  f"Stopping.")
            sys.exit(1)

        banner(f"{drug}: STRATIFIED BY ASSAY RESOLUTION (AUC > {CEILING_AUC})")
        p_all = np.full(len(t), np.nan)
        for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(X):
            p_all[te] = elasticnet_pipeline().fit(X.iloc[tr], t[tr]).predict(X.iloc[te])
        resp = t <= CEILING_AUC
        for lab, msk in [(f"responsive (AUC<={CEILING_AUC})", resp),
                         (f"ceiling (AUC>{CEILING_AUC})", ~resp)]:
            if msk.sum() > 10:
                rr = stats.pearsonr(t[msk], p_all[msk])[0]
                print(f"  {lab:<32} n={int(msk.sum()):>4} ({100*msk.mean():.1f}%)  r={rr:+.3f}")
                all_results.append(dict(drug=drug, screen=screen,
                                        model=f"stratified: {lab}", n=int(msk.sum()),
                                        r_mean=rr, r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
            else:
                print(f"  {lab:<32} n={int(msk.sum())}, too few to report")

        banner(f"{drug}: GENE SELECTION AND SIGNATURE ENRICHMENT")
        g, _ = deconfounded_gene_fit(X, t, lin, keep)
        print(f"  {len(g)} non-zero genes of {K_GENES} selected")
        all_genes[drug] = g

        sigs = S.load_signatures(X.columns, alias_map, verbose=False)
        enr = hypergeom_enrichment(set(g.gene), X.columns, sigs)
        enr.insert(0, "drug", drug)
        enr.insert(1, "screen", screen)
        all_enrichment.append(enr)
        print(enr[["signature", "n_signature_in_universe", "observed", "expected",
                   "fold_enrichment", "p_hypergeom"]].to_string(index=False))

    banner("WRITING OUTPUTS")
    res_df = pd.DataFrame(all_results)
    res_df.to_csv(PROC / "multidrug_model_results.csv", index=False)
    print(f"  -> {PROC / 'multidrug_model_results.csv'}")

    for drug, g in all_genes.items():
        p = PROC / f"multidrug_genes_{drug.replace(' ', '_')}.csv"
        g.to_csv(p, index=False)
        print(f"  -> {p}")

    enr_df = pd.concat(all_enrichment, ignore_index=True)
    enr_df.to_csv(PROC / "multidrug_enrichment.csv", index=False)
    print(f"  -> {PROC / 'multidrug_enrichment.csv'}")

    banner("WRITING FINDINGS DOC")
    lines = ["# Multi-drug models\n",
             "One independent run of the 04_final_model.py pipeline per drug, on "
             "each drug's chosen screen from 10_multidrug_targets.py. Full numbers in "
             "`data/processed/multidrug_model_results.csv` and "
             "`multidrug_enrichment.csv`.\n"]

    lines.append("## Permuted-label check, per drug\n")
    for drug in DRUGS:
        row = res_df[(res_df.drug == drug) &
                     (res_df.model.str.startswith("R1 CHECK"))]
        if len(row):
            r = row.iloc[0]
            lines.append(f"- **{drug}**: r={r.r_mean:+.3f}, R2={r.r_mean**2:.4f} -> PASS "
                         f"(all drugs reaching this line passed; the script stops on "
                         f"first failure).")
    lines.append("")

    lines.append("## Main result per drug: de-confounded model vs B4 baseline\n")
    lines.append("| Drug | Screen | n | De-confounded r [CI] | Raw r [CI] | "
                 "B4 lineage+prolif r | Beats B4 (CIs non-overlapping)? |")
    lines.append("|---|---|---|---|---|---|---|")
    for drug in DRUGS:
        screen = chosen[drug]
        dec = res_df[(res_df.drug == drug) &
                     (res_df.model.str.contains("de-confounded")) &
                     (~res_df.model.str.contains("CHECK"))].iloc[0]
        raw = res_df[(res_df.drug == drug) & (res_df.model.str.contains("raw AUC")) &
                     (~res_df.model.str.contains("permuted"))].iloc[0]
        b4 = res_df[(res_df.drug == drug) & (res_df.model.str.startswith("B4"))].iloc[0]
        beats = "yes" if dec.r_lo > b4.r_mean else ("no" if dec.r_hi < b4.r_mean else "inconclusive")
        lines.append(f"| {drug} | {screen} | {dec.n} | {dec.r_mean:.3f} "
                     f"[{dec.r_lo:.3f}, {dec.r_hi:.3f}] | {raw.r_mean:.3f} "
                     f"[{raw.r_lo:.3f}, {raw.r_hi:.3f}] | {b4.r_mean:.3f} | {beats} |")
    lines.append("")
    lines.append("Primary comparison, pre-specified: de-confounded model CI vs B4 point "
                 "estimate, per drug independently, judged by CI separation, not "
                 "three-decimal point estimates. Cross-drug ranking is in "
                 "12_drug_specificity.py.\n")

    lines.append("## Assay-resolution stratification\n")
    strat = res_df[res_df.model.str.startswith("stratified")]
    lines.append("| Drug | Band | n | r |")
    lines.append("|---|---|---|---|")
    for _, r in strat.iterrows():
        lines.append(f"| {r.drug} | {r.model.replace('stratified: ', '')} | {r.n} | {r.r_mean:+.3f} |")
    lines.append("")

    lines.append("## Signature enrichment, de-confounded model's selected genes\n")
    sig_p = enr_df[enr_df.p_hypergeom < 0.05 / len(enr_df)]
    if len(sig_p):
        lines.append("Bonferroni-significant (p < 0.05 / n_tests) hits:\n")
        for _, r in sig_p.iterrows():
            lines.append(f"- **{r.drug} x {r.signature}**: observed {r.observed}, "
                         f"expected {r.expected:.2f}, fold {r.fold_enrichment:.2f}x, "
                         f"p={r.p_hypergeom:.2e}")
    else:
        lines.append("No signature reached the Bonferroni threshold "
                     f"(p < {0.05/len(enr_df):.2e}) for any drug.")
    lines.append("")

    lines.append("## Expression-cache coverage loss (D-multi3)\n")
    for drug in DRUGS:
        screen = chosen[drug]
        tpath = TARGETS_DIR / f"{drug.replace(' ', '_')}_{screen}.parquet"
        n_target = len(pd.read_parquet(tpath))
        n_modelled = int(res_df[(res_df.drug == drug) &
                                (res_df.model.str.contains("raw AUC")) &
                                (~res_df.model.str.contains("permuted"))].iloc[0].n)
        lines.append(f"- **{drug} ({screen})**: {n_target} in target table, "
                     f"{n_modelled} modelled ({n_target - n_modelled} lost to missing "
                     f"expression cache coverage).")
    lines.append("")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "11_multidrug_models.md").write_text("\n".join(lines) + "\n")
    print(f"  -> {DOCS / '11_multidrug_models.md'}")

    banner("DONE")


if __name__ == "__main__":
    main()
