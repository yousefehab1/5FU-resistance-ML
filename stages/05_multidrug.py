"""Model oxaliplatin, SN-38, irinotecan and cisplatin the way 5-FU was modelled.

First builds and QCs a response table per drug and picks the screen each is
modelled on (stops if a drug has fewer than config.MIN_N_STOP lines). Then runs
the stages/02_fu_model.py recipe for each drug and tests its selected genes for
signature enrichment.

Inputs:  data/raw/GDSC{1,2}_fitted_dose_response_24Jul22.csv,
         screened_compounds_rel_8.4.csv, model_list_20260724.csv
         data/processed/X_*.parquet, signature_scores_*.parquet
Outputs: data/processed/targets/<drug>_<screen>.parquet, multidrug_qc.csv,
         multidrug_model_results.csv, multidrug_enrichment.csv,
         multidrug_genes_<drug>.csv,
         reports/multidrug_targets.md, reports/multidrug_models.md
Run:     python stages/05_multidrug.py
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
from lib.io import load_expression
from lib.modeling import (deconfounded_gene_fit, elasticnet_pipeline,
                          cv_repeats, ridge_cv_predict, summarise)
from lib.report import banner, write_and_report, write_report
from lib import signatures as S
from lib.stats import hypergeometric_overlap

SCREENS = {"GDSC1": C.GDSC1_FILE, "GDSC2": C.GDSC2_FILE}

# Wider than config.DRUG_COLS: the per-drug tables also need the ids.
RESPONSE_COLS = ["SANGER_MODEL_ID", "CELL_LINE_NAME", "TCGA_DESC", "DRUG_ID", "DRUG_NAME",
                 "LN_IC50", "AUC", "RMSE", "Z_SCORE", "MIN_CONC", "MAX_CONC"]


# ============================================================================
# LOADING
# ============================================================================

def read_response_csv(path, label):
    if not path.exists():
        print(f"  !! MISSING: {path}")
        sys.exit(1)
    d = pd.read_csv(path, low_memory=False, usecols=RESPONSE_COLS)
    print(f"  {label}: {len(d):,} rows, {d.DRUG_NAME.nunique()} drugs, "
          f"{d[C.ID_COL].nunique():,} cell lines")
    return d


def check_multi_id_drugs(screens, compounds):
    """
    Stop-and-report check before deduplicating: for every drug with >1 DRUG_ID in any
    screen, confirm the IDs share DRUG_NAME, TARGET and TARGET_PATHWAY in
    screened_compounds_rel_8.4.csv. If they don't, they are not simply
    re-screens and must not be silently averaged.
    """
    for label, d in screens.items():
        for drug in C.DRUGS:
            ids = sorted(d.loc[d.DRUG_NAME == drug, "DRUG_ID"].unique())
            if len(ids) <= 1:
                continue
            rows = compounds[compounds.DRUG_ID.isin(ids)]
            targets = rows[["TARGET", "TARGET_PATHWAY"]].drop_duplicates()
            names = rows["DRUG_NAME"].unique()
            ok = len(targets) <= 1 and len(names) <= 1
            flag = "" if ok else "   <-- MISMATCH, do not average blindly"
            print(f"  {label} {drug:16} DRUG_IDs={ids}  "
                  f"target(s)={targets.TARGET.tolist()}{flag}")
            if not ok:
                print(f"     !! {drug} DRUG_IDs disagree on name/target/pathway. "
                      f"Stopping (expected column/consistency missing).")
                sys.exit(1)


# ============================================================================
# PER-DRUG EXTRACTION
# ============================================================================

def extract_drug(d, drug, screen_label):
    """
    One drug from one screen: drop haem lineages, average across DRUG_IDs
    (checked safe above), return a one-row-per-cell-line table plus a QC dict.
    Returns (None, None) if the drug is absent from this screen.
    """
    sub = d[d.DRUG_NAME == drug].copy()
    if sub.empty:
        return None, None

    ids = sorted(int(i) for i in sub.DRUG_ID.unique())
    id_agreement = None
    if len(ids) > 1:
        wide = sub.pivot_table(index=C.ID_COL, columns="DRUG_ID", values="AUC")
        shared = wide.dropna()
        if len(shared) >= 10 and len(ids) == 2:
            id_agreement = stats.pearsonr(shared[ids[0]], shared[ids[1]])[0]

    sub = sub[~sub.TCGA_DESC.isin(C.HAEM)]
    n_rows, n_lines = len(sub), sub[C.ID_COL].nunique()

    agg = sub.groupby(C.ID_COL).agg(
        CELL_LINE_NAME=("CELL_LINE_NAME", "first"),
        TCGA_DESC=("TCGA_DESC", "first"),
        LN_IC50=("LN_IC50", "mean"),
        AUC=("AUC", "mean"),
        RMSE=("RMSE", "mean"),
        Z_SCORE=("Z_SCORE", "mean"),
        MIN_CONC=("MIN_CONC", "mean"),
        MAX_CONC=("MAX_CONC", "mean"),
    )
    agg["ic50_extrapolated"] = agg.LN_IC50 > np.log(agg.MAX_CONC)

    qc = dict(
        drug=drug, screen=screen_label, drug_ids=ids, id_agreement_r=id_agreement,
        n_rows_before_dedup=n_rows, n=n_lines,
        auc_mean=agg.AUC.mean(), auc_sd=agg.AUC.std(),
        pct_extrapolated=100 * agg.ic50_extrapolated.mean(),
        pct_auc_gt_090=100 * (agg.AUC > 0.90).mean(),
        pct_auc_gt_095=100 * (agg.AUC > C.CEILING_AUC).mean(),
        pct_auc_gt_098=100 * (agg.AUC > 0.98).mean(),
        conc_min=agg.MIN_CONC.min(), conc_max=agg.MAX_CONC.max(),
    )
    return agg, qc


# ============================================================================
# TARGETS
# ============================================================================

def build_targets():
    """Per-drug response tables, QC, and the screen each drug is modelled on."""
    banner("TARGETS 1. LOADING GDSC1 AND GDSC2")
    screens = {label: read_response_csv(path, label) for label, path in SCREENS.items()}
    compounds = pd.read_csv(C.COMPOUNDS_FILE, low_memory=False)

    banner("TARGETS 2. CHECKING MULTI-DRUG_ID DRUGS BEFORE DEDUPLICATING")
    check_multi_id_drugs(screens, compounds)

    banner("TARGETS 3. PER-DRUG x SCREEN EXTRACTION AND QC")
    C.TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    models = (pd.read_csv(C.MODEL_LIST_FILE, low_memory=False)
                .rename(columns={"model_id": C.ID_COL})[[C.ID_COL] + C.MODEL_META]
                .drop_duplicates(C.ID_COL).set_index(C.ID_COL))

    tables, qc_rows = {}, []
    for drug in C.DRUGS:
        for screen_label, d in screens.items():
            agg, qc = extract_drug(d, drug, screen_label)
            if agg is None:
                print(f"  {drug:16} {screen_label}: absent from this screen")
                continue
            flag = "" if qc["n"] >= C.MIN_N_TASK else "  <-- below n=300"
            agree = (f"  DRUG_ID agreement r={qc['id_agreement_r']:.3f}"
                     if qc["id_agreement_r"] is not None else "")
            print(f"  {drug:16} {screen_label}  n={qc['n']:>4}  "
                  f"AUC mean={qc['auc_mean']:.3f} sd={qc['auc_sd']:.3f}  "
                  f"extrap={qc['pct_extrapolated']:.1f}%  "
                  f"ceiling(>{C.CEILING_AUC})={qc['pct_auc_gt_095']:.1f}%"
                  f"{agree}{flag}")
            if qc["n"] < C.MIN_N_STOP:
                print(f"  !! {drug} {screen_label}: n={qc['n']} < {C.MIN_N_STOP}. "
                      f"Stopping.")
                sys.exit(1)

            joined = agg.join(models, how="left")
            out_path = C.TARGETS_DIR / f"{drug.replace(' ', '_')}_{screen_label}.parquet"
            joined.to_parquet(out_path)
            tables[(drug, screen_label)] = agg
            qc_rows.append(qc)

    banner("TARGETS 4. MEASUREMENT CEILING (GDSC1 vs GDSC2 on shared lines, AUC)")
    ceilings = {}
    for drug in C.DRUGS:
        a, b = tables.get((drug, "GDSC1")), tables.get((drug, "GDSC2"))
        if a is None or b is None:
            ceilings[drug] = None
            print(f"  {drug:16} not computable (screened in one arm only)")
            continue
        both = pd.concat([a.AUC, b.AUC], axis=1, keys=[C.TRAIN, C.TEST]).dropna()
        if len(both) < 10:
            ceilings[drug] = None
            print(f"  {drug:16} not computable (only {len(both)} shared lines)")
            continue
        r, p = stats.pearsonr(both.GDSC1, both.GDSC2)
        ceilings[drug] = r
        print(f"  {drug:16} r={r:.3f}  (n={len(both):,} shared lines, p={p:.1e})")

    banner("TARGETS 5. SCREEN CHOICE PER DRUG (dynamic range = AUC SD)")
    chosen = {}
    for drug in C.DRUGS:
        options = {lbl: tables[(drug, lbl)] for lbl in SCREENS
                   if (drug, lbl) in tables}
        if len(options) == 1:
            lbl = next(iter(options))
            chosen[drug] = (lbl, f"only screen available (AUC sd={options[lbl].AUC.std():.3f})")
        else:
            sds = {lbl: float(t.AUC.std()) for lbl, t in options.items()}
            best = max(sds, key=sds.get)
            others = ", ".join(f"{k}={v:.3f}" for k, v in sds.items() if k != best)
            chosen[drug] = (best, f"higher AUC sd ({sds[best]:.3f} vs {others})")
        print(f"  {drug:16} -> {chosen[drug][0]:8} ({chosen[drug][1]})")

    banner("TARGETS 6. IRINOTECAN vs SN-38 (prodrug / active-metabolite divergence check)")
    iri, sn38 = tables.get(("Irinotecan", "GDSC2")), tables.get(("SN-38", "GDSC2"))
    iri_sn38_r = None
    if iri is not None and sn38 is not None:
        both = pd.concat([iri.AUC, sn38.AUC], axis=1, keys=["iri", "sn38"]).dropna()
        iri_sn38_r, p = stats.pearsonr(both.iri, both.sn38)
        print(f"  Irinotecan vs SN-38 AUC: r={iri_sn38_r:.3f}  (n={len(both):,}, p={p:.1e})")
        print(f"  {'Carried as separate drugs.' if iri_sn38_r < 0.7 else 'Track closely, as expected for a prodrug/metabolite pair.'}")
    else:
        print("  one or both unavailable")

    banner("TARGETS 7. WRITING QC TABLE AND FINDINGS")
    qc_df = pd.DataFrame(qc_rows)
    qc_df["ceiling_r"] = qc_df["drug"].map(
        lambda dr: ceilings[dr] if ceilings.get(dr) is not None else np.nan)
    qc_df["ceiling_status"] = qc_df["drug"].map(
        lambda dr: f"{ceilings[dr]:.3f}" if ceilings.get(dr) is not None else "not computable")
    qc_df["chosen_screen"] = qc_df["drug"].map(lambda dr: chosen[dr][0])
    qc_df["is_chosen"] = qc_df["screen"] == qc_df["chosen_screen"]
    write_and_report(qc_df, C.PROCESSED / "multidrug_qc.csv")

    dropped = qc_df[qc_df.n < C.MIN_N_TASK]

    lines = []
    lines.append("# Multi-drug QC\n")
    lines.append("Run against GDSC1/GDSC2, repeating 5-FU's drug-row and "
                  "screen-agreement checks per drug. Full numbers in `data/processed/multidrug_qc.csv`.\n")

    lines.append("## Screen chosen per drug\n")
    for drug in C.DRUGS:
        lbl, reason = chosen[drug]
        lines.append(f"- **{drug}**: {lbl}. Reason: {reason}.")
    lines.append("")

    lines.append("## Sample size (n >= 300 acceptance criterion)\n")
    if dropped.empty:
        lines.append("All drug x screen combinations meet n >= 300; nothing dropped.\n")
    else:
        for _, r in dropped.iterrows():
            lines.append(f"- **{r.drug} ({r.screen})**: n={r.n}, below 300. "
                          f"Reason: {'only screened in one arm' if pd.isna(r.get('id_agreement_r')) else 'small screened cohort'}.")
        lines.append("")

    lines.append("## Measurement ceiling (GDSC1 vs GDSC2 AUC, shared lines)\n")
    for drug in C.DRUGS:
        c = ceilings.get(drug)
        if c is not None:
            lines.append(f"- **{drug}**: r = {c:.3f}.")
        else:
            lines.append(f"- **{drug}**: not computable (screened in one arm only, "
                          f"no other drug's ceiling is substituted).")
    lines.append("")
    fu_r = ceilings.get("5-Fluorouracil")
    fu_r_str = f"{fu_r:.3f}" if fu_r is not None else "n/a"
    lines.append(f"5-FU's own ceiling here (r={fu_r_str}, solid tumours only) "
                  "is lower than the pan-cancer r=0.604 (n=902 shared lines including "
                  "haem lineages). Both are legitimate: 0.604 is the whole-panel number; "
                  "the value here restricts to the solid cohort this project models on. "
                  "Do not treat the gap as an error, and do not substitute "
                  "either number for another drug's ceiling.\n")

    lines.append("## AUC distribution and extrapolation, per drug x screen\n")
    lines.append("| Drug | Screen | n | AUC mean | AUC sd | % extrapolated | "
                  f"% >0.90 | % >{C.CEILING_AUC} (ceiling) | % >0.98 | "
                  "Conc range (uM) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in qc_df.sort_values(["drug", "screen"]).iterrows():
        lines.append(f"| {r.drug} | {r.screen} | {r.n} | {r.auc_mean:.3f} | {r.auc_sd:.3f} | "
                      f"{r.pct_extrapolated:.1f}% | {r.pct_auc_gt_090:.1f}% | "
                      f"{r.pct_auc_gt_095:.1f}% | {r.pct_auc_gt_098:.1f}% | "
                      f"{r.conc_min:.4g}-{r.conc_max:.4g} |")
    lines.append("")

    lines.append("## Multi-DRUG_ID re-screens, averaged\n")
    for _, r in qc_df.iterrows():
        if len(r.drug_ids) > 1:
            ag = f"r={r.id_agreement_r:.3f}" if r.id_agreement_r is not None else "not computable"
            lines.append(f"- **{r.drug} ({r.screen})**: DRUG_IDs {r.drug_ids}, confirmed same "
                          f"DRUG_NAME/TARGET/TARGET_PATHWAY in screened_compounds_rel_8.4.csv, "
                          f"agreement {ag}, averaged per cell line.")
    lines.append("")

    lines.append("## Irinotecan vs SN-38\n")
    if iri_sn38_r is not None:
        lines.append(f"AUC correlation on shared GDSC2 lines: r = {iri_sn38_r:.3f}. "
                      f"Both carried forward as separate drugs; this is a "
                      f"QC-stage flag; 06_drug_specificity.py makes the full comparison.\n")
    else:
        lines.append("Could not be computed (one or both tables unavailable).\n")

    lines.append("## Traps checked\n")
    lines.append("- 5-FU's parameters (dose range, ceiling, screen choice) were not "
                  "inherited; every drug ran its own extraction, QC and ceiling above.")
    lines.append("- DRUG_ID multiplicity checked against screened_compounds_rel_8.4.csv "
                  "before averaging, not assumed.")
    lines.append("- Irinotecan and SN-38 carried as distinct targets given the prodrug "
                  "relationship, not merged.")

    write_report("multidrug_targets.md", "\n".join(lines) + "\n")
    print(f"  Wrote {len(tables)} target tables -> {C.TARGETS_DIR}")


# ============================================================================
# MODELS
# n_jobs left at default (None = 1) throughout: nested parallelism with a
# 786 x 36,000 matrix exhausted memory before.
# ============================================================================


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
    06_drug_specificity.py. What stays here is which sets are counted: the
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


def fit_models():
    """The 5-FU modelling recipe, once per drug, plus signature enrichment of its genes."""
    banner("MODELS: LOADING QC DECISIONS AND SIGNATURE UNIVERSE")
    qc = pd.read_csv(C.PROCESSED / "multidrug_qc.csv")
    chosen = dict(zip(qc.drug, qc.chosen_screen))
    print("  chosen screens:", chosen)

    # Both screens up front: five drugs are modelled below and each one picks a
    # screen, so loading per drug would re-read the same matrix repeatedly.
    # No haematological filter here; the per-drug response tables decide the
    # cohort, and every one of them is intersected with X by index below.
    Xcache = {lbl: load_expression(lbl, impute=True) for lbl in [C.TRAIN, C.TEST]}
    sigscore_cache = {lbl: pd.read_parquet(C.PROCESSED / f"signature_scores_{lbl}.parquet")
                      for lbl in [C.TRAIN, C.TEST]}

    alias_map = S.load_alias_map(required=False)
    all_results, all_genes, all_enrichment = [], {}, []

    for drug in C.DRUGS:
        screen = chosen[drug]
        banner(f"DRUG: {drug}  (screen={screen})")

        tpath = C.TARGETS_DIR / f"{drug.replace(' ', '_')}_{screen}.parquet"
        y_full = pd.read_parquet(tpath)
        X_all = Xcache[screen]
        scores_all = sigscore_cache[screen]

        shared = y_full.index.intersection(X_all.index)
        n_lost = len(y_full) - len(shared)
        print(f"  target table n={len(y_full)}, with cached expression n={len(shared)} "
              f"({n_lost} lost to missing expression)")

        y = y_full.loc[shared]
        X = X_all.loc[shared]
        scores = scores_all.loc[shared]
        lin = y.TCGA_DESC.to_numpy()
        t = y.AUC.to_numpy()

        cnt = pd.Series(lin).value_counts()
        big = set(cnt[cnt >= C.MIN_LINEAGE_N].index)
        keep = np.isin(lin, list(big))
        print(f"  lineages with >= {C.MIN_LINEAGE_N} lines: {len(big)}, "
              f"{keep.sum()} of {len(y)} lines")

        banner(f"{drug}: REPEATED CV, RAW vs DE-CONFOUNDED")
        report_cv(X, t, lin, False, "transcriptome -> raw AUC",
                  all_results, drug, screen, len(y))
        report_cv(X[keep], t[keep], lin[keep], True,
                  "transcriptome -> AUC, lineage de-confounded",
                  all_results, drug, screen, int(keep.sum()))

        banner(f"{drug}: BASELINES (02_fu_model.py convention)")
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

        rng = np.random.default_rng(C.SEED)
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

        banner(f"{drug}: STRATIFIED BY ASSAY RESOLUTION (AUC > {C.CEILING_AUC})")
        p_all = np.full(len(t), np.nan)
        for tr, te in KFold(C.N_FOLDS, shuffle=True, random_state=C.SEED).split(X):
            p_all[te] = elasticnet_pipeline().fit(X.iloc[tr], t[tr]).predict(X.iloc[te])
        resp = t <= C.CEILING_AUC
        for lab, msk in [(f"responsive (AUC<={C.CEILING_AUC})", resp),
                         (f"ceiling (AUC>{C.CEILING_AUC})", ~resp)]:
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
        print(f"  {len(g)} non-zero genes of {C.K_GENES} selected")
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
    write_and_report(res_df, C.PROCESSED / "multidrug_model_results.csv")

    for drug, g in all_genes.items():
        p = C.PROCESSED / f"multidrug_genes_{drug.replace(' ', '_')}.csv"
        write_and_report(g, p)

    enr_df = pd.concat(all_enrichment, ignore_index=True)
    write_and_report(enr_df, C.PROCESSED / "multidrug_enrichment.csv")

    banner("WRITING FINDINGS DOC")
    lines = ["# Multi-drug models\n",
             "One independent run of the 02_fu_model.py pipeline per drug, on "
             "each drug's chosen screen from 05_multidrug.py. Full numbers in "
             "`data/processed/multidrug_model_results.csv` and "
             "`multidrug_enrichment.csv`.\n"]

    lines.append("## Permuted-label check, per drug\n")
    for drug in C.DRUGS:
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
    for drug in C.DRUGS:
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
                 "06_drug_specificity.py.\n")

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

    lines.append("## Expression-cache coverage loss\n")
    for drug in C.DRUGS:
        screen = chosen[drug]
        tpath = C.TARGETS_DIR / f"{drug.replace(' ', '_')}_{screen}.parquet"
        n_target = len(pd.read_parquet(tpath))
        n_modelled = int(res_df[(res_df.drug == drug) &
                                (res_df.model.str.contains("raw AUC")) &
                                (~res_df.model.str.contains("permuted"))].iloc[0].n)
        lines.append(f"- **{drug} ({screen})**: {n_target} in target table, "
                     f"{n_modelled} modelled ({n_target - n_modelled} lost to missing "
                     f"expression cache coverage).")
    lines.append("")

    write_report("multidrug_models.md", "\n".join(lines) + "\n")


def main():
    build_targets()
    fit_models()
    banner("DONE")


if __name__ == "__main__":
    main()
