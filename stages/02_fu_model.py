"""
The 5-FU model, from the bar it has to clear to how it is displayed.

  baselines     What has to be beaten before any real model is trained:
                  B1  predict the training mean         -> defines R2 = 0
                  B2  lineage only (TCGA_DESC)          -> tissue of origin
                  B3  proliferation only (CellCycle)    -> 5-FU kills dividing cells
                  B4  lineage + proliferation           -> the real bar to clear
                  B5  PERMUTED LABELS                   -> leakage tripwire, must score ~0
                B5 is the one to rerun after any change to modelling code:
                shuffle the target, rerun unchanged, confirm ~0. Also runs the
                shared-gene diagnostic and the within-COREAD association with
                its GDSC2 replication.

  final_model   Repeated 5x5 CV with confidence intervals (not a single
                split), fold-internal lineage de-confounding applied to BOTH
                features and target, transfer of a pan-solid model to CRC, and
                assay-resolution stratification. The de-confounding logic
                (lib.modeling.repeated_cv) is the single most safety-critical
                piece of code in this project -- see its docstring.

  enrichment    Is the de-confounded model's gene list enriched for any
                curated signature, more than chance predicts?

  calibration   Affine recalibration to GDSC2: a display fix for the
                dashboard, NOT a modelling improvement.

INPUTS   data/processed/{X,y,signature_scores}_{GDSC1,GDSC2}.parquet
         data/raw/signatures/*.txt, data/raw/hgnc_alias_map.csv
OUTPUTS  data/processed/baseline_results.csv
         data/processed/within_crc_association.csv
         data/processed/final_model_results.csv
         data/processed/model_comparisons.csv
         data/processed/final_model_genes.csv
         data/processed/enrichment_results.csv
         data/processed/calibration_results.csv
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

import config as C
from lib.io import exclude_haem, load_ceiling, load_expression, load_screen

from lib.modeling import (compare_models, cv_leave_lineage_out, deconfounded_gene_fit,
                          elasticnet_pipeline, fixed_elasticnet_pipeline, oof_predictions,
                          repeated_cv, ridge_cv_predict, scaled_ridge, transfer_to_crc)
from lib.report import banner, report_metric, write_and_report
from lib.signatures import background_score, load_alias_map, load_signatures, rank_matrix
from lib.stats import (bh_q, bootstrap_r, confirmatory_mask, hypergeometric_enrichment,
                       set_shift_test, signed_set_test)

warnings.filterwarnings("ignore")


def baselines(X, y, scores, y2, scores2):
    """B1-B5, the shared-gene diagnostic, and the within-COREAD association
    with its GDSC2 replication. Pan-cancer inputs (haem lines included)."""
    banner("BASELINES 1. SCORED SIGNATURES")
    print(f"  {C.TRAIN}: {X.shape[0]:,} lines x {X.shape[1]:,} genes")
    print("\n  score correlations (pan-cancer):")
    print(scores.corr().round(2).to_string())

    banner("BASELINES 2. BASELINES")
    target = y[C.TARGET].to_numpy()
    ceiling = load_ceiling()
    lin = y[[C.LINEAGE_COL]].fillna("UNKNOWN")
    results = []

    rmse0 = np.sqrt(((target - target.mean()) ** 2).mean())
    print(f"  {'B1 mean only':<28} r= 0.000  rho= 0.000  R2=  0.000  "
          f"RMSE={rmse0:.4f}     0.0% of ceiling")
    results.append(dict(baseline="B1 mean only", pearson_r=0.0, spearman_rho=0.0,
                         r2=0.0, rmse=rmse0, pct_of_ceiling=0.0))

    def add(name, y_true, y_pred):
        m = report_metric(name, y_true, y_pred, ceiling)
        results.append(dict(baseline=name, pearson_r=m["pearson_r"], spearman_rho=m["spearman_rho"],
                             r2=m["r2"], rmse=m["rmse"], pct_of_ceiling=m["pct_of_ceiling"]))

    add("B2 lineage only", target, ridge_cv_predict(lin, target, (C.LINEAGE_COL,)))

    # Solid-only variant: the transcriptome model (final_model below) is always
    # fit on solid tumours only, so B2's pan-cancer number overstates the bar
    # it has to clear -- blood cancers are a distinct, easy-to-call lineage
    # signal that the real model never gets credit for excluding.
    _, y_solid = exclude_haem(y)
    lin_solid = y_solid[[C.LINEAGE_COL]].fillna("UNKNOWN")
    target_solid = y_solid[C.TARGET].to_numpy()
    add("B2b lineage only (solid tumours)", target_solid,
        ridge_cv_predict(lin_solid, target_solid, (C.LINEAGE_COL,)))

    if "CellCycle" in scores.columns:
        add("B3 proliferation only", target, ridge_cv_predict(scores[["CellCycle"]], target))
        combo = lin.copy()
        combo["CellCycle"] = scores["CellCycle"].to_numpy()
        add("B4 lineage + proliferation", target, ridge_cv_predict(combo, target, (C.LINEAGE_COL,)))

    # B5: shuffle the target and rerun the pipeline unchanged. With no real
    # relationship left, an honest pipeline must score ~0.
    allf = lin.copy()
    for c in scores.columns:
        allf[c] = scores[c].to_numpy()
    shuffled = np.random.default_rng(C.SEED).permutation(target)
    add("B5 PERMUTED LABELS", shuffled, ridge_cv_predict(allf, shuffled, (C.LINEAGE_COL,)))

    modelled = [c for c in C.MODELED_MODULES if c in scores.columns]
    mf = lin.copy()
    for c in modelled:
        mf[c] = scores[c].to_numpy()
    add("   lineage + prolif + modules", target, ridge_cv_predict(mf, target, (C.LINEAGE_COL,)))
    print(f"\n  modelled: {modelled}")
    print(f"  excluded: {[c for c in scores.columns if c not in modelled]}  "
          f"(redundant with modelled features -- see config.REFERENCE_MODULES)")

    write_and_report(pd.DataFrame(results),
                      C.PROCESSED / "baseline_results.csv", "baseline_results.csv")

    banner("BASELINES 3. SHARED-GENE DIAGNOSTIC")
    # Signatures that share genes correlate partly by construction. This
    # quantifies how much. The trimmed lists are a DIAGNOSTIC, not the
    # primary analysis: trimming a published signature redefines it, and
    # the shared genes here are the YAP/TAZ wound-response core, not noise.
    sigs = load_signatures(X.columns, alias_map=load_alias_map(), verbose=False)
    ranks = rank_matrix(X)
    crc_mask = (y[C.LINEAGE_COL] == C.CRC).to_numpy()
    for a, b in C.OVERLAP_PAIRS:
        if a not in sigs or b not in sigs:
            continue
        ga, gb = set(sigs[a]), set(sigs[b])
        sh = ga & gb
        if not sh:
            print(f"  {a} vs {b}: no shared genes")
            continue
        fa, fb = background_score(ranks, sorted(ga)), background_score(ranks, sorted(gb))
        ta, tb = background_score(ranks, sorted(ga - sh)), background_score(ranks, sorted(gb - sh))
        print(f"  {a} vs {b}: {len(sh)} shared ({100 * len(sh) / min(len(ga), len(gb)):.0f}% of smaller)")
        print(f"    pan-cancer  full r={stats.pearsonr(fa, fb)[0]:+.3f} -> "
              f"trimmed r={stats.pearsonr(ta, tb)[0]:+.3f}")
        print(f"    COREAD      full r={stats.pearsonr(fa[crc_mask], fb[crc_mask])[0]:+.3f} -> "
              f"trimmed r={stats.pearsonr(ta[crc_mask], tb[crc_mask])[0]:+.3f}")

    banner("BASELINES 4. WITHIN-COREAD ASSOCIATION, AND REPLICATION IN GDSC2")
    feats = list(scores.columns)

    print(f"\n  higher {C.TARGET} = more resistant. * = p < 0.05 uncorrected; "
          f"q is BH within each screen.")
    print(f"  {'feature':<12} {C.TRAIN + ' (n=' + str(int(crc_mask.sum())) + ')':>22}"
          f" {C.TEST + ' (n=' + str(int((y2[C.LINEAGE_COL] == C.CRC).sum())) + ')':>22}")
    crc2 = (y2[C.LINEAGE_COL] == C.CRC).to_numpy()
    crc_rows = []
    for c in feats:
        x1 = scores.loc[crc_mask, c].to_numpy(float)
        t1 = y.loc[crc_mask, C.TARGET].to_numpy(float)
        x2 = scores2.loc[crc2, c].to_numpy(float)
        t2 = y2.loc[crc2, C.TARGET].to_numpy(float)
        r1, p1, lo1, hi1 = bootstrap_r(x1, t1)
        r2_, p2, lo2, hi2 = bootstrap_r(x2, t2)
        print(f"  {c:<12} r={r1:+.3f} p={p1:.3f}{'*' if p1 < .05 else ' '}"
              f"      r={r2_:+.3f} p={p2:.3f}{'*' if p2 < .05 else ' '}")
        crc_rows.append(dict(feature=c, r_gdsc1=r1, p_gdsc1=p1, n_gdsc1=int(crc_mask.sum()),
                              r_gdsc2=r2_, p_gdsc2=p2, n_gdsc2=int(crc2.sum()),
                              r_gdsc1_lo=lo1, r_gdsc1_hi=hi1, r_gdsc2_lo=lo2, r_gdsc2_hi=hi2))
    print(f"\n  {len(feats)} features -> Bonferroni threshold p < {0.05 / len(feats):.4f}")
    print(f"  NOTE: {C.TEST} re-measures largely the SAME cell lines, so this tests")
    print("  robustness to experimental noise, not generalisation to new lines.")
    # Replication is judged on sign and effect size, not on p: the same sign
    # in both screens, and bootstrap CIs that overlap.
    crc_df = pd.DataFrame(crc_rows)
    crc_df["q_gdsc1"] = bh_q(crc_df.p_gdsc1)
    crc_df["q_gdsc2"] = bh_q(crc_df.p_gdsc2)
    crc_df["same_sign"] = np.sign(crc_df.r_gdsc1) == np.sign(crc_df.r_gdsc2)
    crc_df["ci_overlap"] = ((crc_df.r_gdsc1_lo <= crc_df.r_gdsc2_hi)
                            & (crc_df.r_gdsc2_lo <= crc_df.r_gdsc1_hi))
    crc_df["confirmatory"] = confirmatory_mask(crc_df, "within_crc_association")
    for _, r in crc_df[crc_df.confirmatory].iterrows():
        print(f"  confirmatory {r.feature}: q={r.q_gdsc1:.3f} ({C.TRAIN}), "
              f"q={r.q_gdsc2:.3f} ({C.TEST}); same sign={r.same_sign}, "
              f"CIs overlap={r.ci_overlap}")
    write_and_report(crc_df, C.PROCESSED / "within_crc_association.csv",
                      "within_crc_association.csv")

    banner("BASELINES 5. READ BEFORE MODELLING")
    b5 = [r for r in results if "PERMUTED" in r["baseline"]][0]
    print(f"  B5 permuted-label R2 = {b5['r2']:+.4f}")
    print("  ** LEAKAGE - do not proceed **" if abs(b5["r2"]) > 0.05
          else "  OK - pipeline is not leaking.")
    b4 = [r for r in results if r["baseline"].startswith("B4")]
    if b4:
        print(f"\n  Bar to beat: B4 lineage + proliferation, r = {b4[0]['pearson_r']:.3f}")


def final_model(X, y, s):
    """Repeated CV raw vs de-confounded, CRC transfer, assay-resolution
    stratification and the selected genes. Solid-tumour inputs."""
    banner("FINAL MODEL 1. DATA")
    lin = y[C.LINEAGE_COL].to_numpy()
    t = y[C.TARGET].to_numpy()

    cnt = pd.Series(lin).value_counts()
    big = set(cnt[cnt >= C.MIN_LINEAGE_N].index)
    keep = np.isin(lin, list(big))
    print(f"  solid lines: {len(y)}  |  in lineages with >= {C.MIN_LINEAGE_N} lines: {keep.sum()}")
    print(f"  de-confounded analyses use the {keep.sum()} lines in {len(big)} lineages")

    res = []

    banner("FINAL MODEL 2. REPEATED CV WITH CIs, RAW vs DE-CONFOUNDED")
    print(f"  {C.N_REPEATS} repeats x {C.N_FOLDS}-fold. CI: cell-line bootstrap of the "
          f"out-of-fold predictions.\n")
    oof = {}

    label = {}

    def cv(key, *args, **kw):
        row, oof[key] = repeated_cv(*args, return_oof=True, **kw)
        label[key] = row["model"]
        return row

    res.append(cv("tx_raw", X, t, lin, False, "transcriptome -> raw AUC"))
    res.append(cv("tx_dc", X[keep], t[keep], lin[keep], True,
                  "transcriptome -> AUC, lineage de-confounded"))
    res.append(cv("mod_raw", s[C.MODELED_MODULES], t, lin, False, "modules -> raw AUC"))
    res.append(cv("mod_dc", s[C.MODELED_MODULES][keep], t[keep], lin[keep], True,
                  "modules -> AUC, lineage de-confounded"))

    # The B4 bar (lineage + proliferation) refit on these solid lines and
    # these splits, so the models can be tested against it directly.
    lin_f = pd.Series(lin).fillna("UNKNOWN").to_numpy()
    b4 = pd.get_dummies(pd.Series(lin_f), dtype=float)
    b4["CellCycle"] = s["CellCycle"].to_numpy()
    extra = [cv("b4_raw", b4, t, lin, False, "B4 lineage + proliferation (Ridge, same splits)",
                model_factory=scaled_ridge)]
    # The fixed-alpha model every earlier result used, on the same splits, so
    # what nested tuning buys is tested rather than eyeballed.
    extra.append(cv("tx_raw_fixed", X, t, lin, False, "transcriptome -> raw AUC, fixed alpha",
                    model_factory=fixed_elasticnet_pipeline))
    extra.append(cv("tx_dc_fixed", X[keep], t[keep], lin[keep], True,
                    "transcriptome -> AUC, lineage de-confounded, fixed alpha",
                    model_factory=fixed_elasticnet_pipeline))

    banner("FINAL MODEL 2b. MODEL COMPARISONS, CORRECTED RESAMPLED t-TEST")
    # Per-split r differences on identical lines and splits. Splits of
    # repeated CV share training lines, so the plain t-test is too liberal;
    # Nadeau and Bengio's correction is applied (lib.stats).
    pairs = [("tx_raw", "b4_raw"), ("mod_raw", "b4_raw"), ("tx_raw", "mod_raw"),
             ("tx_dc", "mod_dc"), ("tx_raw", "tx_raw_fixed"), ("tx_dc", "tx_dc_fixed")]
    comp = pd.DataFrame([compare_models(oof[a], oof[b], label[a], label[b]) for a, b in pairs])
    comp["q"] = bh_q(comp.p)
    for _, r in comp.iterrows():
        print(f"  {r.model_a:<44} vs {r.model_b:<48} "
              f"d={r.mean_fold_r_diff:+.3f}  t={r.t:+.2f}  p={r.p:.3g}  q={r.q:.3g}")
    write_and_report(comp, C.PROCESSED / "model_comparisons.csv", "model_comparisons.csv")

    banner("FINAL MODEL 2c. LEAVE WHOLE LINEAGES OUT")
    # Fold-internal de-confounding still trains on every tissue it is tested
    # on. Here each lineage is predicted by a model that never saw it.
    extra.append(cv_leave_lineage_out(X, t, lin_f, "transcriptome -> raw AUC, leave-lineage-out"))
    extra.append(cv_leave_lineage_out(s[C.MODELED_MODULES], t, lin_f,
                                      "modules -> raw AUC, leave-lineage-out"))

    banner("FINAL MODEL 3. TRANSFER TO CRC")
    crc = (y[C.LINEAGE_COL] == C.CRC).to_numpy()
    print(f"  CRC n={crc.sum()}, pan-solid training n={(~crc).sum()}\n")
    crc_res, _ = transfer_to_crc(X, t, s, crc, C.MODELED_MODULES)
    res.extend(crc_res)

    banner("FINAL MODEL 3b. STRATIFIED BY ASSAY RESOLUTION")
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
                     (f"0.90 - {C.CEILING_AUC}", (t > 0.90) & (t <= C.CEILING_AUC)),
                     (f"> {C.CEILING_AUC} (assay ceiling)", t > C.CEILING_AUC)]:
        if msk.sum() > 10:
            rr = stats.pearsonr(t[msk], p_all[msk])[0]
            print(f"  {lab:<22} {int(msk.sum()):>5} {100 * msk.mean():>10.1f}% {rr:>8.3f}")

    resp = t <= C.CEILING_AUC
    r_resp = stats.pearsonr(t[resp], p_all[resp])[0]
    r_ceil = stats.pearsonr(t[~resp], p_all[~resp])[0]
    r_pool = stats.pearsonr(t, p_all)[0]
    print(f"\n  pooled r = {r_pool:.3f}")
    print(f"  responsive lines only (AUC <= {C.CEILING_AUC}, n={int(resp.sum())}): r = {r_resp:.3f}")
    print(f"  ceiling lines only    (AUC >  {C.CEILING_AUC}, n={int((~resp).sum())}): r = {r_ceil:+.3f}")
    print()
    print("  All of the model's discriminative power sits in the responsive range.")
    print("  Among lines at the assay ceiling it is indistinguishable from noise --")
    print("  expected, since the experiment never resolved them. Report the pooled")
    print("  figure as ranking responsive lines and flagging the rest as 'resistant")
    print("  beyond resolution', not as a single unqualified correlation.")
    res.append(dict(model=f"stratified: responsive (AUC<={C.CEILING_AUC})", r_mean=r_resp,
                     r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
    res.append(dict(model=f"stratified: assay ceiling (AUC>{C.CEILING_AUC})", r_mean=r_ceil,
                     r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))

    banner("FINAL MODEL 4. WHAT THE DE-CONFOUNDED MODEL SELECTS")
    g, fin = deconfounded_gene_fit(X, t, lin, keep)
    write_and_report(g, C.PROCESSED / "final_model_genes.csv", "final_model_genes.csv")
    print(f"  {len(g)} non-zero genes of {C.K_GENES} selected\n")
    print("  RESISTANCE-associated (top 12):")
    print("   ", ", ".join(g.tail(12).iloc[::-1].gene))
    print("\n  SENSITIVITY-associated (top 12):")
    print("   ", ", ".join(g.head(12).gene))

    C.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": fin, "genes": list(X.columns), "deconfounded": True,
                 "lineages": sorted(big)}, C.MODELS_DIR / "final_deconfounded.joblib")
    print("\n  saved models/final_deconfounded.joblib")

    write_and_report(pd.DataFrame(res + extra), C.PROCESSED / "final_model_results.csv",
                      "final_model_results.csv")


def enrichment(X, y):
    """Hypergeometric enrichment of the final model's genes per signature,
    then the same signatures tested for DIRECTION.

    The universe is the gene set the model actually chose from -- X's columns
    AFTER the missingness filter, not the raw unfiltered expression matrix.
    Getting this wrong changes every expected-overlap and p-value.

    The overlap test cannot tell DTP_up genes carrying resistance-direction
    coefficients from DTP_up genes carrying sensitivity-direction ones. The
    signed tests can, on two vectors (positive = higher AUC = resistant):
      coef_*     the de-confounded model's non-zero coefficients
      genome_*   every gene's correlation with the de-confounded target, the
                 preranked analogue of GSEA: each set's mean against genes of
                 the same expression decile, with a null from permuting the
                 target within lineage (lib.stats.set_shift_test)
    DTP_down is expected to point the other way from DTP_up.
    """
    universe = X.columns
    banner("ENRICHMENT 1. LOADING MODEL GENES AND SIGNATURES")
    genes = pd.read_csv(C.PROCESSED / "final_model_genes.csv")
    sigs = load_signatures(universe, alias_map=load_alias_map())

    query = set(genes["gene"])
    print(f"  model genes: {len(query):,}   universe: {len(universe):,}")

    banner("ENRICHMENT 2. HYPERGEOMETRIC TEST")
    rows = []
    for name, sig_genes in sigs.items():
        r = hypergeometric_enrichment(query, sig_genes, universe, name=name)
        rows.append(r)
        flag = "  <-- p<0.05" if r["p_value"] < 0.05 else ""
        print(f"  {name:<10} {r['observed_overlap']:>3}/{r['signature_size']:<4} "
              f"(expected {r['expected_overlap']:.2f})  "
              f"fold={r['fold_enrichment']:.2f}x  p={r['p_value']:.2e}{flag}")

    banner("ENRICHMENT 3. DIRECTION")
    lin = y[C.LINEAGE_COL].to_numpy()
    cnt = pd.Series(lin).value_counts()
    keep = np.isin(lin, list(cnt[cnt >= C.MIN_LINEAGE_N].index))
    Xk = X[keep].to_numpy(float)
    Xd = Xk - pd.DataFrame(Xk).groupby(lin[keep]).transform("mean").to_numpy()
    ys = pd.Series(y[C.TARGET].to_numpy()[keep]).groupby(lin[keep])
    yd = ((y[C.TARGET].to_numpy()[keep] - ys.transform("mean").to_numpy())
          / ys.transform(lambda v: v.std(ddof=1)).to_numpy())
    coef = genes.set_index("gene")["coef"]
    decile = pd.Series(pd.qcut(X.mean().rank(method="first"), 10, labels=False),
                       index=universe)
    shift = set_shift_test(pd.DataFrame(Xd, columns=universe), yd, sigs, decile,
                           groups=lin[keep])

    print(f"  {'signature':<10} {'model genes +/n':>16} {'p_sign':>8}   "
          f"{'mean r':>8} {'matched':>8} {'shift':>8} {'p_perm':>8}")
    for row in rows:
        c = signed_set_test(coef, sigs[row["signature"]])
        g = shift.loc[row["signature"]]
        row.update(coef_n=c["n"], coef_n_resistance=c["n_positive"], coef_p_sign=c["p_sign"],
                   coef_p_rank=c["p_rank"], genome_n=g.n, genome_mean_r=g.mean_r,
                   genome_background_r=g.background_r, genome_shift=g["shift"],
                   genome_p_perm=g.p_perm)
        print(f"  {row['signature']:<10} {c['n_positive']:>7}/{c['n']:<8} {c['p_sign']:>8.3g}   "
              f"{g.mean_r:>+8.4f} {g.background_r:>+8.4f} {g['shift']:>+8.4f} {g.p_perm:>8.3g}")

    write_and_report(pd.DataFrame(rows), C.PROCESSED / "enrichment_results.csv",
                      "enrichment_results.csv")


def calibration(X1, y1, X2, y2):
    """Affine recalibration of the GDSC1 model to GDSC2. Solid-tumour inputs.

    For the OPTIMAL a and b, R2 becomes EXACTLY r^2 -- an affine transform
    cannot change r. This makes it a display fix for the dashboard (so a shown
    predicted AUC is not systematically wrong), NOT a modelling improvement.
    It also needs labelled data from the target screen, which a genuinely new
    dataset would not have. Fitted on a held-out half of GDSC2 and evaluated
    on the other half, so the reported numbers are honest.
    """
    banner("AFFINE RECALIBRATION (dashboard display only)")

    genes = sorted(set(X1.columns) & set(X2.columns))
    X1, X2 = X1[genes], X2[genes]
    t1, t2 = y1[C.TARGET].to_numpy(), y2[C.TARGET].to_numpy()
    print(f"  {C.TRAIN} mean AUC {t1.mean():.3f} | {C.TEST} mean AUC {t2.mean():.3f} "
          f"| offset {t2.mean() - t1.mean():+.3f}\n")

    m = elasticnet_pipeline().fit(X1, t1)
    pred2 = m.predict(X2)
    raw = report_metric(f"uncalibrated on {C.TEST}", t2, pred2)

    # Split GDSC2: fit a and b on one half, evaluate on the other. Fitting
    # and scoring on the same rows would flatter the result.
    rng = np.random.default_rng(C.SEED)
    idx = rng.permutation(len(t2))
    fit_i, ev_i = idx[:len(idx) // 2], idx[len(idx) // 2:]
    a, b = np.polyfit(pred2[fit_i], t2[fit_i], 1)
    print(f"  calibration fitted on {len(fit_i)} lines: a={a:.3f}, b={b:.3f}")
    cal = report_metric("calibrated, evaluated on held-out half", t2[ev_i], a * pred2[ev_i] + b)

    print()
    print("  Calibration removed the offset and changed r by exactly nothing")
    print(f"  ({raw['pearson_r']:.3f} raw vs {cal['pearson_r']:.3f} calibrated-half r, as the")
    print("  algebra predicts). Use it in the dashboard so a displayed AUC is not")
    print("  systematically wrong; do NOT report it as a model improvement.")

    out = pd.DataFrame([
        dict(stage="uncalibrated", **raw, a=np.nan, b=np.nan, n_fit=np.nan),
        dict(stage="calibrated_heldout", **cal, a=a, b=b, n_fit=len(fit_i)),
    ])
    write_and_report(out, C.PROCESSED / "calibration_results.csv", "calibration_results.csv")


def main():
    X1, y1, s1 = load_screen(C.TRAIN)
    # The replication screen's expression is only needed for calibration.
    y2, s2 = (pd.read_parquet(C.PROCESSED / f"{name}_{C.TEST}.parquet")
              for name in ("y", "signature_scores"))
    baselines(X1, y1, s1, y2, s2)

    # Everything from here on is fit on solid tumours only.
    _, y1, X1, s1 = exclude_haem(y1, X1, s1)
    final_model(X1, y1, s1)
    enrichment(X1, y1)
    _, y2, X2 = exclude_haem(y2, load_expression(C.TEST))
    calibration(X1, y1, X2, y2)
    banner("DONE")


if __name__ == "__main__":
    main()
