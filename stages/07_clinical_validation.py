"""Do the scores associate with FOLFOX response in patients?

Five GEO cohorts. Logistic regression of response on each score, unadjusted and
adjusted for purity, CMS, MSI and study. Prognostic only: every patient got FOLFOX.

Inputs:  data/processed/clinical_expr_*.csv, clinical_covariates.csv
         (from stages/prep/clinical_prep.R)
Outputs: data/processed/clinical_validation.csv, reports/clinical_validation.md,
         figures/clinical_validation_pca.png
Run:     Rscript stages/prep/clinical_prep.R && python stages/07_clinical_validation.py
"""

from pathlib import Path
import sys
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config  # not `as C`: the statsmodels formulas below use patsy's C()
from lib.cohorts import RECOVERY_MIN, score_cohort
from lib.report import banner, write_and_report, write_report
from lib import signatures

FEATURES = ["CBC", "CellCycle", "DTP", "Fetal", "MYC", "RSC", "ModelDerived"]
GPL570_ADJUST = "+ purity + C(cms) + C(msi_ntp) + C(study)"
GPL6480_ADJUST = "+ C(cms) + C(msi_ntp)"


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


def fit_logit(df, formula, label):
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = smf.logit(formula, data=df).fit(disp=0)
        if not model.mle_retvals.get("converged", True):
            return {"status": "did not converge", "n": int(model.nobs)}
        coef = model.params.get("z", np.nan)
        p = model.pvalues.get("z", np.nan)
        ci_lo, ci_hi = model.conf_int().loc["z"] if "z" in model.params.index else (np.nan, np.nan)
        return {"status": "ok", "n": int(model.nobs), "coef": coef,
                "OR_per_SD": float(np.exp(coef)), "OR_CI95_lo": float(np.exp(ci_lo)),
                "OR_CI95_hi": float(np.exp(ci_hi)), "p": p}
    except Exception as e:
        return {"status": f"fit failed: {e}", "n": len(df)}


def run_cohort_tests(rank_df, cov_df, feature, adjust_formula_extra, cohort_tag):
    d = cov_df.copy()
    d["z"] = zscore(rank_df.loc[d.index, feature])
    d["response"] = d["response"].astype(int)
    unadj = fit_logit(d, "response ~ z", f"{cohort_tag}/{feature}/unadjusted")
    if adjust_formula_extra:
        adj_df = d.dropna(subset=["z"] + [c for c in ["purity"] if "purity" in adjust_formula_extra])
        adj = fit_logit(adj_df, f"response ~ z {adjust_formula_extra}", f"{cohort_tag}/{feature}/adjusted")
    else:
        adj = {"status": "no covariates available", "n": len(d)}
    return unadj, adj


def pca_2d(expr_genes_by_samples, top_n=2000):
    v = expr_genes_by_samples.var(axis=1).sort_values(ascending=False)
    top = expr_genes_by_samples.loc[v.index[:top_n]]
    x = (top.T - top.T.mean()) / top.T.std(ddof=0).replace(0, 1)
    x = x.fillna(0.0)
    u, s, _ = np.linalg.svd(x.to_numpy(), full_matrices=False)
    scores = u[:, :2] * s[:2]
    var_explained = (s ** 2) / (s ** 2).sum()
    return pd.DataFrame(scores, index=x.index, columns=["PC1", "PC2"]), var_explained[:2]


def fmt_row(r):
    if r["status"] != "ok":
        return f"n={r['n']}, {r['status']}"
    flag = "  **LOW RECOVERY**" if r["low_recovery"] else ""
    sig = "*" if r["p"] < 0.05 else ""
    return (f"n={r['n']}, OR/SD={r['OR_per_SD']:.2f} "
            f"(95% CI {r['OR_CI95_lo']:.2f}-{r['OR_CI95_hi']:.2f}), p={r['p']:.3f}{sig}{flag}")


def main():
    banner("LOADING R-SIDE OUTPUTS (stages/prep/clinical_prep.R)")

    for f in ["clinical_expr_gpl570_combat.csv", "clinical_expr_gpl570_precombat.csv",
              "clinical_expr_gse104645.csv", "clinical_covariates.csv"]:
        if not (config.PROCESSED / f).exists():
            print(f"  !! {config.PROCESSED / f} missing. Run stages/prep/clinical_prep.R first. Stopping.")
            sys.exit(1)

    expr570_post = pd.read_csv(config.PROCESSED / "clinical_expr_gpl570_combat.csv", index_col="gene")
    expr570_pre = pd.read_csv(config.PROCESSED / "clinical_expr_gpl570_precombat.csv", index_col="gene")
    expr6480 = pd.read_csv(config.PROCESSED / "clinical_expr_gse104645.csv", index_col="gene")
    cov = pd.read_csv(config.PROCESSED / "clinical_covariates.csv", index_col="sample_id")

    cov570 = cov[cov.platform == "GPL570"].copy()
    cov6480 = cov[cov.platform == "GPL6480"].copy()
    print(f"  GPL570 cohort:  {expr570_post.shape[0]} genes x {expr570_post.shape[1]} samples, "
          f"{cov570.study.nunique()} studies")
    print(f"  GPL6480 cohort: {expr6480.shape[0]} genes x {expr6480.shape[1]} samples, 1 study (GSE104645)")

    banner("SCORING: GPL570 (primary, ComBat-corrected, 4 studies)")
    alias_map = signatures.load_alias_map()
    rank570, ss570, recov570 = score_cohort(
        expr570_post.T, metadata=cov570, platform="clinical_GPL570_combined",
        alias_map=alias_map, min_recovery=None,
    )
    low570 = {k: v for k, v in recov570.items() if v < RECOVERY_MIN}
    print(f"\n  Below {RECOVERY_MIN:.0f}% recovery on GPL570 (reported, not hidden): {low570}")

    banner("SCORING: GPL6480 (secondary, GSE104645 alone)")
    rank6480, ss6480, recov6480 = score_cohort(
        expr6480.T, metadata=cov6480, platform="clinical_GSE104645_agilent_secondary",
        alias_map=alias_map, min_recovery=None,
    )
    low6480 = {k: v for k, v in recov6480.items() if v < RECOVERY_MIN}
    print(f"\n  Below {RECOVERY_MIN:.0f}% recovery on GPL6480 (reported, not hidden): {low6480}")

    LOW_FLAG = {"GPL570": set(), "GPL6480": set()}
    for feat in FEATURES:
        # DTP is the combined DTP_up/DTP_down feature
        if feat == "DTP":
            if low570.get("DTP_up") is not None or low570.get("DTP_down") is not None:
                LOW_FLAG["GPL570"].add("DTP")
            if low6480.get("DTP_up") is not None or low6480.get("DTP_down") is not None:
                LOW_FLAG["GPL6480"].add("DTP")
        else:
            if feat in low570: LOW_FLAG["GPL570"].add(feat)
            if feat in low6480: LOW_FLAG["GPL6480"].add(feat)

    banner("ASSOCIATION TESTS")

    results = []

    for feat in FEATURES:
        u570, a570 = run_cohort_tests(rank570, cov570, feat, GPL570_ADJUST, "GPL570")
        u6480, a6480 = run_cohort_tests(rank6480, cov6480, feat, GPL6480_ADJUST, "GPL6480")
        for cohort, unadj, adj, low_set in [("GPL570", u570, a570, LOW_FLAG["GPL570"]),
                                              ("GPL6480", u6480, a6480, LOW_FLAG["GPL6480"])]:
            for kind, res in [("unadjusted", unadj), ("adjusted", adj)]:
                row = {"cohort": cohort, "feature": feat, "model": kind,
                       "low_recovery": feat in low_set}
                row.update(res)
                results.append(row)

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))

    banner("BATCH-STRUCTURE PCA (GPL570, before vs after ComBat)")

    pre_pc, pre_var = pca_2d(expr570_pre)
    post_pc, post_var = pca_2d(expr570_post)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    studies = sorted(cov570.study.unique())
    colors = dict(zip(studies, plt.cm.tab10.colors))
    for ax, pc, var, title in [(axes[0], pre_pc, pre_var, "Before ComBat"),
                                (axes[1], post_pc, post_var, "After ComBat")]:
        for s in studies:
            idx = cov570.index[cov570.study == s]
            idx = [i for i in idx if i in pc.index]
            ax.scatter(pc.loc[idx, "PC1"], pc.loc[idx, "PC2"], label=s, color=colors[s], s=18, alpha=0.8)
        ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)")
        ax.set_title(title)
    axes[1].legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), fontsize=8, title="study")
    fig.suptitle("GPL570 clinical cohort: batch structure, top 2000 variable genes")
    fig.tight_layout()
    config.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig_path = config.FIGURES_DIR / "clinical_validation_pca.png"
    fig.savefig(fig_path, dpi=150)
    plt.close(fig)
    print(f"  -> {fig_path}")

    banner("WRITING OUTPUTS")
    write_and_report(results_df, config.PROCESSED / "clinical_validation.csv")

    lines = []
    lines.append("# Clinical cohort analysis\n")
    lines.append(
        "**This is a prognostic analysis, not a predictive one.** Every sample in "
        "both cohorts below received FOLFOX (or FOLFOX+bevacizumab); there is no "
        "untreated or differently-treated comparator arm. An association here "
        "can only mean a module's score correlates with how FOLFOX-treated "
        "patients did, not that the score identifies who benefits FROM FOLFOX "
        "specifically -- that claim needs a comparator this data does not have.\n"
    )
    lines.append("## Cohorts\n")
    lines.append(
        f"**Primary (GPL570, ComBat-corrected):** {cov570.shape[0]} patients across "
        f"{cov570.study.nunique()} studies (GSE28702 n=83, GSE19860 n=40, GSE69657 n=30, "
        "GSE72970 n=36 of 124 total after restricting to its FOLFOX/FOLFOX+bevacizumab "
        "arm). All on mFOLFOX6, FOLFOX4, or FOLFOX, confirmed against each series' own "
        "GEO summary text, not assumed from column labels.\n\n"
        f"**Secondary (GPL6480, GSE104645, not pooled):** {cov6480.shape[0]} patients, "
        "restricted to its FOLFOX/FOLFOX+Bev arm (97 of 193 total). Scored with the "
        "same rank-based, platform-robust method (lib.cohorts.score_cohort) but "
        "kept as a separate cohort: Agilent two-colour log-ratio data is a different "
        "measurement technology from the other four studies' Affymetrix intensities, "
        "not a batch effect ComBat is designed to remove.\n"
    )
    lines.append("## Cohort definitions, checked against source\n")
    lines.append(
        "The five series were first assumed to be "
        "\"All GPL570, roughly 132 patients with FOLFOX responder/non-responder "
        "labels.\" Checked directly against GEO (see stages/prep/clinical_prep.R "
        "for the full per-series audit):\n\n"
        "- **GSE104645 is GPL6480 (Agilent), not GPL570.** Kept as a separate, "
        "non-pooled secondary cohort rather than silently dropped or force-combined.\n"
        f"- The actual n after restricting each series to its FOLFOX-treated arm is "
        f"**{cov570.shape[0]} (GPL570) + {cov6480.shape[0]} (GPL6480) = "
        f"{cov570.shape[0]+cov6480.shape[0]}**, not \"roughly 132\" -- GSE72970 and "
        "GSE104645 are SuperSeries pooling several regimens (FOLFIRI is GSE72970's "
        "most common arm, not FOLFOX), so both needed a regimen filter.\n"
        "- GSE19860's response field is literally labelled `FL_Responder`/"
        "`FL_Non_responder`. Checked against the series' own title (\"Prediction of "
        "response to FOLFOX\") and summary (\"All patients ... received modified "
        "FOLFOX6\") before treating it as a FOLFOX response call -- it is one, the "
        "\"FL\" is the submitters' label choice, not a different regimen.\n"
    )
    lines.append("## Covariates\n")
    lines.append(
        "- **Purity**: `estimate` package (Yoshihara et al. 2013), computed per "
        "study on that study's own pre-ComBat log2 matrix (purity estimated "
        "after batch correction would be estimating an artefact of the "
        "correction, not the tissue). ESTIMATE's calibrated `TumorPurity` "
        "conversion is only defined for `platform=\"affymetrix\"` in the package's "
        "own source -- **GPL6480 (Agilent) has no purity covariate as a result**, "
        "reported as missing rather than estimated from a formula ESTIMATE itself "
        "does not apply to that platform.\n"
        "- **CMS subtype**: CMScaller (Guinney et al. 2015 templates, Hoshida's NTP "
        "algorithm), per study, HGNC-symbol space.\n"
        "- **MSI status**: no series reports a molecular MSI/MMR test in its GEO "
        "metadata (checked directly, none of the 5 have such a field). Reported "
        "instead as an **expression-derived MSI-like NTP classification** "
        "(CMScaller's `templates.MSI`), confidence-gated at FDR<0.05 (below that, "
        "'Indeterminate') -- an expression proxy, not a substitute for PCR/IHC "
        "testing, and never presented as one.\n"
    )
    lines.append("## Recovery: the model-derived signature falls below 80% on both platforms\n")
    fmt_low = lambda d: ", ".join(f"{k}={v:.1f}%" for k, v in d.items()) if d else "none below 80%"
    lines.append(
        f"GPL570: {fmt_low(low570)}. "
        f"GPL6480: {fmt_low(low6480)}. Both calls used "
        "`score_cohort(..., min_recovery=None)` so the required curated-module "
        "scoring could proceed rather than aborting on ModelDerived alone (the same "
        "tension external scoring met on HCT116 RNA-seq, now recurring on two "
        "microarray platforms). Every affected feature below is marked "
        "**LOW RECOVERY** in its result and should be read with that caveat, not "
        "trusted at the same level as a signature that cleared 80%. That the "
        "model-derived gene NAMES fail to transfer at the required threshold on "
        "RNA-seq (64.4%), Affymetrix (73.4%), AND Agilent (69.0%) is "
        "itself a finding: this is upstream of, and independent from, the "
        "already-documented reason the trained ElasticNet's *coefficients* can't "
        "be applied outside GDSC.\n"
    )
    lines.append("## Batch structure\n")
    lines.append(
        "![GPL570 batch structure, before and after ComBat](../../../figures/clinical_validation_pca.png)\n\n"
        "PCA on the top 2000 most-variable genes, GPL570 cohort only. Before "
        "correction, samples separate visibly by study; after ComBat "
        "(`batch=study`, no protected covariate), the four studies mix. This is "
        "the batch-structure evidence.\n\n"
        "**Residual structure noted, not silently smoothed over:** before "
        "correction, GSE19860 itself splits into two tight subclusters "
        "(20/20 samples) that alone explain 82.4% of top-2000-gene variance -- "
        "far more than the separation between the four studies. This split is "
        "not fully resolved by ComBat (a small GSE19860-only cluster remains "
        "visible after correction) because ComBat's `batch` term here is "
        "`study`, not a sub-batch within GSE19860. Checked against every "
        "per-sample field GSE19860 exposes in GEO (no scan-date, hybridization "
        "batch, or lot field beyond series-level metadata) and against "
        "response, purity, CMS, and MSI-NTP (roughly even 10/10-ish splits on "
        "each, no clean 20/0 separation) -- none of these explain the split. "
        "Left as an open, reported limitation rather than a guessed-at fix: "
        "GSE19860 results should be read knowing this structure exists in the "
        "expression data and is not accounted for by the current covariates.\n"
    )
    lines.append("## Results, unadjusted vs adjusted, by cohort (never pooled across modules)\n")
    for cohort, adjust_desc in [("GPL570", "purity + CMS + MSI-NTP + study"),
                                  ("GPL6480", "CMS + MSI-NTP (no purity: unavailable on this platform; no study: single study)")]:
        lines.append(f"### {cohort}  (adjusted for: {adjust_desc})\n")
        sub = results_df[results_df.cohort == cohort]
        tbl = ["| Feature | Unadjusted | Adjusted |", "|---|---|---|"]
        for feat in FEATURES:
            u = sub[(sub.feature == feat) & (sub.model == "unadjusted")].iloc[0]
            a = sub[(sub.feature == feat) & (sub.model == "adjusted")].iloc[0]
            tbl.append(f"| {feat} | {fmt_row(u)} | {fmt_row(a)} |")
        lines.append("\n".join(tbl) + "\n")
    lines.append("## Traps checked\n")
    lines.append(
        "- **The stroma confound**. Purity and CMS are both "
        "in the adjusted model for exactly this reason; compare each feature's "
        "unadjusted vs adjusted row above rather than reading either alone.\n"
        "- Every module reported separately, never pooled into one combined score.\n"
        "- The model-derived signature is tested the same way as the curated "
        "modules (step 5), with its low recovery on both platforms surfaced, not "
        "hidden.\n"
        "- No separate signature-scoring logic: `lib.cohorts.score_cohort()` "
        "is called exactly as its own regression test calls it.\n"
    )

    write_report("clinical_validation.md", "\n".join(lines))

    banner("DONE")


if __name__ == "__main__":
    main()
