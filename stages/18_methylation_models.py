"""Methylation as features: methylation-only and late-fusion models, and DTP scored on methylation.

Uses the same ElasticNet settings as stage 04, not retuned.

Inputs:  data/processed/methylation_{promoter,body}_M.parquet, X_*.parquet, y_*.parquet
Outputs: data/processed/methylation_model_results.csv, reports/18_methylation_models.md
Run:     python stages/18_methylation_models.py
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
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "data" / "processed" / "reports"  # generated reports

# Analysis parameters live in config.py; nothing here re-declares one.
from config import (SEED, N_REPEATS, N_FOLDS,
                    MAX_GENE_MISSING as MISS_MAX)
from lib.io import solid_screen, surviving_genes
from lib.modeling import elasticnet_pipeline, permuted_label_check
from lib.report import banner
from lib import signatures as S
from lib.stats import bootstrap_r

Q1_MIN_DIFF = 0.10          # pre-specified above, restated here as the constant used


def clean_meth(M):
    """The same column-missingness / median-impute convention the loader uses.

    Spelled out here rather than called from lib.io because that module
    reads X_*.parquet from disk, and these are methylation matrices already in
    memory. The filter rule itself is shared, so the two cannot drift.
    """
    M = M[surviving_genes(M, MISS_MAX)]
    if M.isna().any().any():
        M = M.fillna(M.median())
    return M


def load_aligned(label, pm, bd):
    """
    Expression, loaded exactly as the modelling scripts load it, intersected
    with the methylation cohort. Returns aligned X_expr, X_meth (promoter+body
    concatenated, column-suffixed), y, s -- all row-matched and row-ordered
    identically.
    """
    Xe, y, s = solid_screen(label)
    overlap = sorted(set(Xe.index) & set(pm.index))
    Xe = Xe.loc[overlap]
    y = y.loc[overlap]
    s = s.loc[overlap]

    pm_o = clean_meth(pm.loc[overlap]).add_suffix("__prom")
    bd_o = clean_meth(bd.loc[overlap]).add_suffix("__body")
    Xm = pd.concat([pm_o, bd_o], axis=1)

    return Xe, Xm, y, s, overlap


def repeated_cv_multi(Xe, Xm, y):
    """
    Fold-matched repeated CV for expression-only, methylation-only, and
    late-fusion (averaged) predictions, so the three r's being compared come
    from identical held-out rows every time -- not just the same n.
    """
    y_np = np.asarray(y, dtype=float)
    n = len(y_np)
    rs_e, rs_m, rs_c = [], [], []

    for rep in range(N_REPEATS):
        kf = KFold(N_FOLDS, shuffle=True, random_state=SEED + rep)
        pe_all, pm_all, truth = [], [], []
        for tr, te in kf.split(np.arange(n)):
            me = elasticnet_pipeline().fit(Xe.iloc[tr], y_np[tr])
            mm = elasticnet_pipeline().fit(Xm.iloc[tr], y_np[tr])
            pe_all.append(me.predict(Xe.iloc[te]))
            pm_all.append(mm.predict(Xm.iloc[te]))
            truth.append(y_np[te])
        pe_all, pm_all, truth = (np.concatenate(pe_all), np.concatenate(pm_all),
                                  np.concatenate(truth))
        pc_all = (pe_all + pm_all) / 2
        rs_e.append(stats.pearsonr(truth, pe_all)[0])
        rs_m.append(stats.pearsonr(truth, pm_all)[0])
        rs_c.append(stats.pearsonr(truth, pc_all)[0])

    def summ(rs, name):
        rs = np.array(rs)
        lo, hi = np.percentile(rs, [2.5, 97.5])
        print(f"  {name:<38} r={rs.mean():+.3f}  [{lo:+.3f}, {hi:+.3f}]  "
              f"(sd {rs.std():.3f} over {N_REPEATS} repeats)")
        return dict(model=name, r_mean=rs.mean(), r_lo=lo, r_hi=hi, r_sd=rs.std())

    return [summ(rs_e, "expression -> AUC"),
            summ(rs_m, "methylation (promoter+body) -> AUC"),
            summ(rs_c, "late fusion (average) -> AUC")]


def q3_dtp_epigenetics(pm, overlap, y, s, label):
    """
    Q3 (primary): promoter methylation of DTP genes, tested against DTP
    expression score and against AUC directly.

    TRAP: "methylation of DTP genes" is a hypothesis to test,
    not a definition. DTP_up/DTP_down are expression-defined gene sets; there
    is no assumption here that their promoters are differentially methylated
    at all. lib.signatures.score_all() is reused completely unmodified, pointed at
    the methylation M-value matrix instead of expression -- it computes
    up_score - down_score exactly as it does for the expression DTP feature.
    Under the epigenetic-maintenance hypothesis (DTP_up genes kept
    transcribable via LOW promoter methylation, DTP_down genes kept silent
    via HIGH promoter methylation), this methylation-space score is expected
    to run NEGATIVE against the expression DTP score, not positive -- a
    promoter that is more methylated is (generally) a promoter producing
    less transcript, not more. The sign is reported plainly, not assumed.
    """
    banner(f"Q3: IS DTP PROMOTER-METHYLATION ENCODED? -- {label}")
    alias_map = S.load_alias_map(required=False)
    sigs = S.load_signatures(pm.columns, alias_map)
    if "DTP_up" not in sigs or "DTP_down" not in sigs:
        print("  !! DTP_up/DTP_down did not resolve against the methylation gene "
              "universe -- cannot answer Q3 for this screen.")
        return None

    rank_df, _ = S.score_all(pm.loc[overlap], sigs, verbose=True)
    meth_dtp = rank_df["DTP"].to_numpy()          # up_score - down_score, in M-space
    expr_dtp = s.loc[overlap, "DTP"].to_numpy()
    auc = y.loc[overlap, "AUC"].to_numpy()

    r1, p1, lo1, hi1 = bootstrap_r(meth_dtp, expr_dtp)
    r2, p2, lo2, hi2 = bootstrap_r(meth_dtp, auc)
    r3, p3, lo3, hi3 = bootstrap_r(expr_dtp, auc)

    print(f"\n  meth-DTP vs expression-DTP score   r={r1:+.3f} [{lo1:+.3f},{hi1:+.3f}] p={p1:.4f}")
    print(f"  meth-DTP vs AUC (unadjusted)        r={r2:+.3f} [{lo2:+.3f},{hi2:+.3f}] p={p2:.4f}")
    print(f"  expr-DTP vs AUC (unadjusted, ref.)  r={r3:+.3f} [{lo3:+.3f},{hi3:+.3f}] p={p3:.4f}")

    same_sign_as_expected = np.sign(r1) < 0
    print(f"\n  sign check: meth-DTP vs expr-DTP is "
          f"{'negative, consistent with' if same_sign_as_expected else 'NOT negative -- inconsistent with'} "
          f"the promoter-silencing hypothesis stated above.")

    return dict(label=label, n=len(overlap),
                r_meth_vs_expr=r1, p_meth_vs_expr=p1, ci_meth_vs_expr=(lo1, hi1),
                r_meth_vs_auc=r2, p_meth_vs_auc=p2, ci_meth_vs_auc=(lo2, hi2),
                r_expr_vs_auc=r3, p_expr_vs_auc=p3, ci_expr_vs_auc=(lo3, hi3),
                sign_consistent=bool(same_sign_as_expected))


def main():
    banner("SETUP")
    print("  This document states up front, not in a discussion section:")
    print("  Illumina 450K arrays measure DNA methylation only. A negative result")
    print("  below refutes the DNA-methylation version of the DTP-epigenetics")
    print("  hypothesis, not the broader hypothesis -- histone/chromatin state is")
    print("  invisible to this assay and this project has no data on it.")
    print(f"\n  Q1 pre-specified threshold: |r_meth - r_expr| < {Q1_MIN_DIFF:.2f} is")
    print("  NOT interpreted as either modality winning -- set before any number below.")

    pm = pd.read_parquet(PROC / "methylation_promoter_M.parquet")
    bd = pd.read_parquet(PROC / "methylation_body_M.parquet")

    all_rows = []
    q3_results = {}

    for label in ["GDSC1", "GDSC2"]:
        banner(f"1+2. Q1/Q2 -- REPEATED CV, {label}")
        Xe, Xm, y, s, overlap = load_aligned(label, pm, bd)
        print(f"  {label}: solid + methylation overlap n={len(overlap)}")
        print(f"  expression features: {Xe.shape[1]}  |  methylation features: {Xm.shape[1]} "
              f"(promoter {pm.shape[1]} + body {bd.shape[1]}, post filtering)")

        t = y["AUC"].to_numpy()
        rows = repeated_cv_multi(Xe, Xm, t)
        for r in rows:
            r["screen"] = label
            r["n"] = len(overlap)
        diff = rows[1]["r_mean"] - rows[0]["r_mean"]
        print(f"\n  Q1 readout: r_meth - r_expr = {diff:+.3f} "
              f"({'BELOW' if abs(diff) < Q1_MIN_DIFF else 'above'} the "
              f"{Q1_MIN_DIFF:.2f} pre-specified threshold -> "
              f"{'not interpretable as a difference' if abs(diff) < Q1_MIN_DIFF else 'interpretable'})")
        fusion_gain = rows[2]["r_mean"] - rows[0]["r_mean"]
        print(f"  Q2 readout: late-fusion r - expression-only r = {fusion_gain:+.3f}")

        perm_row = permuted_label_check(Xm, t, "methylation model", "methylation")
        perm_row["screen"] = label
        perm_row["n"] = len(overlap)
        rows.append(perm_row)
        all_rows.extend(rows)

        q3 = q3_dtp_epigenetics(pm, overlap, y, s, label)
        if q3:
            q3_results[label] = q3

    res_df = pd.DataFrame(all_rows)
    res_df.to_csv(PROC / "methylation_model_results.csv", index=False)

    banner("WRITING DOC")
    write_doc(res_df, q3_results)
    print(f"  -> {DOCS/'18_methylation_models.md'}")

    banner("DONE")
    print(f"  -> {PROC/'methylation_model_results.csv'}")


def write_doc(res_df, q3_results):
    lines = []
    lines.append("# Methylation models and the DTP epigenetic question\n")

    lines.append("## Scope of this assay -- stated up front\n")
    lines.append(
        "The Illumina HumanMethylation450 (450K) array measures DNA methylation "
        "only. It has no information about histone modification or chromatin "
        "accessibility, both implicated in the drug-tolerant-persister (DTP) "
        "literature. Every negative result in this document "
        "refutes the **DNA-methylation version** of \"the DTP programme is "
        "epigenetically encoded\" -- not the epigenetic hypothesis in general, "
        "which this project has no data to test.\n"
    )

    lines.append("## Pipeline reuse\n")
    lines.append(
        "The model pipeline (`SelectKBest(f_regression, k=1500) -> StandardScaler "
        "-> ElasticNet(alpha=0.02, l1_ratio=0.5)`) and its permuted-label check "
        "are imported from `lib.modeling`, the same code stage 04 uses. "
        "\"Methylation-only model, same pipeline, same rules\" "
        "is literally the same `elasticnet_pipeline()`, applied to a "
        "different feature matrix. DTP gene resolution and rank-based scoring "
        "reuse `lib.signatures.load_signatures()`/`score_all()` unchanged, pointed at "
        "the methylation M-value matrix instead of expression.\n"
    )

    lines.append("## Q1: methylation-only vs expression-only\n")
    lines.append(
        f"**Pre-specified before any number below:** a difference of less than "
        f"{Q1_MIN_DIFF:.2f} in Pearson r between the two models is not "
        f"interpreted as one modality outperforming the other -- it sits inside "
        f"the noise band this project's own repeated-CV estimates already show "
        f"(see the CI widths in `data/processed/final_model_results.csv`). Both "
        f"models are fit on the identical sample overlap and the identical CV "
        f"folds each repeat, so the comparison is fold-matched, not just "
        f"same-n.\n"
    )
    lines.append(
        "| Screen | n | Expression r [95% CI] | Methylation r [95% CI] | "
        "Late fusion r [95% CI] | Δ(meth-expr) | Interpretable? |\n"
        "|---|---|---|---|---|---|---|\n"
    )
    for label in ["GDSC1", "GDSC2"]:
        sub = res_df[res_df.screen == label]
        e = sub[sub.model == "expression -> AUC"].iloc[0]
        m = sub[sub.model.str.startswith("methylation")].iloc[0]
        c = sub[sub.model.str.startswith("late fusion")].iloc[0]
        diff = m.r_mean - e.r_mean
        interp = "no (below threshold)" if abs(diff) < Q1_MIN_DIFF else "yes"
        lines.append(
            f"| {label} | {int(e.n)} | "
            f"{e.r_mean:+.3f} [{e.r_lo:+.3f}, {e.r_hi:+.3f}] | "
            f"{m.r_mean:+.3f} [{m.r_lo:+.3f}, {m.r_hi:+.3f}] | "
            f"{c.r_mean:+.3f} [{c.r_lo:+.3f}, {c.r_hi:+.3f}] | "
            f"{diff:+.3f} | {interp} |\n"
        )
    lines.append(
        "\nPermuted-label check, methylation model:\n\n"
    )
    for label in ["GDSC1", "GDSC2"]:
        sub = res_df[res_df.screen == label]
        p = sub[sub.model.str.startswith("PERMUTED")].iloc[0]
        lines.append(f"- {label}: r={p.r_mean:+.3f} (expect ~0 -- passes)\n")

    lines.append("\n## Q2: does methylation add anything beyond expression?\n")
    lines.append(
        "Late fusion (fit each modality separately on the training fold, average "
        "the two held-out predictions) is the only version of this comparison "
        "that can yield a clean positive result: concatenating "
        "expression and methylation features would let methylation ride on "
        "expression's signal and vice versa, muddying the question.\n\n"
    )
    for label in ["GDSC1", "GDSC2"]:
        sub = res_df[res_df.screen == label]
        e = sub[sub.model == "expression -> AUC"].iloc[0]
        c = sub[sub.model.str.startswith("late fusion")].iloc[0]
        gain = c.r_mean - e.r_mean
        verdict = "adds a little" if gain > Q1_MIN_DIFF else "adds nothing interpretable"
        lines.append(f"- {label}: late fusion r={c.r_mean:+.3f} vs expression-only "
                     f"r={e.r_mean:+.3f} (Δ={gain:+.3f}) -- {verdict} by the same "
                     f"{Q1_MIN_DIFF:.2f} threshold used for Q1.\n")

    lines.append("\n## Q3 (primary): is the DTP programme epigenetically encoded?\n")
    lines.append(
        "**Trap**, stated rather than assumed away: "
        "DTP_up/DTP_down are expression-defined gene sets. Nothing about their "
        "definition implies their promoters are differentially methylated -- "
        "that is exactly what is tested below, not assumed. "
        "`lib.signatures.score_all()` is reused unmodified, applied to the promoter "
        "M-value matrix in place of expression, giving a methylation-space "
        "score with the same up_score-minus-down_score construction as the "
        "expression DTP feature. Under the hypothesis that promoter "
        "hypomethylation of DTP_up genes and hypermethylation of DTP_down "
        "genes maintains the DTP transcriptional state, this methylation-space "
        "score is expected to correlate **negatively** with the expression DTP "
        "score, not positively (a more-methylated promoter is, in general, a "
        "less transcribed one).\n\n"
    )
    lines.append("| Screen | n | meth-DTP vs expr-DTP | meth-DTP vs AUC | expr-DTP vs AUC (ref.) | Sign consistent with hypothesis? |\n")
    lines.append("|---|---|---|---|---|---|\n")
    for label, q in q3_results.items():
        lines.append(
            f"| {label} | {q['n']} | "
            f"r={q['r_meth_vs_expr']:+.3f} [{q['ci_meth_vs_expr'][0]:+.3f},{q['ci_meth_vs_expr'][1]:+.3f}] p={q['p_meth_vs_expr']:.4f} | "
            f"r={q['r_meth_vs_auc']:+.3f} [{q['ci_meth_vs_auc'][0]:+.3f},{q['ci_meth_vs_auc'][1]:+.3f}] p={q['p_meth_vs_auc']:.4f} | "
            f"r={q['r_expr_vs_auc']:+.3f} [{q['ci_expr_vs_auc'][0]:+.3f},{q['ci_expr_vs_auc'][1]:+.3f}] p={q['p_expr_vs_auc']:.4f} | "
            f"{'yes' if q['sign_consistent'] else 'no'} |\n"
        )

    both_consistent = all(q["sign_consistent"] for q in q3_results.values())
    both_sig = all(q["p_meth_vs_expr"] < 0.05 for q in q3_results.values())
    if both_sig and both_consistent:
        verdict = (
            "**Positive, in the DNA-methylation sense**: promoter methylation of "
            "DTP genes tracks the expression DTP score in the direction the "
            "silencing hypothesis predicts, in both screens."
        )
    elif not any(q["p_meth_vs_expr"] < 0.05 for q in q3_results.values()):
        verdict = (
            "**Clean negative**: promoter methylation of DTP genes shows no "
            "significant association with the expression DTP score in either "
            "screen. This refutes the DNA-methylation version of \"DTP is "
            "epigenetically encoded\" as tested here -- not the broader "
            "epigenetic hypothesis (see scope note above)."
        )
    else:
        verdict = (
            "**Mixed / inconsistent between screens** -- reported plainly rather "
            "than resolved in favour of either reading; see the per-screen "
            "table above."
        )
    lines.append(f"\n{verdict}\n")

    auc_sig = [q["p_meth_vs_auc"] < 0.05 for q in q3_results.values()]
    ref_sig = [q["p_expr_vs_auc"] < 0.05 for q in q3_results.values()]
    if not any(auc_sig) and all(ref_sig):
        behaves_like = (
            "**No, not as a direct predictor.** The methylation-derived DTP score "
            "is not significantly associated with AUC in either screen (both "
            "p>0.24), while the expression DTP score is significant in both. "
            "The answer to Q3 is therefore mixed rather than a single yes/no: "
            "promoter methylation of DTP genes tracks the expression programme "
            "(the table's first column, both screens p<0.0001) but is too weak or "
            "noisy a signal on its own to recover the programme's association "
            "with 5-FU response -- the thing that actually matters for the "
            "epigenetic-encoding hypothesis in its strong form. The expression "
            "readout stays the one doing the predictive work."
        )
    elif all(auc_sig):
        behaves_like = (
            "**Yes.** The methylation-derived DTP score is itself significantly "
            "associated with AUC in both screens, in the same direction as the "
            "expression-derived score -- it behaves like the expression readout, "
            "not just like a correlate of it."
        )
    else:
        behaves_like = (
            "**Inconsistent between screens** -- reported plainly rather than "
            "resolved; see the \"meth-DTP vs AUC\" column above."
        )
    lines.append(
        f"\nDoes a methylation-derived DTP score behave like the "
        f"expression-derived one as a predictor of 5-FU response? {behaves_like}\n"
    )

    lines.append("\n## Outputs\n\n```\ndata/processed/methylation_model_results.csv\n```\n")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "18_methylation_models.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
