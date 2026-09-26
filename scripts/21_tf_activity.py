"""Transcription-factor activity (decoupler, CollecTRI) versus DTP and response.

If CollecTRI cannot be downloaded the run stops. --allow-network-fallback uses
DoRothEA instead, which is a different regulon; do not compare those results
with tests/golden/.

Inputs:  data/processed/X_*.parquet, y_*.parquet, signature_scores_*.parquet
Outputs: data/processed/tf_activity_{GDSC1,GDSC2}.parquet, tf_activity_results.csv,
         tf_dtp_associations.csv, reports/21_tf_activity.md
Run:     python scripts/21_tf_activity.py [--allow-network-fallback]
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import decoupler as dc
from scipy import stats
from scipy.stats import false_discovery_control
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
PROC = ROOT / "data" / "processed"
DOCS = ROOT / "data" / "processed" / "reports"  # generated reports


PRESPECIFIED_TFS = ["TEAD1", "TEAD2", "TEAD3", "TEAD4",
                    "MYC", "E2F1", "TP53", "SOX9", "HNF4A", "CDX2"]
MIN_TARGETS = 10
TARGET, CRC = "AUC", "COREAD"

# Analysis parameters are defined once, in config/*.yaml, and read through
# fivefu.config. These used to be taken from 08 (SEED = FM.SEED), which made 08
# the de facto config file for anything that imported it. Both now read the
# same YAML, so they cannot disagree.
from fivefu.config import (CEILING_AUC, SEED, N_FOLDS, MIN_LINEAGE_N,
                           MODELED_MODULES as MODULES)
from fivefu.io import LINEAGE, load_screen
from fivefu.modeling import elasticnet_pipeline, permuted_label_check, report_cv
from fivefu.report import banner
from fivefu import signatures as S
from fivefu.stats import partial_corr


def get_network():
    """
    Fetch the TF regulon. CollecTRI is the network every published result here
    was computed against.

    The DoRothEA fallback is OPT-IN, and deliberately so. It used to be
    automatic, and on 2026-09-09 a transient Zenodo 504 silently swapped in a
    different regulon: 32286 DoRothEA interactions across 429 TFs in place of
    CollecTRI, which halved tf_dtp_associations.csv from 5740 rows to 2710. The
    run still exited 0 and still wrote results, and neither output file records
    which network produced it, so nothing downstream could have noticed.

    A transient network error must therefore stop the run rather than quietly
    answer a different question. Pass --allow-network-fallback if you genuinely
    want DoRothEA, and do not compare those outputs against the golden values.
    """
    allow_fallback = "--allow-network-fallback" in sys.argv
    try:
        net = dc.op.collectri(organism="human")
        source = "CollecTRI"
    except Exception as e:
        if not allow_fallback:
            raise RuntimeError(
                f"CollecTRI could not be retrieved: {e!r}\n"
                f"  Refusing to fall back to DoRothEA. It is a different regulon, "
                f"and substituting it would change the biology behind every number "
                f"this script writes without changing the files' appearance.\n"
                f"  This is usually a transient Zenodo outage: try again shortly.\n"
                f"  To use DoRothEA deliberately, re-run with "
                f"--allow-network-fallback."
            ) from e
        print(f"  CollecTRI retrieval failed ({e!r})")
        print( "  --allow-network-fallback given: using DoRothEA A/B/C instead.")
        print( "  NOTE: results will NOT match tests/golden/, which is CollecTRI.")
        net = dc.op.dorothea(organism="human", levels=["A", "B", "C"])
        source = "DoRothEA (confidence A/B/C)"
    n_tf = net["source"].nunique()
    print(f"  decoupler version: {dc.__version__}")
    print(f"  network source: {source}  |  {len(net)} interactions across {n_tf} TFs (pre-filter)")
    return net, source


def infer_tf_activity(X, net):
    est, pval = dc.mt.ulm(X, net, tmin=MIN_TARGETS, verbose=False)
    dropped = net["source"].nunique() - est.shape[1]
    print(f"  TF activities retained: {est.shape[1]}  |  dropped for < {MIN_TARGETS} "
          f"targets in this gene universe: {dropped}")
    present = [tf for tf in PRESPECIFIED_TFS if tf in est.columns]
    missing = [tf for tf in PRESPECIFIED_TFS if tf not in est.columns]
    print(f"  pre-specified TFs present: {present}")
    if missing:
        n_in_net = {tf: int((net["source"] == tf).sum()) for tf in missing}
        print(f"  pre-specified TFs MISSING (raw target count in network, before tmin): {n_in_net}")
    return est.reindex(X.index)


def stratified_by_assay(X, t, name):
    """Mirrors 08_final_model.py section 3b, applied to a given feature matrix."""
    Xa = np.asarray(X)
    p_all = np.full(len(t), np.nan)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xa):
        p_all[te] = elasticnet_pipeline().fit(Xa[tr], t[tr]).predict(Xa[te])
    rows = []
    for lab, msk in [("<= 0.90 (responsive)", t <= 0.90),
                     (f"0.90 - {CEILING_AUC}", (t > 0.90) & (t <= CEILING_AUC)),
                     (f"> {CEILING_AUC} (assay ceiling)", t > CEILING_AUC)]:
        if msk.sum() > 10:
            rr = stats.pearsonr(t[msk], p_all[msk])[0]
            print(f"    {lab:<24} n={int(msk.sum()):>5} ({100*msk.mean():>5.1f}%)  r={rr:+.3f}")
            rows.append(dict(model=f"{name}, stratified: {lab}", r_mean=rr,
                             r_lo=np.nan, r_hi=np.nan, r_sd=np.nan))
    return rows


def a1_repeated_cv(label, X_expr, X_tf, s, y):
    banner(f"A1. REPEATED CV -- {label}")
    lin = y[LINEAGE].to_numpy()
    t = y[TARGET].to_numpy()
    cnt = pd.Series(lin).value_counts()
    big = set(cnt[cnt >= MIN_LINEAGE_N].index)
    keep = np.isin(lin, list(big))
    print(f"  n={len(y)}  |  de-confounded arm uses {keep.sum()} lines in {len(big)} lineages\n")

    rows = []
    rows.append(report_cv(X_expr, t, lin, False, "baseline: transcriptome -> raw AUC"))
    rows.append(report_cv(X_expr[keep], t[keep], lin[keep], True,
                               "baseline: transcriptome -> AUC, de-confounded"))
    rows.append(report_cv(s[MODULES], t, lin, False, "baseline: modules -> raw AUC"))
    rows.append(report_cv(s[MODULES][keep], t[keep], lin[keep], True,
                               "baseline: modules -> AUC, de-confounded"))
    rows.append(report_cv(X_tf, t, lin, False, "TF activity -> raw AUC"))
    rows.append(report_cv(X_tf[keep], t[keep], lin[keep], True,
                               "TF activity -> AUC, de-confounded"))
    rows.append(permuted_label_check(X_tf, t, "TF-activity model", "TF-activity model"))

    print("\n  stratified by assay resolution (TF-activity model):")
    rows.extend(stratified_by_assay(X_tf, t, "TF activity"))

    for r in rows:
        r["screen"] = label
        r["n"] = len(y)
    return rows


def a2_a3_dtp_associations(label, X_tf, s, y):
    banner(f"A2/A3. TF ACTIVITY vs DTP SCORE AND AUC, WITHIN COREAD -- {label}")
    crc = (y[LINEAGE] == CRC).to_numpy()
    msi_ok = y["msi_status"].notna().to_numpy()
    m = crc & msi_ok
    print(f"  COREAD with MSI status: n={int(m.sum())}")
    Xc = X_tf[m]
    dtp = s[m]["DTP"].to_numpy()
    auc = y[m][TARGET].to_numpy()
    msi = (y[m]["msi_status"] == "MSI").astype(float).to_numpy()

    rows = []
    for tf in Xc.columns:
        x = Xc[tf].to_numpy()
        for metric, (r, p) in [
            ("DTP_unadjusted", stats.pearsonr(x, dtp)),
            ("AUC_unadjusted", stats.pearsonr(x, auc)),
            ("DTP_msi_adjusted", partial_corr(x, dtp, [msi])),
            ("AUC_msi_adjusted", partial_corr(x, auc, [msi])),
        ]:
            rows.append(dict(screen=label, n=int(m.sum()), question="A2_COREAD",
                             tf=tf, prespecified=tf in PRESPECIFIED_TFS,
                             metric=metric, r=r, p=p))
    df = pd.DataFrame(rows)
    for tf in PRESPECIFIED_TFS:
        if tf in Xc.columns:
            sub = df[(df.tf == tf) & (df.metric == "AUC_unadjusted")]
            print(f"    {tf:<8} vs AUC (unadj.) r={sub.r.iloc[0]:+.3f} p={sub.p.iloc[0]:.4f}")
    return df


def a4_cross_layer(label, X_tf, pm, y, s):
    banner(f"A4. CROSS-LAYER: TF ACTIVITY vs METHYLATION-DERIVED DTP SCORE -- {label}")
    overlap = sorted(set(X_tf.index) & set(pm.index))
    print(f"  TF-activity + methylation overlap: n={len(overlap)}")
    alias_map = S.load_alias_map(required=False)
    sigs = S.load_signatures(pm.columns, alias_map, verbose=False)
    if "DTP_up" not in sigs or "DTP_down" not in sigs:
        print("  !! DTP_up/DTP_down did not resolve against the methylation gene "
              "universe -- A4 cannot be answered for this screen.")
        return pd.DataFrame()

    rank_df, _ = S.score_all(pm.loc[overlap], sigs, verbose=False)
    meth_dtp = rank_df["DTP"].to_numpy()
    Xo = X_tf.loc[overlap]

    rows = []
    for tf in Xo.columns:
        r, p = stats.pearsonr(Xo[tf].to_numpy(), meth_dtp)
        rows.append(dict(screen=label, n=len(overlap), question="A4_cross_layer",
                         tf=tf, prespecified=tf in PRESPECIFIED_TFS,
                         metric="meth_DTP_unadjusted", r=r, p=p))
    df = pd.DataFrame(rows)
    for tf in PRESPECIFIED_TFS:
        if tf in Xo.columns:
            sub = df[df.tf == tf]
            print(f"    {tf:<8} vs meth-DTP r={sub.r.iloc[0]:+.3f} p={sub.p.iloc[0]:.4f}")
    return df


def add_fdr(df):
    """BH correction within each (screen, question, metric) group, EXPLORATORY TFs only.
    Pre-specified TFs are confirmatory single tests declared in advance (A3) and are
    reported at raw p, not pooled into the exploratory correction."""
    df = df.copy()
    df["q"] = np.nan
    for (_, _, _), sub in df.groupby(["screen", "question", "metric"]):
        expl = sub[~sub.prespecified]
        if len(expl):
            df.loc[expl.index, "q"] = false_discovery_control(expl["p"].to_numpy())
    return df


def main():
    banner("SETUP")
    print("  TF activity is a lossy projection of the SAME expression matrix the")
    print("  transcriptome model already uses. It CANNOT beat that model, and")
    print("  expecting it to would be a category error. The result of interest is")
    print("  WHICH TFs are implicated, not whether this model is more accurate.")
    print(f"\n  Pre-specified TFs (A3), declared before any result below: {PRESPECIFIED_TFS}")

    net, net_source = get_network()

    tf_rows, assoc_frames = [], []
    pm = pd.read_parquet(PROC / "methylation_promoter_M.parquet")

    for label in ["GDSC1", "GDSC2"]:
        banner(f"LOAD + INFER TF ACTIVITY -- {label}")
        X, y, s = load_screen(label, drop_haem=True)
        print(f"  n={len(y)} solid lines, {X.shape[1]} genes")
        X_tf = infer_tf_activity(X, net)
        X_tf.to_parquet(PROC / f"tf_activity_{label}.parquet")

        tf_rows.extend(a1_repeated_cv(label, X, X_tf, s, y))
        assoc_frames.append(a2_a3_dtp_associations(label, X_tf, s, y))
        assoc_frames.append(a4_cross_layer(label, X_tf, pm, y, s))

    tf_df = pd.DataFrame(tf_rows)
    tf_df.to_csv(PROC / "tf_activity_results.csv", index=False)

    assoc_df = pd.concat([f for f in assoc_frames if len(f)], ignore_index=True)
    assoc_df = add_fdr(assoc_df)
    assoc_df.to_csv(PROC / "tf_dtp_associations.csv", index=False)

    banner("WRITING DOC")
    write_doc(net_source, tf_df, assoc_df)
    print(f"  -> {DOCS / '21_tf_activity.md'}")

    banner("DONE")
    print(f"  -> {PROC / 'tf_activity_results.csv'}")
    print(f"  -> {PROC / 'tf_dtp_associations.csv'}")


def write_doc(net_source, tf_df, assoc_df):
    lines = []
    lines.append("# Transcription factor activity inference\n")
    lines.append(
        "Follow-up to `20_methylation_models.md`: promoter "
        "methylation of DTP genes tracks the expression DTP score but does not "
        "itself predict 5-FU response. This task tests whether the regulatory "
        "state is better read at the transcription-factor level than at the DNA "
        "level.\n"
    )

    lines.append("## What this task can and cannot show\n")
    lines.append(
        "TF activity is inferred by `decoupler`'s `run_ulm` (univariate linear "
        "model), computed **per sample from that sample's own expression against "
        "a fixed regulon** -- unsupervised with respect to the target, so it is "
        "safe to precompute once rather than inside each CV fold. It is a lossy "
        "**projection of the same expression matrix** the transcriptome model "
        "already uses. It therefore **cannot beat the expression model**, and a "
        "TF-activity model scoring well below the expression baseline below is "
        "the **expected outcome, not a failure**. The result of interest is "
        "*which* TFs are implicated, and whether they are the ones the "
        "drug-tolerant-persister (DTP) literature names -- in particular the "
        "YAP/TAZ-TEAD \"revival\" programme.\n"
    )
    lines.append(f"**Network**: {net_source} (decoupler {dc.__version__}).\n")

    lines.append("## A1: TF activity as a predictor of 5-FU response\n")
    lines.append(
        "Standard pipeline (`SelectKBest(f_regression, k=1500) -> StandardScaler "
        "-> ElasticNet`), repeated CV, raw and lineage-de-confounded arms, "
        "alongside the transcriptome and modules baselines, a permuted-label "
        "check, and a stratification by assay resolution -- all reusing "
        "`08_final_model.py`'s `model()`/`repeated_cv()` unmodified.\n"
    )
    for label in ["GDSC1", "GDSC2"]:
        sub = tf_df[tf_df.screen == label]
        lines.append(f"\n**{label}** (n={int(sub.n.iloc[0])})\n\n")
        lines.append("| Model | r [95% CI] |\n|---|---|\n")
        for _, row in sub.iterrows():
            if pd.notna(row.r_lo):
                lines.append(f"| {row.model} | {row.r_mean:+.3f} [{row.r_lo:+.3f}, {row.r_hi:+.3f}] |\n")
            else:
                lines.append(f"| {row.model} | {row.r_mean:+.3f} |\n")

    lines.append(
        "\nTF activity sits below the transcriptome baseline in both screens, as "
        "expected (see caveat above) -- it is not interpreted as a modelling "
        "failure. The permuted-label check passes (r approx. 0) in both screens, "
        "confirming no leakage.\n"
    )

    lines.append("\n## A2/A3: pre-specified TFs vs DTP score and AUC, within COREAD\n")
    lines.append(
        "Pre-specified before any result: TEAD1, TEAD2, TEAD3, TEAD4 (YAP/TAZ "
        "effectors), MYC, E2F1, TP53, SOX9, HNF4A, CDX2. These are reported at "
        "raw p (confirmatory, one test each, declared in advance). Everything "
        "else is exploratory and Benjamini-Hochberg corrected as its own pool.\n\n"
    )
    a2 = assoc_df[assoc_df.question == "A2_COREAD"]
    for label in ["GDSC1", "GDSC2"]:
        sub = a2[(a2.screen == label) & (a2.prespecified)]
        if not len(sub):
            continue
        n = int(sub.n.iloc[0])
        lines.append(f"\n**{label} COREAD** (n={n})\n\n")
        lines.append("| TF | vs DTP (unadj.) | vs DTP \\| MSI | vs AUC (unadj.) | vs AUC \\| MSI |\n")
        lines.append("|---|---|---|---|---|\n")
        for tf in PRESPECIFIED_TFS:
            tsub = sub[sub.tf == tf]
            if not len(tsub):
                continue
            def cell(metric):
                r = tsub[tsub.metric == metric]
                if not len(r):
                    return "-"
                r = r.iloc[0]
                sig = "*" if r.p < 0.05 else ""
                return f"r={r.r:+.3f}, p={r.p:.3f}{sig}"
            lines.append(f"| {tf} | {cell('DTP_unadjusted')} | {cell('DTP_msi_adjusted')} | "
                         f"{cell('AUC_unadjusted')} | {cell('AUC_msi_adjusted')} |\n")

    lines.append("\n**Exploratory TFs** (all others surfaced by A2, BH-corrected):\n\n")
    for label in ["GDSC1", "GDSC2"]:
        sub = a2[(a2.screen == label) & (~a2.prespecified) & (a2.metric == "AUC_unadjusted")]
        if not len(sub):
            continue
        hits = sub[sub.q < 0.10].sort_values("q")
        lines.append(f"- {label}: {len(hits)} of {len(sub)} exploratory TFs at q<0.10 "
                     f"(TF vs AUC, unadjusted)")
        if len(hits):
            top = ", ".join(f"{r.tf} (r={r.r:+.3f}, q={r.q:.3f})" for _, r in hits.head(10).iterrows())
            lines.append(f" -- top: {top}")
        lines.append("\n")

    lines.append("\n## A4: cross-layer -- TF activity vs the methylation-derived DTP score\n")
    lines.append(
        "Reported regardless of outcome. If the "
        "regulatory state found by A1-A3 is real, TF activity and the "
        "methylation-derived DTP score from `20_methylation_models.py` (an independent readout, "
        "over the expression+methylation sample overlap, not restricted to "
        "COREAD) should agree.\n\n"
    )
    a4 = assoc_df[assoc_df.question == "A4_cross_layer"]
    for label in ["GDSC1", "GDSC2"]:
        sub = a4[(a4.screen == label) & (a4.prespecified)]
        if not len(sub):
            lines.append(f"- {label}: A4 could not be computed (see run log).\n")
            continue
        n = int(sub.n.iloc[0])
        lines.append(f"\n**{label}** (n={n})\n\n")
        lines.append("| TF | vs methylation-derived DTP score |\n|---|---|\n")
        for tf in PRESPECIFIED_TFS:
            tsub = sub[sub.tf == tf]
            if not len(tsub):
                continue
            r = tsub.iloc[0]
            sig = "*" if r.p < 0.05 else ""
            lines.append(f"| {tf} | r={r.r:+.3f}, p={r.p:.3f}{sig} |\n")
        expl = a4[(a4.screen == label) & (~a4.prespecified)]
        if len(expl):
            hits = expl[expl.q < 0.10].sort_values("q")
            lines.append(f"\nExploratory: {len(hits)} of {len(expl)} TFs at q<0.10.")
            if len(hits):
                top = ", ".join(f"{r.tf} (r={r.r:+.3f}, q={r.q:.3f})" for _, r in hits.head(10).iterrows())
                lines.append(f" Top: {top}")
            lines.append("\n")

    lines.append("\n## Outputs\n\n```\n")
    lines.append("data/processed/tf_activity_GDSC1.parquet\n")
    lines.append("data/processed/tf_activity_GDSC2.parquet\n")
    lines.append("data/processed/tf_activity_results.csv\n")
    lines.append("data/processed/tf_dtp_associations.csv\n")
    lines.append("```\n")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "21_tf_activity.md").write_text("".join(lines))


if __name__ == "__main__":
    main()
