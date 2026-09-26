"""
stages/08_armB_induction.py
==============================

Arm B. Does 5-FU treatment induce the DTP and regenerative programmes over
time in HCT116?

Arm A (stages 01-07) asks whether BASELINE expression predicts 5-FU
sensitivity ACROSS cell lines. Arm B asks whether TREATMENT changes these
programmes WITHIN one line. Different data, different failure modes -- two
weak but independent lines of evidence pointing the same way is an
argument; one is not.

WHAT THIS CAN AND CANNOT CLAIM
--------------------------------
Untreated samples exist ONLY at 0h; there is no time-matched vehicle
control at 6, 24 or 48 hours. So a change by 48h confounds the effect of
5-FU with the effect of 48 more hours in culture (confluence, media
exhaustion, contact inhibition). No analysis fixes this -- only a vehicle
arm would. The honest claim is "changed under 5-FU treatment over time",
NOT "induced by 5-FU specifically". With n=3 per timepoint this supports a
DIRECTION and an EFFECT SIZE; p-values are descriptive, not confident
inference.

NON-CIRCULARITY: the DTP signature was derived from a SEPARATE HCT116 5-FU
experiment, not from this time-course, so induction testing here is a
genuine test, not true by construction.

Uses the identical lib.signatures scoring as Arm A, so the two arms cannot
drift apart against different HGNC vintages -- see lib/signatures.py.

INPUTS   data/raw/Sup_Table_2_HCT116_5FU_timecourse_treatment.txt
         data/raw/signatures/*.txt, data/raw/hgnc_alias_map.csv
         data/processed/signature_scores_GDSC1.parquet, y_GDSC1.parquet (cross-arm compare)
OUTPUTS  data/processed/timecourse_scores.csv
         data/processed/armB_induction_stats.csv
         data/processed/armB_compositional_control.csv
         data/processed/armB_cellcycle_adjusted.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from lib.report import banner, write_and_report
from lib.signatures import load_alias_map, load_signatures, rank_matrix, score_all
from lib.stats import compositional_null, partial_corr


def load_counts():
    """Load, collapse duplicate symbols by SUM (these are counts), filter to detected genes."""
    df = pd.read_csv(C.TIMECOURSE_FILE, sep="\t", low_memory=False)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])

    samples = list(C.TIMECOURSE_SAMPLES)
    missing = [s for s in samples if s not in df.columns]
    if missing:
        print(f"  !! Missing expected samples: {missing}")
        sys.exit(1)

    counts = df.set_index(C.TIMECOURSE_GENE_COL)[samples].apply(pd.to_numeric, errors="coerce")
    print(f"  raw: {counts.shape[0]:,} rows x {counts.shape[1]} samples")

    n_dup = counts.index.duplicated().sum()
    if n_dup:
        counts = counts.groupby(level=0).sum()
        print(f"  collapsed {n_dup:,} duplicate symbols by sum -> {counts.shape[0]:,} genes")

    # Unexpressed genes matter: thousands of tied zeros distort within-sample
    # ranks, which is what the scoring depends on.
    detected = (counts > 0).sum(axis=1)
    keep = detected >= C.TIMECOURSE_MIN_DETECTED
    print(f"  detected in >={C.TIMECOURSE_MIN_DETECTED}/12 samples: {int(keep.sum()):,} "
          f"({int((~keep).sum()):,} dropped)")
    counts = counts[keep]

    return counts.T   # samples in rows, genes in columns


def trend_stats(scores, feature):
    """
    Per-timepoint mean/SD, linear trend against time, and 48h-vs-0h
    standardised difference. With n=3 per timepoint these are descriptive
    effect sizes, not confident inference.
    """
    t = np.array([C.TIMECOURSE_SAMPLES[s][0] for s in scores.index], dtype=float)
    v = scores[feature].to_numpy()

    r, p = stats.pearsonr(t, v)
    rho, p_rho = stats.spearmanr(t, v)

    base = v[t == 0]
    last = v[t == 48]
    pooled_sd = np.sqrt((base.var(ddof=1) + last.var(ddof=1)) / 2)
    d = (last.mean() - base.mean()) / pooled_sd if pooled_sd > 0 else np.nan
    tt, p_t = stats.ttest_ind(last, base)

    return dict(feature=feature, trend_r=r, trend_p=p, spearman=rho,
                delta_48_0=last.mean() - base.mean(), cohens_d=d, ttest_p=p_t,
                **{f"mean_{tp}h": v[t == tp].mean() for tp in C.TIMECOURSE_TIMEPOINTS},
                **{f"sd_{tp}h": v[t == tp].std(ddof=1) for tp in C.TIMECOURSE_TIMEPOINTS})


def main():
    banner("1. LOADING TIME-COURSE")
    expr = load_counts()
    print(f"  matrix: {expr.shape[0]} samples x {expr.shape[1]:,} genes")

    banner("2. SIGNATURES")
    alias_map = load_alias_map(required=False)
    sigs = load_signatures(expr.columns, alias_map)

    banner("3. SCORING")
    print("  Same lib.signatures implementation as Arm A. Raw counts are fine:")
    print("  ranks depend only on gene order within a sample, and library size")
    print("  scales every gene by the same factor.")
    scores, _ = score_all(expr, sigs)
    out = scores.copy()
    out["timepoint_h"] = [C.TIMECOURSE_SAMPLES[s][0] for s in out.index]
    out["treatment"] = [C.TIMECOURSE_SAMPLES[s][1] for s in out.index]
    write_and_report(out.reset_index(names="sample"),
                      C.PROCESSED / "timecourse_scores.csv", "timecourse_scores.csv")

    banner("4. TRAJECTORIES  (mean +/- SD, n=3 per timepoint)")
    feats = list(scores.columns)
    print(f"  {'feature':<12}" + "".join(f"{str(tp) + 'h':>16}" for tp in C.TIMECOURSE_TIMEPOINTS))
    for f in feats:
        row = f"  {f:<12}"
        for tp in C.TIMECOURSE_TIMEPOINTS:
            m = scores.loc[[s for s in scores.index if C.TIMECOURSE_SAMPLES[s][0] == tp], f]
            row += f"{m.mean():>9.2f}+-{m.std(ddof=1):<5.2f}"
        print(row)

    banner("5. INDUCTION TEST")
    res = pd.DataFrame([trend_stats(scores, f) for f in feats]).sort_values(
        "trend_r", ascending=False)
    print(f"  {'feature':<12} {'trend r':>9} {'p':>7} {'48h-0h':>9} "
          f"{'Cohen d':>9} {'t-test p':>9}")
    for _, r in res.iterrows():
        star = "*" if r.trend_p < 0.05 else " "
        print(f"  {r.feature:<12} {r.trend_r:>+9.3f}{star}{r.trend_p:>6.3f} "
              f"{r.delta_48_0:>+9.2f} {r.cohens_d:>+9.2f} {r.ttest_p:>9.3f}")
    write_and_report(res, C.PROCESSED / "armB_induction_stats.csv", "armB_induction_stats.csv")

    banner("5b. COMPOSITIONAL CONTROL  (essential)")
    # Rank-based scores are COMPOSITIONAL. Cell-cycle genes are numerous and
    # highly expressed; if they collapse under treatment, every other gene
    # rises in rank mechanically -- that alone would manufacture "induction"
    # for any signature. So observed trends mean nothing until compared
    # against RANDOM gene sets of the same size, scored the same way.
    t = np.array([C.TIMECOURSE_SAMPLES[s][0] for s in expr.index], dtype=float)
    ranks = rank_matrix(expr)
    rng = np.random.default_rng(C.SEED)   # one continuous stream across all features

    print(f"  {'feature':<12} {'observed':>10} {'null mean':>10} {'null sd':>9} "
          f"{'z':>7} {'emp. p':>8}")
    ctrl_rows = []
    for f in feats:
        n = len(sigs["DTP_up"]) if f == "DTP" else len(sigs[f])
        row = compositional_null(scores[f], t, n, ranks, rng, feature=f)
        print(f"  {f:<12} {row['observed_r']:>+10.3f} {row['null_mean']:>+10.3f} "
              f"{row['null_sd']:>9.3f} {row['z']:>+7.2f} {row['p_empirical']:>8.3f}")
        ctrl_rows.append(row)
    write_and_report(pd.DataFrame(ctrl_rows), C.PROCESSED / "armB_compositional_control.csv",
                      "armB_compositional_control.csv")
    print()
    print("  Note the null SD is large (~0.4): with only 4 timepoints, random gene")
    print("  sets reach |r| ~ 0.7 easily -- the effective sample size is 4, not 12.")

    banner("5c. IS IT JUST THE PROLIFERATION COLLAPSE?")
    # If the regenerative modules only rise because cell cycle falls, their
    # trend should vanish once CellCycle is partialled out.
    cc_rows = []
    cc = scores["CellCycle"].to_numpy()
    for f in [x for x in feats if x != "CellCycle"]:
        r, p = partial_corr(t, scores[f].to_numpy(), [cc])
        print(f"  {f:<12} trend vs time, controlling CellCycle:  r={r:+.3f}  p={p:.3f}")
        cc_rows.append(dict(feature=f, r_adjusted=r, p_adjusted=p))
    write_and_report(pd.DataFrame(cc_rows), C.PROCESSED / "armB_cellcycle_adjusted.csv",
                      "armB_cellcycle_adjusted.csv")

    banner("6. CROSS-ARM COMPARISON")
    # Arm A found which modules track RESISTANCE across cell lines. Arm B
    # finds which are INDUCED by treatment. The interesting case is a module
    # doing both: it marks intrinsically resistant cells AND is switched on
    # by the drug -- adaptive tolerance, not just pre-existing resistance.
    s1_path = C.PROCESSED / "signature_scores_GDSC1.parquet"
    y1_path = C.PROCESSED / "y_GDSC1.parquet"
    if s1_path.exists() and y1_path.exists():
        s1, y1 = pd.read_parquet(s1_path), pd.read_parquet(y1_path)
        crc = (y1[C.LINEAGE_COL] == C.CRC).to_numpy()
        print(f"  {'feature':<12} {'ArmA r (resistance)':>21} {'ArmB r (induction)':>20}")
        for f in feats:
            if f in s1.columns:
                ra, _ = stats.pearsonr(s1.loc[crc, f], y1.loc[crc, C.TARGET])
                rb = res.loc[res.feature == f, "trend_r"].iloc[0]
                flag = "  <-- both positive" if (ra > 0.2 and rb > 0.2) else ""
                print(f"  {f:<12} {ra:>+21.3f} {rb:>+20.3f}{flag}")
    else:
        print("  Run stages 01-03 first for the cross-arm comparison.")

    banner("7. HOW TO READ THIS")
    print("  * n=3 per timepoint. These are effect sizes with a direction, not")
    print("    confident inference. A large Cohen's d on n=3 is still n=3.")
    print("  * No time-matched vehicle control: any change confounds 5-FU with")
    print("    48 more hours in culture. The claim is 'changed under treatment")
    print("    over time', not 'induced by 5-FU specifically'.")
    print("  * Non-circular: the DTP signature came from a different experiment,")
    print("    so this is a real test rather than true by construction.")
    banner("DONE")


if __name__ == "__main__":
    main()
