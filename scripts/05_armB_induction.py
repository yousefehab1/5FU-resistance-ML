"""Arm B: does 5-FU induce the DTP and regenerative programmes in HCT116 over time?

Scores the 0/6/24/48 h time course, tests each programme for a monotonic trend,
and compares against random gene sets of the same size, because proliferation
genes collapsing under treatment shift every rank.

Inputs:  data/raw/Sup_Table_2_HCT116_5FU_timecourse_treatment.txt, signatures/
Outputs: data/processed/timecourse_scores.csv, armB_induction_stats.csv,
         armB_compositional_control.csv, armB_cellcycle_control.csv
Run:     python scripts/05_armB_induction.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats


sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

PROJECT_ROOT = project_root()
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

TIMECOURSE = RAW / "Sup_Table_2_HCT116_5FU_timecourse_treatment.txt"
GENE_COL = "NAME"
MIN_DETECTED = 6          # keep genes with a non-zero count in >= 6 of 12 samples

SAMPLES = {
    "PN0129B_P_0h_Ctrl_black":  (0,  "Ctrl"), "PN0129B_P_0h_Ctrl_blue":  (0,  "Ctrl"),
    "PN0129B_P_0h_Ctrl_red":    (0,  "Ctrl"),
    "PN0129B_P_6h_5FU_black":   (6,  "5FU"),  "PN0129B_P_6h_5FU_blue":   (6,  "5FU"),
    "PN0129B_P_6h_5FU_red":     (6,  "5FU"),
    "PN0129B_P_24h_5FU_black":  (24, "5FU"),  "PN0129B_P_24h_5FU_blue":  (24, "5FU"),
    "PN0129B_P_24h_5FU_red":    (24, "5FU"),
    "PN0129B_P_48h_5FU_black":  (48, "5FU"),  "PN0129B_P_48h_5FU_blue":  (48, "5FU"),
    "PN0129B_P_48h_5FU_red":    (48, "5FU"),
}
TIMEPOINTS = [0, 6, 24, 48]


from fivefu.config import SEED
from fivefu.report import banner
from fivefu import signatures as S
from fivefu.stats import partial_corr


def load_counts():
    """Load, collapse duplicate symbols, filter to detected genes."""
    df = pd.read_csv(TIMECOURSE, sep="\t", low_memory=False)
    df = df.drop(columns=[c for c in df.columns if str(c).startswith("Unnamed")])

    missing = [s for s in SAMPLES if s not in df.columns]
    if missing:
        print(f"  !! Missing expected samples: {missing}")
        sys.exit(1)

    samples = list(SAMPLES)
    counts = df.set_index(GENE_COL)[samples].apply(pd.to_numeric, errors="coerce")
    print(f"  raw: {counts.shape[0]:,} rows x {counts.shape[1]} samples")

    # A symbol appearing on several rows is summed -- these are counts, so
    # summing is the meaningful operation (mean would understate the gene).
    n_dup = counts.index.duplicated().sum()
    if n_dup:
        counts = counts.groupby(level=0).sum()
        print(f"  collapsed {n_dup:,} duplicate symbols by sum -> {counts.shape[0]:,} genes")

    # Unexpressed genes matter: thousands of tied zeros distort within-sample
    # ranks, which is what the scoring depends on.
    detected = (counts > 0).sum(axis=1)
    keep = detected >= MIN_DETECTED
    print(f"  detected in >={MIN_DETECTED}/12 samples: {int(keep.sum()):,} "
          f"({int((~keep).sum()):,} dropped)")
    counts = counts[keep]

    # Samples in rows, genes in columns -- the orientation fivefu.signatures expects.
    return counts.T


def trend_stats(scores, feature):
    """
    Describe how one feature changes over the time-course.

    Reported: per-timepoint mean and SD, the linear trend against time, and the
    48h vs 0h standardised difference. With 3 replicates per timepoint these are
    descriptive -- an effect size with a direction, not a confident inference.
    """
    t = np.array([SAMPLES[s][0] for s in scores.index], dtype=float)
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
                **{f"mean_{tp}h": v[t == tp].mean() for tp in TIMEPOINTS},
                **{f"sd_{tp}h": v[t == tp].std(ddof=1) for tp in TIMEPOINTS})


def main():

    banner("1. LOADING TIME-COURSE")
    if not TIMECOURSE.exists():
        print(f"  !! Not found: {TIMECOURSE}")
        sys.exit(1)
    expr = load_counts()
    print(f"  matrix: {expr.shape[0]} samples x {expr.shape[1]:,} genes")

    banner("2. SIGNATURES")
    alias_map = S.load_alias_map(required=False)
    if alias_map:
        print(f"  HGNC alias map: {len(alias_map):,} entries")
    sigs = S.load_signatures(expr.columns, alias_map)

    banner("3. SCORING")
    print("  Same fivefu.signatures implementation as Arm A. Raw counts are fine: ranks")
    print("  depend only on gene order within a sample, and library size scales")
    print("  every gene by the same factor.")
    scores, ss = S.score_all(expr, sigs)
    out = scores.copy()
    out["timepoint_h"] = [SAMPLES[s][0] for s in out.index]
    out["treatment"] = [SAMPLES[s][1] for s in out.index]
    PROCESSED.mkdir(parents=True, exist_ok=True)
    out.to_csv(PROCESSED / "timecourse_scores.csv")

    banner("4. TRAJECTORIES  (mean +/- SD, n=3 per timepoint)")
    feats = list(scores.columns)
    print(f"  {'feature':<12}" + "".join(f"{str(tp) + 'h':>16}" for tp in TIMEPOINTS))
    for f in feats:
        row = f"  {f:<12}"
        for tp in TIMEPOINTS:
            m = scores.loc[[s for s in scores.index if SAMPLES[s][0] == tp], f]
            row += f"{m.mean():>9.2f}+-{m.std(ddof=1):<5.2f}"
        print(row)

    banner("5. INDUCTION TEST")
    res = pd.DataFrame([trend_stats(scores, f) for f in feats])
    res = res.sort_values("trend_r", ascending=False)
    print(f"  {'feature':<12} {'trend r':>9} {'p':>7} {'48h-0h':>9} "
          f"{'Cohen d':>9} {'t-test p':>9}")
    for _, r in res.iterrows():
        star = "*" if r.trend_p < 0.05 else " "
        print(f"  {r.feature:<12} {r.trend_r:>+9.3f}{star}{r.trend_p:>6.3f} "
              f"{r.delta_48_0:>+9.2f} {r.cohens_d:>+9.2f} {r.ttest_p:>9.3f}")

    res.to_csv(PROCESSED / "armB_induction_stats.csv", index=False)

    banner("5b. COMPOSITIONAL CONTROL  (essential)")
    # Rank-based scores are COMPOSITIONAL. Cell-cycle genes are numerous and
    # highly expressed; if they collapse under treatment, every other gene rises
    # in rank mechanically. That alone would manufacture "induction" for any
    # signature -- the same effect seen in the D13 null test, where a random
    # gene set correlated at rho ~ -0.43 with a planted gradient.
    #
    # So the observed trends mean nothing until compared against RANDOM gene
    # sets of the same size scored the same way. Without this control the whole
    # of section 5 would be uninterpretable.
    ranks = S.rank_matrix(expr)
    t = np.array([SAMPLES[s][0] for s in expr.index], dtype=float)
    rng = np.random.default_rng(SEED)
    N_NULL = 200

    print(f"  {'feature':<12} {'observed':>10} {'null mean':>10} {'null sd':>9} "
          f"{'z':>7} {'emp. p':>8}")
    ctrl_rows = []
    for f in feats:
        n = len(sigs.get(f, sigs.get("DTP_up", []))) if f == "DTP" else len(sigs[f])
        null = np.array([
            stats.pearsonr(t, S.background_score(
                ranks, list(rng.choice(ranks.columns, n, replace=False))))[0]
            for _ in range(N_NULL)])
        obs = stats.pearsonr(t, scores[f])[0]
        z = (obs - null.mean()) / null.std()
        p = (np.abs(null - null.mean()) >= abs(obs - null.mean())).mean()
        print(f"  {f:<12} {obs:>+10.3f} {null.mean():>+10.3f} {null.std():>9.3f} "
              f"{z:>+7.2f} {p:>8.3f}")
        ctrl_rows.append(dict(feature=f, observed=obs, null_mean=null.mean(),
                              null_sd=null.std(), z=z, emp_p=p))
    pd.DataFrame(ctrl_rows).to_csv(PROCESSED / "armB_compositional_control.csv",
                                   index=False)
    print()
    print("  Note the null SD is large (~0.4): with only 4 timepoints, random")
    print("  gene sets reach |r| ~ 0.7 easily. So a trend of +0.96 is genuinely")
    print("  extreme, but the effective sample size is 4, not 12.")

    banner("5c. IS IT JUST THE PROLIFERATION COLLAPSE?")
    # If the regenerative modules only rise because cell cycle falls, their
    # trend should vanish once CellCycle is partialled out.
    cc = scores["CellCycle"].to_numpy()
    cc_rows = []
    for f in [x for x in feats if x != "CellCycle"]:
        r, p = partial_corr(t, scores[f].to_numpy(), [cc])
        print(f"  {f:<12} trend vs time, controlling CellCycle:  r={r:+.3f}  p={p:.3f}")
        cc_rows.append(dict(feature=f, partial_r=r, p=p))
    pd.DataFrame(cc_rows).to_csv(PROCESSED / "armB_cellcycle_control.csv", index=False)

    banner("6. CROSS-ARM COMPARISON")
    # Arm A found which modules track RESISTANCE across cell lines. Arm B finds
    # which are INDUCED by treatment. The interesting case is a module that does
    # both: it marks intrinsically resistant cells AND is switched on by the
    # drug -- an adaptive-tolerance pattern rather than pre-existing resistance.
    try:
        s1 = pd.read_parquet(PROCESSED / "signature_scores_GDSC1.parquet")
        y1 = pd.read_parquet(PROCESSED / "y_GDSC1.parquet")
        crc = (y1.TCGA_DESC == "COREAD").to_numpy()
        print(f"  {'feature':<12} {'ArmA r (resistance)':>21} {'ArmB r (induction)':>20}")
        for f in feats:
            if f in s1.columns:
                ra, pa = stats.pearsonr(s1.loc[crc, f], y1.loc[crc, "AUC"])
                rb = res.loc[res.feature == f, "trend_r"].iloc[0]
                flag = "  <-- both positive" if (ra > 0.2 and rb > 0.2) else ""
                print(f"  {f:<12} {ra:>+21.3f} {rb:>+20.3f}{flag}")
    except FileNotFoundError:
        print("  Run 03_baselines.py first for the cross-arm comparison.")

    banner("7. HOW TO READ THIS")
    print("  * n=3 per timepoint. These are effect sizes with a direction, not")
    print("    confident inference. A large Cohen's d on n=3 is still n=3.")
    print("  * No time-matched vehicle control: any change confounds 5-FU with")
    print("    48 more hours in culture. The claim is 'changed under treatment")
    print("    over time', not 'induced by 5-FU specifically'.")
    print("  * Non-circular: the DTP signature came from a different experiment,")
    print("    so this is a real test rather than true by construction.")


if __name__ == "__main__":
    main()
