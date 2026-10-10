"""
lib/timecourse.py
=================

The HCT116 5-FU time-course (Arm B): loading the counts and the trend test.
Used by stages/04_armB_induction.py and by the DTP sub-programme test in
stages/11_regulatory_architecture.py, which must read the time-course
exactly as stage 04 does.
"""

import sys

import numpy as np
import pandas as pd
from scipy import stats

import config as C


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


def hours(samples):
    """Hours since treatment for each time-course sample name."""
    return np.array([C.TIMECOURSE_SAMPLES[s][0] for s in samples], dtype=float)


def trend_stats(scores, feature):
    """
    Per-timepoint mean/SD, linear trend against time, and 48h-vs-0h
    standardised difference. With n=3 per timepoint these are descriptive
    effect sizes, not confident inference.
    """
    t = hours(scores.index)
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
