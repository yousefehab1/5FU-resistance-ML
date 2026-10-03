"""DTP methylation versus response, split by CpG-island context.

First reports how much of each gene set each context covers, then reruns the
methylation-DTP test within each stratum with BH correction per screen.

Inputs:  data/processed/methylation_by_context/*_M.csv
         (from stages/prep/methylation_context_prep.R),
         data/raw/methylation/humanmethylation450_manifest.csv
Outputs: data/processed/methylation_context_coverage.csv,
         methylation_context_results.csv, reports/methylation_context.md
Run:     Rscript stages/prep/methylation_context_prep.R && python stages/09_methylation_context.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy.stats import false_discovery_control

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from lib.io import solid_screen
from lib.report import banner, write_and_report, write_report
from lib import signatures as S
from lib.stats import bootstrap_r

STRATA = ["Island", "N_Shore", "S_Shore", "N_Shelf", "S_Shelf", "OpenSea"]
SIGNATURE_SETS = ["DTP_up", "DTP_down", "RSC", "CBC", "Fetal"]
ENHANCER_COVERAGE_MIN = 0.05     # stop, don't model, below this


def explode_manifest(man):
    """Probe -> (gene, region) long table, genes and regions kept positionally
    aligned (same convention as methylation_context_prep.R's split)."""
    m = man.copy()
    m["gene_list"] = m["gene"].fillna("").str.split(";")
    m["region_list"] = m["region"].fillna("").str.split(";")
    long = m.explode(["gene_list", "region_list"])
    return long[long.gene_list != ""]


def c1_coverage_audit():
    banner("C1: HONEST COVERAGE AUDIT (before any modelling)")
    print("  Array-design coverage: all 485,512 probes on the HumanMethylation450")
    print("  manifest, BEFORE the cross-reactive/SNP/sex-chromosome QC filter used")
    print("  for modelling (that filter removes ~29% of probes but is not")
    print("  gene-set-selective, so it does not change which genes are covered).")

    man = pd.read_csv(C.METHYLATION_MANIFEST, dtype=str, low_memory=False)
    universe = set()
    for g in man["gene"].dropna():
        universe.update(g.split(";"))
    universe.discard("")
    alias_map = S.load_alias_map(required=False)
    sigs = S.load_signatures(universe, alias_map, verbose=True)

    long = explode_manifest(man)
    rows = []
    for name in SIGNATURE_SETS:
        if name not in sigs:
            print(f"  !! {name} did not resolve against the manifest gene universe -- skipped")
            continue
        genes = set(sigs[name])
        sub = long[long.gene_list.isin(genes)]
        n_genes_covered = sub.gene_list.nunique()
        region_counts = sub.region_list.value_counts().to_dict()
        enh = sub[sub.enhancer == "TRUE"]
        n_genes_enh = enh.gene_list.nunique()
        pct_enh = n_genes_enh / len(genes)
        print(f"\n  {name} ({len(genes)} genes resolved)")
        print(f"    genes with >=1 probe (any context): {n_genes_covered} ({100*n_genes_covered/len(genes):.1f}%)")
        print(f"    region breakdown (probe-gene pairs): {region_counts}")
        print(f"    genes with >=1 ENHANCER-flagged probe: {n_genes_enh} ({100*pct_enh:.1f}%)")
        rows.append(dict(signature=name, n_genes=len(genes), n_genes_any_probe=n_genes_covered,
                         pct_genes_any_probe=100 * n_genes_covered / len(genes),
                         n_probe_gene_pairs=len(sub), n_enhancer_probe_gene_pairs=len(enh),
                         n_genes_with_enhancer_probe=n_genes_enh,
                         pct_genes_with_enhancer_probe=100 * pct_enh,
                         **{f"region_{k}": v for k, v in region_counts.items()}))

    cov_df = pd.DataFrame(rows)
    write_and_report(cov_df, C.PROCESSED / "methylation_context_coverage.csv")

    print("\n  NOTE ON WHAT 'Enhancer=TRUE' MEANS HERE: this manifest column flags a")
    print("  probe as overlapping a catalogued (FANTOM5-derived) enhancer element")
    print("  ANYWHERE in the genome, not that the element has been validated as")
    print("  regulating the specific gene the probe is annotated to. It is a")
    print("  genomic-overlap flag, not a confirmed distal-regulatory link.")

    min_enh_cov = cov_df.pct_genes_with_enhancer_probe.min() / 100 if len(cov_df) else 0
    can_test_enhancer = min_enh_cov >= ENHANCER_COVERAGE_MIN
    print(f"\n  Lowest enhancer-probe gene coverage across signatures: {100*min_enh_cov:.1f}%")
    if can_test_enhancer:
        print(f"  Above the {100*ENHANCER_COVERAGE_MIN:.0f}% floor for every signature -- "
              f"contrary to this project's own prior expectation. The enhancer stratum IS modelled below (C3), with the caveat above "
              f"about what the flag actually measures.")
    else:
        print(f"  Below the {100*ENHANCER_COVERAGE_MIN:.0f}% floor for at least one signature -- "
              f"so the enhancer stratum is NOT modelled for "
              f"that signature; an underpowered null would not be evidence of absence.")
    return cov_df, can_test_enhancer


def load_stratum(name):
    df = pd.read_csv(C.METHYLATION_CONTEXT_DIR / f"{name}_M.csv")
    df = df.set_index("gene").T
    df.index.name = None
    return df


def q3_in_stratum(stratum_df, overlap, y, s, label, stratum_name):
    alias_map = S.load_alias_map(required=False)
    sigs = S.load_signatures(stratum_df.columns, alias_map, verbose=False)
    if "DTP_up" not in sigs or "DTP_down" not in sigs:
        print(f"    {stratum_name:<10} DTP not resolvable in this stratum's gene set -- skipped")
        return None
    rank_df, _ = S.score_all(stratum_df.loc[overlap], sigs, verbose=False)
    meth_dtp = rank_df["DTP"].to_numpy()
    expr_dtp = s.loc[overlap, "DTP"].to_numpy()
    auc = y.loc[overlap, "AUC"].to_numpy()

    r1, p1, lo1, hi1 = bootstrap_r(meth_dtp, expr_dtp)
    r2, p2, lo2, hi2 = bootstrap_r(meth_dtp, auc)
    print(f"    {stratum_name:<10} n={len(overlap):<5} meth-DTP vs expr-DTP r={r1:+.3f} "
          f"[{lo1:+.3f},{hi1:+.3f}] p={p1:.4f}  |  meth-DTP vs AUC r={r2:+.3f} p={p2:.4f}")
    return dict(screen=label, stratum=stratum_name, n=len(overlap),
               r_meth_vs_expr=r1, p_meth_vs_expr=p1, ci_lo_meth_vs_expr=lo1, ci_hi_meth_vs_expr=hi1,
               r_meth_vs_auc=r2, p_meth_vs_auc=p2, ci_lo_meth_vs_auc=lo2, ci_hi_meth_vs_auc=hi2)


def c3_rerun_by_stratum(can_test_enhancer):
    banner("C2/C3: Q3 RERUN WITHIN EACH REGULATORY STRATUM")
    strata_to_run = list(STRATA)
    if can_test_enhancer:
        strata_to_run.append("enhancer")
    else:
        print("  Enhancer stratum excluded from modelling per the C1 coverage floor.")

    stratum_dfs = {}
    for st in strata_to_run:
        name = f"promoter_{st}" if st in STRATA else "enhancer"
        path = C.METHYLATION_CONTEXT_DIR / f"{name}_M.csv"
        if not path.exists():
            print(f"  !! {name}_M.csv not found -- skipping stratum {st}")
            continue
        stratum_dfs[st] = load_stratum(name)
        print(f"  loaded stratum {st:<10} {stratum_dfs[st].shape[1]} genes x {stratum_dfs[st].shape[0]} samples")

    rows = []
    for label in [C.TRAIN, C.TEST]:
        banner(f"Q3 per stratum -- {label}")
        Xe, y, s = solid_screen(label)
        for st, df in stratum_dfs.items():
            overlap = sorted(set(Xe.index) & set(df.index))
            if len(overlap) < 30:
                print(f"    {st:<10} overlap n={len(overlap)} -- too small, skipped")
                continue
            row = q3_in_stratum(df, overlap, y, s, label, st)
            if row:
                rows.append(row)

    res_df = pd.DataFrame(rows)
    if len(res_df):
        res_df["q_meth_vs_expr"] = np.nan
        res_df["q_meth_vs_auc"] = np.nan
        for label, sub in res_df.groupby("screen"):
            res_df.loc[sub.index, "q_meth_vs_expr"] = false_discovery_control(sub["p_meth_vs_expr"].to_numpy())
            res_df.loc[sub.index, "q_meth_vs_auc"] = false_discovery_control(sub["p_meth_vs_auc"].to_numpy())
    write_and_report(res_df, C.PROCESSED / "methylation_context_results.csv")
    return res_df


def main():
    cov_df, can_test_enhancer = c1_coverage_audit()
    res_df = c3_rerun_by_stratum(can_test_enhancer)

    banner("WRITING DOC")
    write_doc(cov_df, res_df, can_test_enhancer)

    banner("DONE")


def write_doc(cov_df, res_df, can_test_enhancer):
    lines = []
    lines.append("# Methylation by regulatory context\n")
    lines.append(
        "Follow-up to `methylation_models.md`: that test "
        "pooled all promoter-associated probes together. This reruns the same "
        "meth-DTP-vs-expression-DTP test **stratified by CpG-island relation "
        "and by an enhancer-probe flag**, to check whether pooling hid a "
        "context-specific signal.\n"
    )

    lines.append("## C1: coverage audit, reported before any model is fit\n")
    lines.append(
        "Array-design coverage across all 485,512 HumanMethylation450 probes, "
        "before the cross-reactive/SNP/sex-chromosome QC filter used for "
        "modelling (that filter is not gene-set-selective).\n\n"
        "**What the `Enhancer` flag means**: a probe overlaps a catalogued "
        "(FANTOM5-derived) enhancer element somewhere in the genome -- it is a "
        "genomic-overlap flag, not a validated regulatory link from that "
        "element to the specific gene the probe is annotated to.\n\n"
    )
    lines.append("| Signature | genes | genes w/ any probe | genes w/ enhancer probe |\n|---|---|---|---|\n")
    for _, r in cov_df.iterrows():
        lines.append(f"| {r.signature} | {int(r.n_genes)} | "
                     f"{int(r.n_genes_any_probe)} ({r.pct_genes_any_probe:.1f}%) | "
                     f"{int(r.n_genes_with_enhancer_probe)} ({r.pct_genes_with_enhancer_probe:.1f}%) |\n")
    if can_test_enhancer:
        lines.append(
            f"\nEnhancer-probe coverage exceeds the {100*ENHANCER_COVERAGE_MIN:.0f}% floor for "
            f"every signature -- **contrary to this project's own prior expectation** "
            f". Reported plainly rather than silently revised: the enhancer stratum "
            f"is modelled below, but the caveat on what the flag means still applies -- this "
            f"is genomic overlap with a catalogued regulatory element, not proof the array is "
            f"reading the DTP-relevant enhancer for any given gene.\n"
        )
    else:
        lines.append(
            f"\nAt least one signature falls below the {100*ENHANCER_COVERAGE_MIN:.0f}% floor. "
            f"The array cannot test the enhancer "
            f"hypothesis for that signature, stated plainly rather than modelled as an "
            f"underpowered null.\n"
        )

    lines.append("\n## C2/C3: Q3 rerun per regulatory stratum\n")
    lines.append(
        "Same construction as Q3 in 08_methylation.py (`lib.signatures.score_all()` applied to a "
        "methylation M-value matrix in place of expression, giving a "
        "methylation-space DTP score), rerun separately in each stratum instead "
        "of once on pooled promoter probes. **Multiple-testing burden**: one test "
        "per stratum, 7 strata per screen -- raw p and Benjamini-Hochberg q are "
        "both reported for BOTH the meth-DTP-vs-expr-DTP test and the "
        "meth-DTP-vs-AUC test (q computed within each screen across the 7 strata); "
        "treat isolated raw-p hits with q not significant as noise, not "
        "discoveries.\n\n"
    )
    if len(res_df):
        lines.append("| Screen | Stratum | n | meth-DTP vs expr-DTP | q | meth-DTP vs AUC | q |\n")
        lines.append("|---|---|---|---|---|---|---|\n")
        for _, r in res_df.iterrows():
            sig1 = "*" if r.p_meth_vs_expr < 0.05 else ""
            qsig = "**" if r.q_meth_vs_expr < 0.05 else ""
            sig2 = "*" if r.p_meth_vs_auc < 0.05 else ""
            qsig2 = "**" if r.q_meth_vs_auc < 0.05 else ""
            lines.append(
                f"| {r.screen} | {r.stratum} | {int(r.n)} | "
                f"r={r.r_meth_vs_expr:+.3f} [{r.ci_lo_meth_vs_expr:+.3f},{r.ci_hi_meth_vs_expr:+.3f}] "
                f"p={r.p_meth_vs_expr:.4f}{sig1} | q={r.q_meth_vs_expr:.4f}{qsig} | "
                f"r={r.r_meth_vs_auc:+.3f} p={r.p_meth_vs_auc:.4f}{sig2} | q={r.q_meth_vs_auc:.4f}{qsig2} |\n"
            )
        n_q_sig = int((res_df.q_meth_vs_expr < 0.05).sum())
        n_auc_sig_raw = int((res_df.p_meth_vs_auc < 0.05).sum())
        n_auc_sig_q = int((res_df.q_meth_vs_auc < 0.05).sum())
        lines.append(
            f"\n{n_q_sig} of {len(res_df)} stratum x screen combinations remain significant "
            f"at q<0.05 for meth-DTP vs expr-DTP after BH correction. For meth-DTP vs AUC -- "
            f"the question that actually decides whether the pooled-promoter negative was a coverage "
            f"artefact of pooling promoter contexts -- {n_auc_sig_raw} of {len(res_df)} are "
            f"nominally significant at raw p<0.05, and {n_auc_sig_q} of {len(res_df)} survive "
            f"BH correction (q<0.05, corrected across the 7 strata within each screen).\n"
        )

        replicated = []
        for stratum, sub in res_df.groupby("stratum"):
            if (sub.q_meth_vs_auc < 0.05).all() and len(sub) == 2:
                replicated.append(stratum)
        single_screen = sorted(set(res_df[res_df.q_meth_vs_auc < 0.05].stratum) - set(replicated))

        if not replicated:
            lines.append(
                "\n**No stratum shows a BH-corrected meth-DTP-to-AUC association that "
                "replicates in both screens.** "
            )
            if single_screen:
                lines.append(
                    f"{len(single_screen)} stratum/strata clear q<0.05 in one screen only "
                    f"({', '.join(single_screen)}) -- treated as noise, not a discovery, "
                    "absent replication.\n"
                )
            else:
                lines.append(
                    "The pooled-promoter negative is not explained by CpG-island-relation "
                    "pooling, nor by the array's inability to see enhancers where coverage "
                    "allowed a test -- the methylation-derived DTP score does not predict "
                    "5-FU response in any single regulatory context checked here either.\n"
                )
        else:
            rep_rows = res_df[res_df.stratum.isin(replicated)]
            lines.append(
                "\n**Replicated finding -- survives BH correction in BOTH screens "
                "independently:** "
                + "; ".join(
                    f"{r.stratum} ({r.screen}: r={r.r_meth_vs_auc:+.3f}, q={r.q_meth_vs_auc:.4f})"
                    for _, r in rep_rows.iterrows()
                )
                + ". This directly contradicts the pooled-promoter negative "
                  "(meth-DTP vs AUC p>0.24 in both screens) for these specific "
                  "contexts -- pooling all promoter probes together hid a real, "
                  "reproducible, context-specific association. This is the strongest "
                  "evidence in this analysis and should be treated as a discovery, not noise.\n"
            )
            if single_screen:
                lines.append(
                    f"\nSingle-screen-only hits (q<0.05 in one screen, not the other): "
                    f"{', '.join(single_screen)} -- weaker evidence, not treated as "
                    "replicated.\n"
                )
    else:
        lines.append("No stratum produced a usable overlap -- see the run log.\n")

    lines.append("\n## Outputs\n\n```\n")
    lines.append("data/processed/methylation_context_coverage.csv\n")
    lines.append("data/processed/methylation_by_context/*.parquet-equivalent (csv)\n")
    lines.append("data/processed/methylation_context_results.csv\n")
    lines.append("```\n")

    write_report("methylation_context.md", "".join(lines))


if __name__ == "__main__":
    main()
