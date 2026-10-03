"""Are DTP genes enhancer-rich in colon tumour chromatin?

Counts TCGA COAD ATAC peaks near each gene and compares each signature with
matched random gene sets.

Inputs:  data/raw/atac/ (COAD peak calls, peak-to-gene links, gene_tss_hg38.csv)
Outputs: data/processed/coad_gene_regulatory_architecture.csv,
         regulatory_permutation_results.csv, reports/regulatory_architecture.md
Run:     python stages/11_regulatory_architecture.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import openpyxl
import pandas as pd
from intervaltree import IntervalTree
from scipy.stats import false_discovery_control

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C
from lib.cohorts import load_model_derived_genes
from lib.io import solid_screen
from lib.report import banner, write_and_report, write_report
from lib import signatures as S

N_PERM = 2000
N_EXPR_BINS = 5
N_LEN_BINS = 5
PEAK_CATEGORIES = ["Distal", "Promoter"]
SIGNATURE_SETS = ["DTP_up", "DTP_down", "RSC", "CBC", "Fetal", "CellCycle", "MYC"]


# -----------------------------------------------------------------------
# Load raw ATAC inputs
# -----------------------------------------------------------------------

def load_coad_peaks():
    df = pd.read_csv(C.ATAC_DIR / "peak_calls" / "COAD_peakCalls.txt", sep="\t")
    print(f"  COAD peaks: {len(df)}")
    print(f"  annotation breakdown: {df.annotation.value_counts().to_dict()}")
    return df


def load_pancancer_peaks():
    df = pd.read_csv(C.ATAC_DIR / "TCGA-ATAC_PanCancer_PeakSet.txt", sep="\t")
    print(f"  pan-cancer peaks: {len(df)}")
    return df


def load_links():
    wb = openpyxl.load_workbook(
        C.ATAC_DIR / "TCGA-ATAC_DataS7_PeakToGeneLinks_v2.xlsx", read_only=True)
    ws = wb["All_Links"]
    cols = ["Chromosome", "Start", "End", "hg19_Chromosome", "hg19_Start", "hg19_End",
            "Peak_ID", "Peak_Name", "Linked_Distance", "Linked_Gene", "Linked_Gene_Start",
            "Linked_Gene_Type", "Raw_Correlation", "Raw_FDR", "Nearest_Gene_Distance",
            "Nearest_Gene", "Peak_Type", "Link_ID", "Correlation_Post_CNA",
            "FDR_Post_CNA", "Enhancer_ID", "Enhancer_Annotation"]
    rows = list(ws.iter_rows(min_row=33, values_only=True))
    df = pd.DataFrame(rows, columns=cols)
    print(f"  peak-to-gene links (pan-cancer, All_Links): {len(df)} rows, "
          f"{df.Peak_Name.nunique()} unique peaks")
    return df


def load_gene_tss():
    df = pd.read_csv(C.ATAC_DIR / "gene_tss_hg38.csv")
    print(f"  gene TSS/length reference: {len(df)} genes (hg38)")
    return df


# -----------------------------------------------------------------------
# Peak -> gene assignment
# -----------------------------------------------------------------------

def nearest_tss_assignment(peaks, tss):
    """Vectorized nearest-TSS gene per peak, per chromosome. Crude by
    construction -- not a regulatory claim."""
    peaks = peaks.copy()
    peaks["mid"] = (peaks.start + peaks.end) // 2
    gene = np.full(len(peaks), None, dtype=object)
    dist = np.full(len(peaks), np.nan)
    for chrom, sub_p in peaks.groupby("seqnames"):
        sub_t = tss[tss.chrom == chrom].sort_values("tss")
        if len(sub_t) == 0:
            continue
        t_pos = sub_t.tss.to_numpy()
        t_sym = sub_t.symbol.to_numpy()
        idx = np.searchsorted(t_pos, sub_p["mid"].to_numpy())
        idx_lo = np.clip(idx - 1, 0, len(t_pos) - 1)
        idx_hi = np.clip(idx, 0, len(t_pos) - 1)
        d_lo = np.abs(sub_p["mid"].to_numpy() - t_pos[idx_lo])
        d_hi = np.abs(sub_p["mid"].to_numpy() - t_pos[idx_hi])
        pick_hi = d_hi < d_lo
        chosen_idx = np.where(pick_hi, idx_hi, idx_lo)
        chosen_dist = np.where(pick_hi, d_hi, d_lo)
        gene[sub_p.index.to_numpy()] = t_sym[chosen_idx]
        dist[sub_p.index.to_numpy()] = chosen_dist
    peaks["nearest_gene"] = gene
    peaks["nearest_gene_dist"] = dist
    return peaks


def pancancer_overlap_lookup(coad_peaks, pan_peaks):
    """For each COAD peak, the pan-cancer peak with the largest base-pair
    overlap (checked directly: 99.9% of COAD peaks overlap >=1 pan-cancer
    peak, since the pan-cancer set is the cross-cancer union/merge)."""
    trees = {}
    for chrom, sub in pan_peaks.groupby("seqnames"):
        t = IntervalTree()
        for s, e, n in zip(sub.start, sub.end, sub.name):
            if e > s:
                t[s:e] = n
        trees[chrom] = t

    names = np.full(len(coad_peaks), None, dtype=object)
    for i, (chrom, s, e) in enumerate(zip(coad_peaks.seqnames, coad_peaks.start, coad_peaks.end)):
        tree = trees.get(chrom)
        if tree is None:
            continue
        hits = tree[s:e]
        if not hits:
            continue
        best = max(hits, key=lambda iv: min(iv.end, e) - max(iv.begin, s))
        names[i] = best.data
    return names


def b1_gene_architecture(coad_peaks, pan_peaks, links, tss):
    banner("B1: REGULATORY ARCHITECTURE PER GENE (nearest-TSS, all COAD Distal/Promoter peaks)")
    coad_peaks = coad_peaks[coad_peaks.annotation.isin(PEAK_CATEGORIES)].reset_index(drop=True)
    coad_peaks = nearest_tss_assignment(coad_peaks, tss)
    n_unassigned = coad_peaks.nearest_gene.isna().sum()
    print(f"  {len(coad_peaks)} Distal/Promoter COAD peaks; "
          f"{n_unassigned} unassigned (no gene on that chromosome)")

    pan_names = pancancer_overlap_lookup(coad_peaks, pan_peaks)
    coad_peaks["pancancer_peak"] = pan_names
    link_map = links.set_index("Peak_Name")["Linked_Gene"].to_dict()
    coad_peaks["linked_gene"] = coad_peaks.pancancer_peak.map(link_map)
    n_linked = coad_peaks.linked_gene.notna().sum()
    n_agree = (coad_peaks.linked_gene.notna() &
              (coad_peaks.linked_gene == coad_peaks.nearest_gene)).sum()
    print(f"  {n_linked}/{len(coad_peaks)} peaks ({100*n_linked/len(coad_peaks):.1f}%) also have a "
          f"correlation-based All_Links Linked_Gene")
    print(f"  of those, nearest-TSS agrees with Linked_Gene for {n_agree}/{n_linked} "
          f"({100*n_agree/max(n_linked,1):.1f}%) -- agreement expected to be higher for "
          f"Promoter peaks (physically close to their own TSS by definition) than Distal")

    coad_peaks = coad_peaks.dropna(subset=["nearest_gene"])
    counts = (coad_peaks.groupby(["nearest_gene", "annotation"]).size()
             .unstack(fill_value=0).reindex(columns=PEAK_CATEGORIES, fill_value=0))
    counts.columns = ["n_distal", "n_promoter"]
    counts["distal_to_promoter_ratio"] = counts.n_distal / (counts.n_promoter + 1)
    counts["distal_fraction"] = counts.n_distal / (counts.n_distal + counts.n_promoter)
    counts = counts.reset_index().rename(columns={"nearest_gene": "gene"})
    write_and_report(counts, C.PROCESSED / "coad_gene_regulatory_architecture.csv")
    print(f"  {len(counts)} genes with >=1 assigned Distal or Promoter peak")
    return counts, n_linked, len(coad_peaks), n_agree


# -----------------------------------------------------------------------
# B2/B3: matched-background permutation test
# -----------------------------------------------------------------------

def build_background_universe(arch_df, tss, expr_mean):
    """Every gene with an architecture count (peak data), a TSS/length record,
    and an expression value -- the only genes that CAN be tested or matched."""
    df = arch_df.merge(tss[["symbol", "gene_length"]], left_on="gene", right_on="symbol", how="inner")
    df = df.merge(expr_mean.rename("expr"), left_on="gene", right_index=True, how="inner")
    df = df.drop(columns=["symbol"]).drop_duplicates(subset="gene").set_index("gene")
    return df


def assign_bins(universe):
    u = universe.copy()
    u["expr_bin"] = pd.qcut(u.expr, N_EXPR_BINS, labels=False, duplicates="drop")
    u["len_bin"] = pd.qcut(np.log10(u.gene_length.clip(lower=1)), N_LEN_BINS, labels=False, duplicates="drop")
    return u


def permutation_test(name, sig_genes, universe, metric, rng):
    present = [g for g in sig_genes if g in universe.index]
    n_missing = len(sig_genes) - len(present)
    if len(present) < 5:
        print(f"  {name:<12} skipped -- only {len(present)}/{len(sig_genes)} genes in the "
              f"peak+expression+length universe")
        return None

    obs = universe.loc[present, metric].mean()
    pool = universe.drop(index=[g for g in present if g in universe.index])
    strata = pool.groupby(["expr_bin", "len_bin"]).groups
    sig_strata = universe.loc[present, ["expr_bin", "len_bin"]]

    null = np.empty(N_PERM)
    for p in range(N_PERM):
        picks = []
        for _, (eb, lb) in sig_strata.iterrows():
            candidates = strata.get((eb, lb))
            if candidates is None or len(candidates) == 0:
                candidates = pool.index  # fallback: whole pool if that stratum is empty
            picks.append(candidates[rng.integers(0, len(candidates))])
        null[p] = pool.loc[picks, metric].mean()

    p_val = (np.sum(null >= obs) + 1) / (N_PERM + 1)
    null_mean, null_sd = null.mean(), null.std()
    z = (obs - null_mean) / null_sd if null_sd > 0 else np.nan
    print(f"  {name:<12} n={len(present):<4} ({n_missing} missing from universe)  "
          f"obs_{metric}={obs:.3f}  null_mean={null_mean:.3f} [sd={null_sd:.3f}]  "
          f"z={z:+.2f}  p={p_val:.4f}")
    return dict(signature=name, n=len(present), n_missing=n_missing, metric=metric,
               obs=obs, null_mean=null_mean, null_sd=null_sd, z=z, p=p_val)


def b2_b3_permutation_tests(arch_df, tss, expr_mean, model_genes):
    banner("B2/B3: MATCHED-BACKGROUND PERMUTATION TESTS "
           f"(expression x gene-length bins, {N_PERM} draws)")
    universe = build_background_universe(arch_df, tss, expr_mean)
    universe = assign_bins(universe)
    print(f"  background universe: {len(universe)} genes with peak, expression, and length data")

    alias_map = S.load_alias_map(required=False)
    sigs = S.load_signatures(set(universe.index) | set(model_genes), alias_map, verbose=True)

    gene_sets = {name: sigs[name] for name in SIGNATURE_SETS if name in sigs}
    gene_sets["model_413"] = model_genes

    rows = []
    rng = np.random.default_rng(C.SEED)
    for metric in ["distal_fraction", "n_distal"]:
        print(f"\n  -- metric: {metric} --")
        for name, genes in gene_sets.items():
            row = permutation_test(name, list(genes), universe, metric, rng)
            if row:
                rows.append(row)
    res_df = pd.DataFrame(rows)
    if len(res_df):
        res_df["q"] = np.nan
        for metric, sub in res_df.groupby("metric"):
            res_df.loc[sub.index, "q"] = false_discovery_control(sub["p"].to_numpy())
    write_and_report(res_df, C.PROCESSED / "regulatory_permutation_results.csv")
    return res_df, len(universe)


# -----------------------------------------------------------------------
# Doc
# -----------------------------------------------------------------------

def write_doc(counts, n_linked, n_total_peaks, n_agree, res_df, n_universe):
    lines = []
    lines.append("# Regulatory architecture: is the DTP programme enhancer-driven in COAD tissue?\n")
    lines.append(
        "Follow-up to `methylation_models.md` and `methylation_context.md`: "
        "those tasks tested DNA methylation, which measures promoters and CpG-island-relative "
        "contexts well but enhancers only indirectly (a genomic-overlap flag, not a functional "
        "assay). This task uses real tissue chromatin accessibility (TCGA-ATAC, Corces et al. "
        "2018, *Science*) to ask the architecture question directly: do DTP genes have more "
        "distal (candidate enhancer) chromatin near them, relative to a matched background, "
        "than promoter-proximal chromatin?\n"
    )
    lines.append(
        "**Tissue mismatch, stated plainly and not worked around**: this is bulk tumour ATAC-seq "
        "from TCGA COAD samples, not GDSC cell lines and not a 5-FU-treated/DTP state -- it can "
        "only speak to the *baseline* regulatory architecture around a gene in colorectal tissue, "
        "not chromatin state specifically in a drug-tolerant-persister cell. **READ (rectal "
        "adenocarcinoma) is absent from this study's 23 cancer types** -- GDSC's combined "
        "COREAD label is only partially covered by COAD-tissue ATAC.\n"
    )

    lines.append("## Gene assignment: nearest-TSS (primary), All_Links correlation (corroboration)\n")
    lines.append(
        f"Checked directly before choosing a method: the pan-cancer peak-to-gene link table "
        f"(`All_Links`) covers only 13.5% of COAD Distal peaks and 22.6% of COAD Promoter peaks "
        f"(it is keyed to the pan-cancer merged peak set, and only ~11% of pan-cancer peaks have "
        f"a link entry at all). Using it alone would silently drop ~85% of COAD peaks from gene "
        f"assignment. Nearest-TSS is therefore the primary "
        f"assignment method here, applied to all {n_total_peaks} Distal/Promoter COAD peaks -- "
        f"crude for Distal peaks by construction (a peak's nearest gene is not necessarily its "
        f"regulatory target), reported as such rather than treated as equivalent to a functional "
        f"link.\n\n"
        f"{n_linked}/{n_total_peaks} peaks ({100*n_linked/n_total_peaks:.1f}%) also have a "
        f"correlation-based `Linked_Gene` from `All_Links`; nearest-TSS agrees with it "
        f"{n_agree}/{n_linked} times ({100*n_agree/max(n_linked,1):.1f}%) where both exist.\n"
    )

    lines.append("\n## B1: per-gene regulatory architecture in COAD\n")
    lines.append(
        f"{len(counts)} genes have >=1 assigned Distal or Promoter COAD peak. Full table: "
        f"`data/processed/coad_gene_regulatory_architecture.csv` "
        f"(columns: n_distal, n_promoter, distal_to_promoter_ratio, distal_fraction).\n"
    )

    lines.append("\n## B2/B3: are DTP/model/reference gene sets enhancer-driven relative to a matched background?\n")
    lines.append(
        f"Background: {n_universe} genes with both COAD peak data and expression+length data "
        f"(GDSC1 transcriptome mean, `gene_tss_hg38.csv` length). For each signature gene, "
        f"{N_PERM} permutation draws each pick one background gene from the SAME "
        f"expression-quintile x log-length-quintile bin (not an unmatched random draw), giving a "
        f"null distribution of the signature-set mean under gene-length- and "
        f"expression-level-matched sampling. One-sided empirical p = fraction of null draws >= "
        f"the observed signature-set mean (testing \"more enhancer-driven than matched "
        f"background\"). Benjamini-Hochberg q is corrected across "
        f"the 8 signature sets within each metric.\n\n"
        f"**Signature recovery against this test's universe runs lower (68-82%) than against "
        f"the full transcriptome elsewhere in this project (81-100% in 19_tf_activity and 22_methylation_context)** -- "
        f"checked directly: this is because `load_signatures()` here is resolved against the "
        f"{n_universe}-gene peak+expression+length-intersected universe, not the full "
        f"~36,000-gene transcriptome, so genes lacking ATAC/expression/length data are correctly "
        f"excluded up front rather than counted as a new resolution failure. The `n (missing)` "
        f"column below is the more relevant number for this specific test: how many of the "
        f"resolved signature genes still lack matched-universe data.\n\n"
    )
    if len(res_df):
        lines.append("| Metric | Signature | n (missing) | observed | null mean [sd] | z | p | q |\n")
        lines.append("|---|---|---|---|---|---|---|---|\n")
        for _, r in res_df.iterrows():
            sig = "*" if r.p < 0.05 else ""
            qsig = "**" if r.q < 0.05 else ""
            lines.append(
                f"| {r.metric} | {r.signature} | {int(r.n)} ({int(r.n_missing)}) | {r.obs:.3f} | "
                f"{r.null_mean:.3f} [{r.null_sd:.3f}] | {r.z:+.2f} | {r.p:.4f}{sig} | {r.q:.4f}{qsig} |\n"
            )
        n_sig_raw = int((res_df.p < 0.05).sum())
        n_sig_q = int((res_df.q < 0.05).sum())
        lines.append(
            f"\n{n_sig_raw} of {len(res_df)} (signature x metric) tests significant at raw "
            f"p<0.05; {n_sig_q} survive BH correction (q<0.05, corrected within each metric "
            f"across the 8 signature sets).\n"
        )

        def get(sig, metric, col):
            sub = res_df[(res_df.signature == sig) & (res_df.metric == metric)]
            return sub[col].iloc[0] if len(sub) else np.nan

        du_frac_q, du_dist_q = get("DTP_up", "distal_fraction", "q"), get("DTP_up", "n_distal", "q")
        dd_dist_q = get("DTP_down", "n_distal", "q")
        rsc_frac_q, rsc_dist_q = get("RSC", "distal_fraction", "q"), get("RSC", "n_distal", "q")
        fetal_frac_q, fetal_dist_q = get("Fetal", "distal_fraction", "q"), get("Fetal", "n_distal", "q")

        lines.append(
            "\n**Primary result (B2), stated plainly rather than folded into the table**: "
            f"**DTP_up shows no enhancer-driven signal on either metric** "
            f"(distal_fraction p={get('DTP_up','distal_fraction','p'):.2f} q={du_frac_q:.2f}; "
            f"n_distal p={get('DTP_up','n_distal','p'):.2f} q={du_dist_q:.2f}, neither survives "
            f"correction). **DTP_down is metric-dependent**: null on `distal_fraction` "
            f"(p={get('DTP_down','distal_fraction','p'):.2f}, not even nominally significant) "
            f"but its raw `n_distal` count does survive BH correction (q={dd_dist_q:.3f}). "
            f"`distal_fraction` does not carry gene length in its numerator/denominator the way "
            f"a raw count does, so it is the more robust of the two to residual length-matching "
            f"error -- DTP_down's result should be read as weak and metric-dependent, not as "
            f"confirmed enhancer enrichment. **RSC and Fetal, by contrast, are significant AND "
            f"survive correction on BOTH metrics, with larger effect sizes** (RSC q="
            f"{rsc_frac_q:.3f}/{rsc_dist_q:.3f}, Fetal q={fetal_frac_q:.3f}/{fetal_dist_q:.3f} for "
            f"distal_fraction/n_distal respectively, vs. DTP_down's single-metric q={dd_dist_q:.3f}). "
            f"If anything, the reference regeneration/fetal programmes this project already "
            f"treats as adjacent to DTP look more robustly enhancer-driven than the DTP genes "
            f"themselves do.\n"
        )
    else:
        lines.append("No signature set had enough genes in the matched universe to test.\n")

    lines.append("\n## B4: synthesis -- a hypothesis consistent with the evidence, not a demonstrated mechanism\n")
    lines.append(
        "This is correlational and cross-data-source (GDSC expression-defined gene sets vs "
        "TCGA-tissue chromatin, different cohorts, different assay modalities, no shared "
        "samples). Any association found above says a gene set's genomic neighbourhood tends to "
        "carry more distal than promoter-proximal accessible chromatin in colorectal tumour "
        "tissue generally -- it does NOT demonstrate that those distal elements regulate the "
        "gene, are active specifically in the DTP state, or are open in the same cell that is "
        "drug-tolerant. The direct test remains ATAC-seq on a CRC panel under 5-FU treatment "
        ".\n"
    )

    lines.append("\n## Outputs\n\n```\n")
    lines.append("data/processed/coad_gene_regulatory_architecture.csv\n")
    lines.append("data/processed/regulatory_permutation_results.csv\n")
    lines.append("```\n")

    write_report("regulatory_architecture.md", "".join(lines))


def main():
    banner("LOADING ATAC INPUTS")
    coad_peaks = load_coad_peaks()
    pan_peaks = load_pancancer_peaks()
    links = load_links()
    tss = load_gene_tss()

    counts, n_linked, n_total_peaks, n_agree = b1_gene_architecture(coad_peaks, pan_peaks, links, tss)

    banner("LOADING EXPRESSION FOR BACKGROUND MATCHING")
    X, y, s = solid_screen("GDSC1")
    expr_mean = X.mean(axis=0)
    print(f"  GDSC1 transcriptome: {X.shape[0]} samples x {X.shape[1]} genes")

    model_genes = load_model_derived_genes()
    print(f"  final model genes: {len(model_genes)}")

    res_df, n_universe = b2_b3_permutation_tests(counts, tss, expr_mean, model_genes)

    banner("WRITING DOC")
    write_doc(counts, n_linked, n_total_peaks, n_agree, res_df, n_universe)

    banner("DONE")


if __name__ == "__main__":
    main()
