"""Signature loading and scoring, shared by Arm A (cell lines) and Arm B (time course).

Both arms must score identically, so the code lives here once.

  load_alias_map()     HGNC alias -> official symbol (exported from R)
  resolve_symbols()    match signature genes to a dataset's gene universe
  load_signatures()    read data/raw/signatures/, resolve, report recovery
  rank_matrix()        within-sample percentile ranks
  background_score()   rank score z-scored against the exact random-gene-set null
  ssgsea_score()       ssGSEA enrichment score (sensitivity check)
"""

import re
import sys

import numpy as np
import pandas as pd

from .paths import raw_dir

# NOTE: three failure paths below call sys.exit(1) rather than raising. That is
# how this module behaved as a script-directory import, every caller is a stage
# that would exit anyway, and changing it here would change what a broken
# signature directory does to a pipeline run. It is worth revisiting as a
# deliberate change, not as part of a move.

RAW = raw_dir()
SIGDIR = RAW / "signatures"
ALIAS_MAP = RAW / "hgnc_alias_map.csv"

PLACEHOLDERS = {"None", "NA", "NaN", "null", "-", ""}

# Canonical feature names, so DTP_UP.txt / DTP_up.txt / dtp_up.txt all resolve
# to the same feature. Without this the bidirectional DTP combination fails
# silently on a filename capitalisation difference.
CANON = {"dtp_up": "DTP_up", "dtp_down": "DTP_down",
         "rsc": "RSC", "cbc": "CBC", "fetal": "Fetal",
         "cellcycle": "CellCycle", "cell_cycle": "CellCycle",
         "myc": "MYC", "ibd": "IBD"}


# ============================================================================
# SYMBOL RESOLUTION
# ============================================================================

def load_alias_map(required=True):
    """
    Load the HGNC alias -> official symbol table exported by
    scripts/export_hgnc_aliases.R.

    Exported from org.Hs.eg.db rather than reimplemented in Python so that this
    project and dtp-external-validation resolve symbols against the SAME
    annotation. A different source would resolve some symbols differently and
    the two projects' numbers would silently stop being comparable.
    """
    if not ALIAS_MAP.exists():
        msg = (f"  !! Alias map not found: {ALIAS_MAP}\n"
               f"     Run:  Rscript scripts/export_hgnc_aliases.R\n"
               f"     Without it, the time-course (old symbols) and GDSC (current\n"
               f"     symbols) score different gene sets under the same name.")
        if required:
            print(msg)
            sys.exit(1)
        print(msg.replace("!!", "NOTE:") + "\n     Continuing WITHOUT alias resolution.")
        return {}

    m = pd.read_csv(ALIAS_MAP)
    return dict(zip(m["alias"].astype(str), m["symbol"].astype(str)))


def resolve_symbols(genes, universe, alias_map):
    """
    Match signature genes to a dataset's gene universe, using HGNC synonyms.

    Mirrors resolve_against_universe() in dtp-external-validation/R/symbol_aliases.R:

      1. Genes already in the universe are kept as-is.
      2. Each remaining gene is reduced to its official HGNC symbol, and the
         universe is reduced the same way.
      3. A synonym match is accepted ONLY when exactly one universe entry has
         that official symbol. Ties are dropped, never guessed.

    Returns (matched_universe_names, n_direct, n_by_alias, n_unresolved).

    Note the returned names are the DATASET's names, not the signature's --
    they have to index the expression matrix.
    """
    universe = set(universe)
    direct = [g for g in genes if g in universe]
    remaining = [g for g in genes if g not in universe]

    if not remaining or not alias_map:
        return sorted(set(direct)), len(set(direct)), 0, len(set(remaining))

    # Canonical form of every universe entry, keeping only unambiguous targets.
    canon_to_universe = {}
    for u in universe:
        c = alias_map.get(u, u)
        canon_to_universe.setdefault(c, []).append(u)

    by_alias, unresolved = [], []
    for g in remaining:
        c = alias_map.get(g, g)
        hits = canon_to_universe.get(c, [])
        if len(hits) == 1:
            by_alias.append(hits[0])
        else:
            unresolved.append(g)     # 0 hits = absent, >1 = ambiguous

    matched = sorted(set(direct) | set(by_alias))
    return matched, len(set(direct)), len(set(by_alias)), len(set(unresolved))


def load_signatures(universe, alias_map=None, verbose=True):
    """
    Read every .txt in data/raw/signatures/, resolve against `universe`, and
    report recovery. Signature files are HGNC symbols, one per line, no header.

    The recovery percentage matters: a signature with 60% of its genes present
    is scoring something other than what its authors defined, and that needs
    saying out loud rather than being discovered later.
    """
    if not SIGDIR.exists() or not list(SIGDIR.glob("*.txt")):
        print(f"  !! No signature files in {SIGDIR}")
        sys.exit(1)

    if alias_map is None:
        alias_map = load_alias_map()

    universe = set(universe)
    sigs = {}

    for path in sorted(SIGDIR.glob("*.txt")):
        genes = [ln.strip() for ln in open(path) if ln.strip()]
        if genes and genes[0].lower() in ("gene", "symbol", "gene_symbol", "x"):
            genes = genes[1:]                      # tolerate a stray header

        if sum(g.startswith("ENSG") for g in genes) > 0.5 * len(genes):
            print(f"  !! {path.name} contains Ensembl IDs, not HGNC symbols.")
            sys.exit(1)

        # Literal "None" entries are failed ID conversions written as text, not
        # gene names. Counting them inflates the denominator: DTP_UP has 28, so
        # its apparent 91% recovery is really 100% of the 285 real genes.
        n_placeholder = sum(g in PLACEHOLDERS for g in genes)
        genes = [g for g in genes if g not in PLACEHOLDERS]

        # Repair spreadsheet number damage: VNN1 -> "VNN1.00". Applied only when
        # the stripped symbol IS in the universe, so nothing is guessed.
        repaired, fixed = [], []
        for g in genes:
            if re.fullmatch(r".+\.\d+", g) and g not in universe:
                stripped = re.sub(r"\.\d+$", "", g)
                if stripped in universe:
                    fixed.append(f"{g}->{stripped}")
                    repaired.append(stripped)
                    continue
            repaired.append(g)
        genes = repaired

        matched, n_direct, n_alias, n_unres = resolve_symbols(genes, universe, alias_map)

        name = CANON.get(path.stem.lower(), path.stem)
        pct = 100 * len(matched) / len(genes) if genes else 0
        if verbose:
            flag = "" if pct >= 80 else "   <-- LOW RECOVERY"
            src = f"  (from {path.name})" if name != path.stem else ""
            print(f"  {name:12} {len(genes):>4} genes | {len(matched):>4} matched "
                  f"({pct:5.1f}%){flag}{src}")
            if n_alias:
                print(f"               {n_direct} direct, {n_alias} via HGNC synonym, "
                      f"{n_unres} unresolved")
            if n_placeholder:
                print(f"               dropped {n_placeholder} 'None' placeholders "
                      f"(failed conversions at source)")
            if fixed:
                print(f"               repaired: {', '.join(fixed)}")

        sigs[name] = matched

    return sigs


# ============================================================================
# SCORING
# ============================================================================

def rank_matrix(expr):
    """
    Within-sample percentile ranks. axis=1 ranks across genes within each row.

    Ranks depend only on the ORDER of genes inside one sample, which gives two
    properties this project needs:

    * LEAKAGE-SAFE. Nothing is estimated across samples, so no information can
      cross from a test fold into training.
    * PLATFORM-ROBUST. Arm A is log2 TPM from RSEM; Arm B is raw counts from a
      different pipeline. Library size scales every gene in a sample by the same
      factor and cannot change their order, so no CPM or VST is needed and both
      arms are scored the same way.
    """
    return expr.rank(axis=1, pct=True)


def background_score(ranks, genes):
    """
    Signature score calibrated against the null of random gene sets of the same
    size: (observed - null mean) / null sd, computed per sample.

    WHY CALIBRATION IS NEEDED
    -------------------------
    Genes in any curated signature tend to be well-expressed -- that is partly
    how they got noticed and published. Uncorrected, every signature's score
    partly measures "how highly expressed are well-expressed genes in this
    sample", so all signatures correlate spuriously however little they share.

    WHY THE NULL IS ANALYTIC RATHER THAN SAMPLED
    --------------------------------------------
    An earlier version drew 200 random gene sets per signature and used their
    empirical mean and sd. That is Monte Carlo, and 200 draws turned out to be
    far too few: rescoring the same data with a different seed gave r = 0.74 for
    MYC and 0.86 for CellCycle. Those scores were substantially sampling noise,
    and it propagated -- MYC and CellCycle were the two features that flipped
    sign between GDSC1 and GDSC2, and MYC was the one where rank-based and
    ssGSEA scoring disagreed. Some of that "instability" was ours, not the data's.

    No sampling is needed. The mean of n values drawn WITHOUT replacement from a
    finite population of N has, exactly:

        mean = population mean
        sd   = sqrt( population variance / n  *  (N - n) / (N - 1) )

    the last factor being the finite population correction. Here the population
    is the sample's own gene ranks. This is exact, deterministic, seed-free, and
    about 66x faster than 5,000 Monte Carlo draws -- which themselves only
    approximate it (r = 0.99+ against this formula).
    """
    n, N = len(genes), ranks.shape[1]
    obs = ranks[genes].mean(axis=1).to_numpy()
    pop_mean = ranks.mean(axis=1).to_numpy()
    pop_var = ranks.var(axis=1, ddof=1).to_numpy()

    null_sd = np.sqrt(pop_var / n * (N - n) / (N - 1))
    null_sd[null_sd == 0] = np.nan          # avoid divide-by-zero
    return (obs - pop_mean) / null_sd


def ssgsea_score(ranks, genes, alpha=0.25, batch=100):
    """
    ssGSEA enrichment score (Barbie et al. 2009). Sensitivity check only (D13).

    Within each sample, order genes high to low, then walk the list building two
    cumulative curves: P_in (fraction of the signature seen so far, weighted by
    rank^alpha) and P_out (fraction of all other genes, unweighted). The score is
    the area between them.

    The final normalisation divides by the range across samples, which is why
    ssGSEA cannot score a single sample alone and is not comparable across
    datasets -- the reason it is not the primary method here.
    """
    cols = ranks.columns.to_numpy()
    in_set = np.isin(cols, np.asarray(genes))
    n_genes, n_in = len(cols), int(in_set.sum())
    if n_in == 0:
        return np.full(len(ranks), np.nan)

    X = ranks.to_numpy(dtype="float32")
    weights = np.arange(n_genes, 0, -1, dtype="float32") ** alpha
    out = np.empty(len(ranks), dtype="float32")

    for start in range(0, len(ranks), batch):
        block = X[start:start + batch]
        order = np.argsort(-block, axis=1)
        sorted_in = in_set[order]

        cum_in = np.cumsum(weights[None, :] * sorted_in, axis=1)
        cum_in /= cum_in[:, -1][:, None]
        cum_out = np.cumsum(~sorted_in, axis=1) / (n_genes - n_in)

        out[start:start + batch] = (cum_in - cum_out).sum(axis=1)

    rng_ = out.max() - out.min()
    return out / rng_ if rng_ > 0 else out


def score_all(expr, sigs, verbose=True):
    """
    Score every signature with both methods and combine DTP up/down.

    Returns (rank_based_scores, ssgsea_scores) as DataFrames indexed like expr.
    """
    if verbose:
        print("  ranking genes within each sample ...")
    ranks = rank_matrix(expr)

    rank_df = pd.DataFrame(index=expr.index)
    ss_df = pd.DataFrame(index=expr.index)
    for name, genes in sigs.items():
        if genes:
            rank_df[name] = background_score(ranks, genes)
            ss_df[name] = ssgsea_score(ranks, genes)

    # D12: bidirectional DTP cancels technical variation that pushes all genes
    # in a sample the same way (library size, RNA quality, tumour purity).
    if {"DTP_up", "DTP_down"} <= set(rank_df.columns):
        for df in (rank_df, ss_df):
            df["DTP"] = df["DTP_up"] - df["DTP_down"]
            df.drop(columns=["DTP_up", "DTP_down"], inplace=True)
        if verbose:
            print("  combined DTP_up - DTP_down -> DTP (bidirectional)")
    elif "DTP_up" in rank_df.columns and verbose:
        print("  !! DTP_down missing -- DTP scored UP-ONLY, not the agreed feature (D12).")

    return rank_df, ss_df
