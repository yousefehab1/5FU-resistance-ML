"""
lib/signatures.py
==================

Signature loading and scoring, shared by every stage that touches a
signature -- both Arm A (GDSC cell lines) and Arm B (HCT116 time-course).

WHY THIS IS ONE MODULE
-----------------------
Arm A and Arm B must score signatures identically, or comparing them is
meaningless. Gene-symbol vintage is the classic way two copies drift: the
same signature recovered 91% of its genes in GDSC but 67% in the
time-course until both datasets were routed through one alias-resolving
loader.

CONTENTS
--------
  load_alias_map()        HGNC alias -> official symbol
  resolve_symbols()       match signature genes to a dataset's gene universe
  load_signatures()       read data/raw/signatures/, resolve, report recovery
  rank_matrix()            within-sample percentile ranks
  background_score()       rank score calibrated against an exact analytic null
  ssgsea_score()            ssGSEA enrichment score (sensitivity check only)
  score_all()               score every signature, combine DTP_up/DTP_down
"""

import re
import sys

import numpy as np
import pandas as pd

import config as C


def load_alias_map(required=True):
    """
    Load the HGNC alias -> official symbol table (exported once by
    scripts/one_time/export_hgnc_aliases.R). Without it, datasets whose gene
    symbols come from different annotation vintages (e.g. GDSC's current
    symbols vs. the time-course's older ones) silently score different gene
    sets under the same signature name.
    """
    if not C.ALIAS_MAP_FILE.exists():
        msg = (f"  !! Alias map not found: {C.ALIAS_MAP_FILE}\n"
               f"     Run:  Rscript scripts/one_time/export_hgnc_aliases.R")
        if required:
            print(msg)
            sys.exit(1)
        print(msg.replace("!!", "NOTE:") + "\n     Continuing WITHOUT alias resolution.")
        return {}

    m = pd.read_csv(C.ALIAS_MAP_FILE)
    return dict(zip(m["alias"].astype(str), m["symbol"].astype(str)))


def resolve_symbols(genes, universe, alias_map):
    """
    Match signature genes to a dataset's gene universe via HGNC synonyms.

      1. Genes already in the universe are kept as-is.
      2. Each remaining gene is reduced to its official HGNC symbol, and the
         universe is reduced the same way.
      3. A synonym match is accepted ONLY when exactly one universe entry has
         that official symbol. Ties are dropped, never guessed.

    Returns (matched_universe_names, n_direct, n_by_alias, n_unresolved).
    The returned names are the DATASET's names (they must index the
    expression matrix), not the signature's.
    """
    universe = set(universe)
    direct = [g for g in genes if g in universe]
    remaining = [g for g in genes if g not in universe]

    if not remaining or not alias_map:
        return sorted(set(direct)), len(set(direct)), 0, len(set(remaining))

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

    A signature with low recovery is scoring something other than what its
    authors defined -- that needs saying out loud, not discovering later.
    """
    if not C.SIGDIR.exists() or not list(C.SIGDIR.glob("*.txt")):
        print(f"  !! No signature files in {C.SIGDIR}")
        sys.exit(1)

    if alias_map is None:
        alias_map = load_alias_map()

    universe = set(universe)
    sigs = {}

    for path in sorted(C.SIGDIR.glob("*.txt")):
        genes = [ln.strip() for ln in open(path) if ln.strip()]
        if genes and genes[0].lower() in ("gene", "symbol", "gene_symbol", "x"):
            genes = genes[1:]                      # tolerate a stray header

        if sum(g.startswith("ENSG") for g in genes) > 0.5 * len(genes):
            print(f"  !! {path.name} contains Ensembl IDs, not HGNC symbols.")
            sys.exit(1)

        # Literal "None"/"NA"/etc. entries are failed upstream ID conversions
        # written as text, not gene names. Counting them inflates the
        # denominator and must not happen silently.
        n_placeholder = sum(g in C.PLACEHOLDERS for g in genes)
        genes = [g for g in genes if g not in C.PLACEHOLDERS]

        # Repair spreadsheet number damage: "VNN1.00" -> "VNN1". Applied only
        # when the stripped symbol IS in the universe, so nothing is guessed.
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

        name = C.CANON.get(path.stem.lower(), path.stem)
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
                print(f"               dropped {n_placeholder} placeholder entries "
                      f"(failed conversions at source)")
            if fixed:
                print(f"               repaired: {', '.join(fixed)}")

        sigs[name] = matched

    return sigs


def rank_matrix(expr):
    """
    Within-sample percentile ranks (axis=1 ranks across genes within a row).

    Ranks depend only on gene ORDER within one sample, giving two properties
    this project needs: leakage-safe (nothing estimated across samples, so
    no test-fold information can cross into training), and platform-robust
    (a library-size scale factor cannot change within-sample gene order, so
    log2-TPM data and raw-count data score the same way with no extra
    normalisation step).
    """
    return expr.rank(axis=1, pct=True)


def background_score(ranks, genes):
    """
    Signature score calibrated against the null of random gene sets of the
    same size: (observed - null mean) / null sd, per sample.

    Curated signature genes tend to be well-expressed (partly how they got
    noticed and published), so an uncorrected score partly measures "how
    highly expressed are well-expressed genes here" and every signature ends
    up spuriously correlated however little they share biologically.

    The null is computed ANALYTICALLY, not by Monte Carlo sampling. An
    earlier Monte Carlo version (200 random draws) was found to be far too
    few: rescoring with a different seed gave r=0.74 for MYC and r=0.86 for
    CellCycle against itself -- i.e. those scores were substantially sampling
    noise. The mean of n values drawn WITHOUT replacement from a finite
    population of N has, exactly:

        mean = population mean
        sd   = sqrt( population variance / n * (N - n) / (N - 1) )

    (the last factor is the finite population correction; the population is
    the sample's own gene ranks). This is exact, deterministic, seed-free.
    """
    n, N = len(genes), ranks.shape[1]
    obs = ranks[genes].mean(axis=1).to_numpy()
    pop_mean = ranks.mean(axis=1).to_numpy()
    pop_var = ranks.var(axis=1, ddof=1).to_numpy()

    null_sd = np.sqrt(pop_var / n * (N - n) / (N - 1))
    null_sd[null_sd == 0] = np.nan
    return (obs - pop_mean) / null_sd


def ssgsea_score(ranks, genes, alpha=0.25, batch=100):
    """
    ssGSEA enrichment score (Barbie et al. 2009). Sensitivity check only --
    its final normalisation divides by the range across samples, so it
    cannot score a single sample alone and is not comparable across datasets,
    which is why `background_score` is the primary method here.
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

    # Bidirectional DTP cancels technical variation that pushes all genes in
    # a sample the same way (library size, RNA quality, tumour purity).
    if {"DTP_up", "DTP_down"} <= set(rank_df.columns):
        for df in (rank_df, ss_df):
            df["DTP"] = df["DTP_up"] - df["DTP_down"]
            df.drop(columns=["DTP_up", "DTP_down"], inplace=True)
        if verbose:
            print("  combined DTP_up - DTP_down -> DTP (bidirectional)")
    elif "DTP_up" in rank_df.columns and verbose:
        print("  !! DTP_down missing -- DTP scored UP-ONLY.")

    return rank_df, ss_df
