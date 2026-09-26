"""
lib/io.py
=========

Data loading shared by every stage: X/y/signature-score loading, the
gene-missingness filter, median fill and the haematological exclusion, in
one place so every stage sees the same matrix.
"""

import sys

import pandas as pd

import config as C


def surviving_genes(X, max_missing=C.MAX_GENE_MISSING):
    """Columns whose missing fraction is at or below the threshold (a column
    exactly at the threshold is kept)."""
    m = X.isna().mean()
    return X.columns[m <= max_missing]


def load_expression(label, impute=True):
    """
    One screen's gene matrix, missingness-filtered and (optionally)
    median-filled.

    Both steps are UNSUPERVISED, they never look at the target, so doing them
    outside cross-validation cannot leak. The median is computed before any
    haematological lines are dropped; reversing the two steps changes
    published numbers. impute=False is for callers that only need the column
    names.
    """
    X = pd.read_parquet(C.PROCESSED / f"X_{label}.parquet")
    X = X[surviving_genes(X)]
    if impute and X.isna().any().any():
        X = X.fillna(X.median())
    return X


def load_screen(label, with_signatures=True):
    """(X, y) or (X, y, signature_scores) for one screen, row-aligned. Use
    exclude_haem() afterwards for the solid-tumour cohort."""
    X = load_expression(label)
    y = pd.read_parquet(C.PROCESSED / f"y_{label}.parquet")
    if not with_signatures:
        return X, y
    s = pd.read_parquet(C.PROCESSED / f"signature_scores_{label}.parquet")
    return X, y, s


def exclude_haem(y, *others):
    """
    Boolean mask (and matching filtered frames) dropping haematological
    lineages. Usage: `keep, y, X, s = exclude_haem(y, X, s)`.
    """
    keep = (~y[C.LINEAGE_COL].isin(C.HAEM)).to_numpy()
    filtered = tuple(o[keep] for o in others)
    return (keep, y[keep]) + filtered


def solid_screen(label):
    """(X, y, signature_scores) for one screen's solid-tumour lines."""
    X, y, s = load_screen(label)
    _, y, X, s = exclude_haem(y, X, s)
    return X, y, s


def load_expression_cached(needed_ids):
    """
    Stream data/raw/rnaseq_all_*.csv (long format, ~79M rows, 5.7GB) and
    pivot it into a (models x genes) matrix, keeping only `needed_ids`.
    Cached to data/processed/expression_tpm.parquet after the first run;
    delete that file to force a rebuild.

    `chunksize=` makes read_csv return an ITERATOR so the file is processed
    CHUNK_ROWS rows at a time and everything not needed is discarded as we
    go -- loading the file whole would need well over 10GB of RAM.
    """
    cache = C.PROCESSED / "expression_tpm.parquet"
    if cache.exists():
        print(f"  loading cached {cache.name}")
        expr = pd.read_parquet(cache)
        print(f"  {expr.shape[0]:,} models x {expr.shape[1]:,} genes")
        return expr

    usecols = ["model_id", "gene_symbol", "rsem_tpm", "duplicate", "data_source"]
    kept, rows_read, n_dropped_dup, sources = [], 0, 0, {}

    print(f"  streaming {C.RNASEQ_FILE.name} in {C.CHUNK_ROWS:,}-row chunks")
    print(f"  looking for {len(needed_ids):,} cell lines")

    for i, chunk in enumerate(pd.read_csv(C.RNASEQ_FILE, usecols=usecols,
                                           chunksize=C.CHUNK_ROWS, low_memory=False)):
        rows_read += len(chunk)
        chunk = chunk[chunk["model_id"].isin(needed_ids)]

        # Drop rows Cell Model Passports flags as duplicate (some models were
        # sequenced at both Sanger and Broad); using their own flag is
        # cleaner than averaging two different sequencing runs together.
        if len(chunk):
            is_dup = (chunk["duplicate"].astype(str).str.strip().str.lower()
                      .isin(["true", "1", "1.0", "yes", "y"]))
            n_dropped_dup += int(is_dup.sum())
            chunk = chunk[~is_dup]

        if len(chunk):
            for src, n in chunk["data_source"].value_counts().items():
                sources[src] = sources.get(src, 0) + int(n)
            chunk["rsem_tpm"] = chunk["rsem_tpm"].astype("float32")
            kept.append(chunk[["model_id", "gene_symbol", "rsem_tpm"]])

        if (i + 1) % 10 == 0:
            print(f"    {rows_read:>12,} read | {sum(len(c) for c in kept):>11,} kept")

    print(f"    {rows_read:>12,} read total")
    print(f"    dropped {n_dropped_dup:,} duplicate-flagged rows")
    print(f"    data_source: {sources}")

    if not kept:
        print("  !! No matching cell lines found. Check model_id values.")
        sys.exit(1)

    long_df = pd.concat(kept, ignore_index=True)
    del kept

    print("  pivoting to models x genes ...")
    wide = long_df.pivot_table(index="model_id", columns="gene_symbol",
                                values="rsem_tpm", aggfunc="mean")
    wide.index.name = C.ID_COL
    wide.columns.name = None
    print(f"  matrix: {wide.shape[0]:,} models x {wide.shape[1]:,} genes")

    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    wide.to_parquet(cache)
    print(f"  cached -> {cache.name}")
    return wide


def load_mutation_flags(genes, model_ids=None):
    """
    Binary (model x gene) mutation-flag matrix for `genes`, from
    data/raw/mutations_summary_*.csv.
    """
    mut = pd.read_csv(C.MUTATIONS_FILE, low_memory=False)
    mut = mut[mut["gene_symbol"].isin(genes)]
    flags = (mut.assign(v=1)
             .pivot_table(index="model_id", columns="gene_symbol", values="v",
                          aggfunc="max")
             .fillna(0))
    if model_ids is not None:
        flags = flags.reindex(model_ids).fillna(0)
    return flags
