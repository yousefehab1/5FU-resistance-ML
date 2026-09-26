"""Join 5-FU response to cell-line expression: one X/y pair per screen.

Cell Model Passports is keyed by SANGER_MODEL_ID, the same ID as GDSC, so no
name matching is needed. The first run streams the 5.7 GB expression file and
caches it; delete expression_tpm.parquet to rebuild.

Inputs:  data/processed/fu_gdsc{1,2}_raw.csv
         data/raw/rnaseq_all_20260323.csv, model_list_20260724.csv
Outputs: data/processed/expression_tpm.parquet, X_GDSC{1,2}.parquet, y_GDSC{1,2}.parquet
Run:     python scripts/02_build_modelling_table.py
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd

# ============================================================================
# CONFIG -- exact filenames and column names, verified against the real data
# ============================================================================
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

PROJECT_ROOT = project_root()
RAW = PROJECT_ROOT / "data" / "raw"
PROCESSED = PROJECT_ROOT / "data" / "processed"

# --- input files ---
GDSC1_FU = PROCESSED / "fu_gdsc1_raw.csv"          # written by script 01
GDSC2_FU = PROCESSED / "fu_gdsc2_raw.csv"
RNASEQ   = RAW / "rnaseq_all_20260323.csv"
MODELS   = RAW / "model_list_20260724.csv"

# --- columns in rnaseq_all_20260323.csv (long format, 13 columns) ---
EXPR_MODEL  = "model_id"        # SANGER_MODEL_ID, e.g. SIDM00783
EXPR_GENE   = "gene_symbol"     # HGNC symbol
EXPR_VALUE  = "rsem_tpm"        # NOTE: already log2(TPM+1) -- see below
EXPR_DUP    = "duplicate"       # flags records repeated across data sources
EXPR_SOURCE = "data_source"     # Sanger or Broad

# IMPORTANT: despite the name, rsem_tpm from Cell Model Passports is ALREADY
# log2(TPM + 1), not raw TPM. Confirmed empirically -- values top out around
# 18 and per-sample sums are ~6e4 rather than the 1e6 raw TPM would give.
# Do NOT log-transform it again. See D7 in docs/METHODS.md.

# --- columns in model_list_20260724.csv ---
MODEL_ID_COL = "model_id"
MODEL_META = ["cancer_type", "tissue", "cancer_type_detail", "growth_properties",
              "msi_status", "mutational_burden", "model_name", "gender",
              "age_at_sampling"]

# --- columns kept from the GDSC drug files ---
DRUG_COLS = ["CELL_LINE_NAME", "TCGA_DESC", "LN_IC50", "AUC",
             "RMSE", "Z_SCORE", "MIN_CONC", "MAX_CONC"]

ID_COL = "SANGER_MODEL_ID"     # the join key everywhere downstream
CRC_CODE = "COREAD"
CHUNK_ROWS = 2_000_000         # rows per streaming chunk

EXPR_CACHE = PROCESSED / "expression_tpm.parquet"


from fivefu.report import banner


def require(path):
    """Stop with a clear message if an expected input is missing."""
    if not path.exists():
        print(f"\n  !! Missing required file: {path}")
        print(f"     See this script's docstring for what it is and where it comes from.")
        sys.exit(1)
    return path


# ============================================================================
# EXPRESSION
# ============================================================================

def stream_expression_matrix(needed_ids):
    """
    Stream rnaseq_all_20260323.csv and pivot it into a (models x genes) matrix,
    keeping only the cell lines in `needed_ids`.

    The file is long format -- one row per model x gene, ~79 million rows and
    5.7 GB. Loading it whole would need well over 10 GB of RAM. Instead,
    `chunksize=` makes read_csv return an ITERATOR of DataFrames so we process
    2 million rows at a time and discard what we don't need as we go. This is
    the standard approach for any file larger than memory.
    """
    usecols = [EXPR_MODEL, EXPR_GENE, EXPR_VALUE, EXPR_DUP, EXPR_SOURCE]
    kept, rows_read, n_dropped_dup, sources = [], 0, 0, {}

    print(f"  streaming {RNASEQ.name} in {CHUNK_ROWS:,}-row chunks")
    print(f"  looking for {len(needed_ids):,} cell lines")

    for i, chunk in enumerate(pd.read_csv(RNASEQ, usecols=usecols,
                                          chunksize=CHUNK_ROWS, low_memory=False)):
        rows_read += len(chunk)

        # Keep only the cell lines we need.
        chunk = chunk[chunk[EXPR_MODEL].isin(needed_ids)]

        # Drop rows Cell Model Passports flags as duplicates. Some models were
        # sequenced at both Sanger and Broad; the flag marks the record CMP
        # itself treats as secondary. Using their flag is cleaner than
        # averaging two different sequencing runs together.
        # The column arrives as bool, 0/1 or "True"/"False" depending on how
        # pandas typed that chunk, so normalise to text before testing.
        if len(chunk):
            is_dup = (chunk[EXPR_DUP].astype(str).str.strip().str.lower()
                      .isin(["true", "1", "1.0", "yes", "y"]))
            n_dropped_dup += int(is_dup.sum())
            chunk = chunk[~is_dup]                      # ~ inverts the mask

        if len(chunk):
            for src, n in chunk[EXPR_SOURCE].value_counts().items():
                sources[src] = sources.get(src, 0) + int(n)
            # float32 halves memory versus float64 at precision we don't need.
            chunk[EXPR_VALUE] = chunk[EXPR_VALUE].astype("float32")
            kept.append(chunk[[EXPR_MODEL, EXPR_GENE, EXPR_VALUE]])

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
    print(f"  kept {len(long_df):,} rows for "
          f"{long_df[EXPR_MODEL].nunique():,} cell lines")

    # pivot_table reshapes long -> wide and averages any remaining duplicate
    # model/gene pairs (e.g. one symbol mapped by several gene_ids).
    print("  pivoting to models x genes ...")
    wide = long_df.pivot_table(index=EXPR_MODEL, columns=EXPR_GENE,
                               values=EXPR_VALUE, aggfunc="mean")
    wide.index.name = ID_COL
    wide.columns.name = None

    print(f"  matrix: {wide.shape[0]:,} models x {wide.shape[1]:,} genes")
    return wide


def describe_expression(expr):
    """Sanity-check the scale, so a silent units problem can't slip through."""
    v = expr.to_numpy()
    per_sample_sum = np.nansum(v, axis=1)
    print(f"  value range : {np.nanmin(v):.3f} to {np.nanmax(v):.1f}")
    print(f"  median      : {np.nanmedian(v):.3f}")
    print(f"  sum/sample  : {per_sample_sum.mean():,.0f}  "
          f"(raw TPM would be ~1,000,000)")
    print(f"  -> treating as log2(TPM+1); do NOT log again")

    n_na = int(expr.isna().sum().sum())
    print(f"  missing     : {n_na:,} of {expr.size:,} "
          f"({100 * n_na / expr.size:.2f}%)")
    if n_na:
        print(f"     genes fully missing in >50% of models: "
              f"{int((expr.isna().mean(axis=0) > 0.5).sum()):,}")
        print(f"     (these get filtered in script 03, not here)")


# ============================================================================
# MAIN
# ============================================================================

def main():

    banner("1. CHECKING INPUTS")
    for p in [GDSC1_FU, GDSC2_FU, RNASEQ, MODELS]:
        require(p)
        print(f"  ok  {p.name}")

    banner("2. WHICH CELL LINES DO WE NEED?")
    # Work this out before touching the 5.7 GB file, so most of its rows can
    # be discarded on sight rather than loaded then filtered.
    needed_ids = set()
    for p in [GDSC1_FU, GDSC2_FU]:
        ids = pd.read_csv(p, usecols=[ID_COL])[ID_COL].dropna().unique()
        needed_ids |= set(ids)
        print(f"  {p.name}: {len(ids):,} cell lines")
    print(f"  union: {len(needed_ids):,} distinct cell lines")

    banner("3. EXPRESSION")
    if EXPR_CACHE.exists():
        print(f"  loading cached {EXPR_CACHE.name}")
        expr = pd.read_parquet(EXPR_CACHE)
        print(f"  {expr.shape[0]:,} models x {expr.shape[1]:,} genes")
    else:
        expr = stream_expression_matrix(needed_ids)
        PROCESSED.mkdir(parents=True, exist_ok=True)
        expr.to_parquet(EXPR_CACHE)
        print(f"  cached -> {EXPR_CACHE.name}")
    describe_expression(expr)

    banner("4. MODEL METADATA")
    models = pd.read_csv(MODELS, low_memory=False)
    print(f"  {models.shape[0]:,} models x {models.shape[1]} columns")
    models = (models.rename(columns={MODEL_ID_COL: ID_COL})
                    [[ID_COL] + MODEL_META]
                    .drop_duplicates(ID_COL)
                    .set_index(ID_COL))
    print(f"  keeping: {MODEL_META}")
    print("  NOTE: no doubling_time in this release. The D5 proliferation")
    print("        baseline will use a proliferation gene signature instead")
    print("        (MKI67, PCNA, TOP2A, CCNB1 ...) -- handled in script 03.")

    banner("5. JOINING")
    written = {}
    for label, path in [("GDSC1", GDSC1_FU), ("GDSC2", GDSC2_FU)]:
        fu = pd.read_csv(path).drop_duplicates(ID_COL).set_index(ID_COL)

        shared = sorted(set(fu.index) & set(expr.index))
        print(f"\n  {label}")
        print(f"    drug data          : {len(fu):,} lines")
        print(f"    with expression    : {len(shared):,} lines "
              f"({len(fu) - len(shared):,} lost)")

        y = fu.loc[shared, DRUG_COLS].join(models, how="left")

        # Flag extrapolated IC50s (Decision D2) once, here, so downstream code
        # never has to recompute it: LN_IC50 above ln(MAX_CONC) means the curve
        # fit extrapolated past the highest dose actually tested.
        y["ic50_extrapolated"] = y["LN_IC50"] > np.log(y["MAX_CONC"])

        X = expr.loc[shared]
        assert (X.index == y.index).all(), "X and y row order diverged"

        print(f"    COREAD             : {int((y.TCGA_DESC == CRC_CODE).sum()):,}")
        print(f"    AUC                : mean {y.AUC.mean():.3f}  sd {y.AUC.std():.3f}")
        print(f"    IC50 extrapolated  : {int(y.ic50_extrapolated.sum()):,} "
              f"({100 * y.ic50_extrapolated.mean():.1f}%)")

        X.to_parquet(PROCESSED / f"X_{label}.parquet")
        y.to_parquet(PROCESSED / f"y_{label}.parquet")
        written[label] = (X.shape, y.shape)
        print(f"    wrote X_{label}.parquet {X.shape}  y_{label}.parquet {y.shape}")

    banner("6. TRAIN / TEST DESIGN  (Decision D3)")
    y1 = pd.read_parquet(PROCESSED / "y_GDSC1.parquet")
    y2 = pd.read_parquet(PROCESSED / "y_GDSC2.parquet")
    print(f"  GDSC1 {len(y1):,} lines (train)   GDSC2 {len(y2):,} lines (test)")
    print(f"  in both: {len(set(y1.index) & set(y2.index)):,}")
    print()
    print("  Because most lines appear in both screens, GDSC2 is not a set of")
    print("  unseen cell lines -- it is an independent RE-MEASUREMENT of largely")
    print("  the same ones. That still tests something real: whether the model")
    print("  predicts signal that reproduces across experiments rather than noise")
    print("  specific to one screen. Describe it that way and it is strong")
    print("  validation; call it 'held-out cell lines' and it would be wrong.")
    print("  For unseen-line validation, hold out GDSC1 lines before training.")

    banner("DONE")
    print("  Next: script 03 -- baselines before any model.")
    print("  mean / lineage / proliferation / permuted-label (Decision D5).")


if __name__ == "__main__":
    main()
