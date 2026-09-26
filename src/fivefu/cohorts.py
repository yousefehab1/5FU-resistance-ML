"""Score an external expression cohort with the project's signatures.

score_cohort() applies the curated signatures and the 413 model genes (as a
plain gene set) to any matrix. Low gene recovery or an unrecognised gene
namespace stops the run with sys.exit(1) rather than scoring a different gene
set under the same name. Used by scripts 16 and 17.
"""

from __future__ import annotations

import contextlib
import io
import re
import sys

import pandas as pd

from . import signatures
from .paths import processed_dir

MODEL_GENES_FILE = processed_dir() / "final_model_genes.csv"
RECOVERY_MIN = 80.0


_RECOVERY_LINE = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+genes\s+\|\s+(\d+)\s+matched\s+\(\s*([\d.]+)%\)",
    re.MULTILINE,
)


def _parse_recovery(console_text):
    """Pull {signature_name: recovery_pct} out of load_signatures' own
    verbose output, rather than recomputing recovery with separate logic
    that could silently drift from what fivefu.signatures actually did."""
    return {m.group(1): float(m.group(4)) for m in _RECOVERY_LINE.finditer(console_text)}


def detect_gene_axis(expr, alias_map, verbose=True):
    """
    Auto-detect whether genes sit on rows or columns, and whether the
    identifiers are a namespace this project can score at all.

    Returns expr reoriented to samples (rows) x genes (columns), the
    orientation fivefu.signatures expects.
    """
    reference = set(alias_map.keys()) | set(alias_map.values())
    col_ids = [str(c) for c in expr.columns]
    row_ids = [str(i) for i in expr.index]
    col_hits = len(set(col_ids) & reference)
    row_hits = len(set(row_ids) & reference)

    if col_hits == 0 and row_hits == 0:
        def looks_ensembl(ids):
            sample = ids[:200]
            return sample and sum(str(x).startswith("ENSG") for x in sample) > 0.5 * len(sample)

        if looks_ensembl(col_ids) or looks_ensembl(row_ids):
            print("  !! Expression matrix is keyed on Ensembl IDs, not HGNC symbols.")
            print("     There is deliberately no Ensembl->symbol mapper: a mapping layer")
            print("     is where genes disappear silently. Convert at source and supply")
            print("     a matrix with an HGNC gene symbol axis.")
        else:
            print("  !! Neither axis of the expression matrix matches any known HGNC")
            print("     symbol or alias (checked against data/raw/hgnc_alias_map.csv).")
            print("     Cannot auto-detect orientation or namespace. Stopping.")
        sys.exit(1)

    genes_on = "columns" if col_hits >= row_hits else "rows"
    if verbose:
        print(f"  auto-detected: genes on {genes_on} "
              f"({max(col_hits, row_hits):,} of {len(col_ids if genes_on == 'columns' else row_ids):,} "
              f"identifiers matched a known HGNC symbol or alias)")
    return expr if genes_on == "columns" else expr.T


def load_model_derived_genes(path=MODEL_GENES_FILE):
    df = pd.read_csv(path)
    return df["gene"].astype(str).tolist()


def score_cohort(expr, metadata=None, platform=None, alias_map=None,
                  min_recovery=RECOVERY_MIN, verbose=True):
    """
    Score one external expression matrix against the project's curated
    signatures plus the model-derived gene list.

    expr: DataFrame, genes x samples OR samples x genes (auto-detected).
          Gene axis must be HGNC symbols (current or superseded --
          resolved via the alias map); Ensembl IDs are rejected (see
          detect_gene_axis).
    metadata: optional DataFrame indexed like expr's sample axis, joined
              onto the returned score table.
    platform: optional label string, added as a constant column so
              scores from several cohorts can be concatenated and
              traced back to their source.
    min_recovery: stop threshold in percent, default 80.0. Real
                  cohorts should keep the default. Pass None only for
                  diagnostics that need to inspect a low-recovery result
                  (script 16's regression check does this once).

    Returns (rank_df, ss_df, recovery): rank_df is the primary
    background_score-based table (sample x [CBC, CellCycle, DTP, Fetal,
    MYC, RSC, ModelDerived] plus metadata/platform if given); ss_df is
    the ssGSEA sensitivity-check equivalent (D13); recovery is
    {signature_name: pct} for all 8 inputs (7 raw signatures + the
    model-derived list), the numbers the 80% threshold is checked against.
    """
    if alias_map is None:
        alias_map = signatures.load_alias_map()

    expr = detect_gene_axis(expr, alias_map, verbose=verbose)
    universe = set(expr.columns.astype(str))

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        sigs = signatures.load_signatures(universe, alias_map=alias_map)
    console = buf.getvalue()
    if verbose:
        print(console)
    recovery = _parse_recovery(console)

    model_genes = load_model_derived_genes()
    matched, n_direct, n_alias, n_unres = signatures.resolve_symbols(model_genes, universe, alias_map)
    pct_model = 100 * len(matched) / len(model_genes) if model_genes else 0.0
    if verbose:
        flag = "" if pct_model >= RECOVERY_MIN else "   <-- LOW RECOVERY"
        print(f"  {'ModelDerived':12} {len(model_genes):>4} genes | {len(matched):>4} matched "
              f"({pct_model:5.1f}%){flag}  (from final_model_genes.csv, coefficients discarded)")
        print(f"               {n_direct} direct, {n_alias} via HGNC synonym, {n_unres} unresolved")
    sigs["ModelDerived"] = matched
    recovery["ModelDerived"] = pct_model

    if min_recovery is not None:
        failed = {name: pct for name, pct in recovery.items() if pct < min_recovery}
        if failed:
            print(f"\n  !! Signature recovery below {min_recovery:.0f}%, stopping: {failed}")
            sys.exit(1)

    rank_df, ss_df = signatures.score_all(expr, sigs, verbose=verbose)

    if metadata is not None:
        rank_df = rank_df.join(metadata)
        ss_df = ss_df.join(metadata)
    if platform is not None:
        rank_df["platform"] = platform
        ss_df["platform"] = platform

    return rank_df, ss_df, recovery
