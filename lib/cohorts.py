"""
lib/cohorts.py
==============

Score an external expression cohort with the project's signatures.

score_cohort() applies the curated signatures and the model genes (as a
plain gene set) to any matrix. Low gene recovery or an unrecognised gene
namespace stops the run with sys.exit(1) rather than scoring a different gene
set under the same name. Used by stages/07_clinical_validation.py and tests/test_score_cohort.py.

load_clinical() reads the cohorts stages/prep/clinical_prep.R writes, and
AdjustedLogitZ gives the Wald z of a gene-set score in one adjusted
response model with the design built once, so many gene sets (random ones
for stage 07's null, sub-programmes for stage 11) cost one fit each.
"""

import contextlib
import io
import re
import sys
import warnings

import numpy as np
import pandas as pd
import patsy
from patsy.builtins import C as patsy_C
import statsmodels.api as sm

import config as C
from lib import signatures

MODEL_GENES_FILE = C.PROCESSED / "final_model_genes.csv"
RECOVERY_MIN = 80.0

# Covariates of the adjusted response models (stages/07_clinical_validation.py).
GPL570_ADJUST = "+ purity + C(cms) + C(msi_ntp) + C(study)"
GPL6480_ADJUST = "+ C(cms) + C(msi_ntp)"
CLINICAL_FILES = {"gpl570": "clinical_expr_gpl570_combat.csv",
                  "gpl570_precombat": "clinical_expr_gpl570_precombat.csv",
                  "gpl6480": "clinical_expr_gse104645.csv",
                  "covariates": "clinical_covariates.csv"}


def load_clinical():
    """{name: table} for CLINICAL_FILES (expression genes x samples,
    covariates indexed by sample_id), or None if any file is missing."""
    paths = {k: C.PROCESSED / f for k, f in CLINICAL_FILES.items()}
    if not all(p.exists() for p in paths.values()):
        return None
    return {k: pd.read_csv(p, index_col="sample_id" if k == "covariates" else "gene")
            for k, p in paths.items()}


def zscore(s):
    return (s - s.mean()) / s.std(ddof=0)


class AdjustedLogitZ:
    """
    Wald z of a gene-set score in `response ~ z <adjust>`, fitted on the
    cohort's rows of `cov_df` (dropping rows without purity when the model
    adjusts for it). Call it with [genes] or [up_genes, down_genes]: the set
    is scored with background_score on `ranks` (up minus down for two),
    z-scored over the cohort and refitted. The design is built once and only
    the score column changes. Returns NaN when the fit fails or does not
    converge.
    """

    def __init__(self, ranks, cov_df, adjust):
        self.ranks, self.index = ranks, cov_df.index
        d = cov_df.copy()
        d["response"] = d["response"].astype(int)
        d["z"] = 0.0
        if "purity" in adjust:
            d = d.dropna(subset=["purity"])
        # An explicit namespace: this module imports config as C, which would
        # otherwise shadow patsy's C() in the formula.
        yv, X = patsy.dmatrices(f"response ~ z {adjust}", d, return_type="dataframe",
                                eval_env=patsy.EvalEnvironment([{"C": patsy_C}]))
        self.j, self.rows = X.columns.get_loc("z"), X.index
        self.X, self.y = np.array(X, dtype=float), yv.to_numpy().ravel()

    def score(self, sets):
        sc = signatures.background_score(self.ranks, sets[0])
        if len(sets) == 2:
            sc = sc - signatures.background_score(self.ranks, sets[1])
        return zscore(pd.Series(sc, index=self.ranks.index).loc[self.index])

    def __call__(self, sets):
        self.X[:, self.j] = self.score(sets).loc[self.rows].to_numpy()
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                m = sm.Logit(self.y, self.X).fit(disp=0)
        except Exception:
            return np.nan
        return m.tvalues[self.j] if m.mle_retvals.get("converged", True) else np.nan


_RECOVERY_LINE = re.compile(
    r"^\s*(\S+)\s+(\d+)\s+genes\s+\|\s+(\d+)\s+matched\s+\(\s*([\d.]+)%\)",
    re.MULTILINE,
)


def _parse_recovery(console_text):
    """Pull {signature_name: recovery_pct} out of load_signatures' own
    verbose output, rather than recomputing recovery with separate logic
    that could silently drift from what lib.signatures actually did."""
    return {m.group(1): float(m.group(4)) for m in _RECOVERY_LINE.finditer(console_text)}


def detect_gene_axis(expr, alias_map, verbose=True):
    """
    Auto-detect whether genes sit on rows or columns, and whether the
    identifiers are a namespace this project can score at all.

    Returns expr reoriented to samples (rows) x genes (columns), the
    orientation lib.signatures expects.
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


def resolve_cohort(expr, alias_map=None, verbose=True):
    """
    (expr, sigs, recovery) for one external cohort: expr turned to samples x
    HGNC genes, every signature plus the model-derived list resolved against
    its genes, and the percent of each recovered. score_cohort scores these;
    stages/07_clinical_validation.py also needs the resolved sizes for its
    size-matched random-signature null.
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

    return expr, sigs, recovery


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
                  (tests/test_score_cohort.py does this once).

    Returns (rank_df, ss_df, recovery): rank_df is the primary
    background_score-based table (sample x [CBC, CellCycle, DTP, Fetal,
    MYC, RSC, ModelDerived] plus metadata/platform if given); ss_df is
    the ssGSEA sensitivity-check equivalent; recovery is
    {signature_name: pct} for all 8 inputs (7 raw signatures + the
    model-derived list), the numbers the 80% threshold is checked against.
    """
    expr, sigs, recovery = resolve_cohort(expr, alias_map, verbose)

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
