"""
Builds the two tables every later stage reads.

1. Finds 5-FU in GDSC1/GDSC2 and joins it to expression and cell-line
   metadata: X_{label}.parquet (expression) and y_{label}.parquet
   (AUC/LN_IC50 + metadata), for label in {GDSC1, GDSC2}.
2. Scores every signature module (DTP, RSC, CBC, Fetal, MYC, CellCycle)
   against both screens.

INPUTS   data/raw/{GDSC1,GDSC2}_fitted_dose_response_24Jul22.csv
         data/raw/rnaseq_all_20260323.csv   (5.7GB, streamed -- see lib.io)
         data/raw/model_list_20260724.csv
         data/raw/signatures/*.txt, data/raw/hgnc_alias_map.csv
OUTPUTS  data/processed/X_{GDSC1,GDSC2}.parquet
         data/processed/y_{GDSC1,GDSC2}.parquet
         data/processed/expression_tpm.parquet   (cache; delete to rebuild)
         data/processed/signature_scores_{GDSC1,GDSC2}.parquet
         data/processed/signature_scores_ssgsea_{GDSC1,GDSC2}.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

import config as C
from lib.io import load_expression_cached, load_screen
from lib.report import banner
from lib.signatures import load_alias_map, load_signatures, score_all

DRUG_SEARCH = "fluorouracil"


def find_drug_rows(df, label):
    mask = df["DRUG_NAME"].astype(str).str.contains(DRUG_SEARCH, case=False, na=False)
    hits = df[mask]
    print(f"  {label}: {len(hits):,} rows matching '{DRUG_SEARCH}'  "
          f"({hits['SANGER_MODEL_ID'].nunique():,} cell lines)")
    return hits


def report_gdsc_agreement(fu1, fu2):
    """
    Pearson r between GDSC1 and GDSC2 LN_IC50 on shared cell lines: two
    independent screens of the same drug on the same lines. However well
    they agree with EACH OTHER is roughly the ceiling on how well any model
    could predict sensitivity from expression -- see config.CEILING_R.
    """
    a = fu1.groupby("SANGER_MODEL_ID")["LN_IC50"].mean()
    b = fu2.groupby("SANGER_MODEL_ID")["LN_IC50"].mean()
    both = pd.concat([a, b], axis=1, join="inner", keys=["GDSC1", "GDSC2"]).dropna()
    r, p = stats.pearsonr(both["GDSC1"], both["GDSC2"])
    print(f"  GDSC1 vs GDSC2 LN_IC50, {len(both):,} shared lines: r={r:.3f} (p={p:.2e})")
    print(f"  (config.CEILING_R['LN_IC50']={C.CEILING_R['LN_IC50']} -- judge every "
          f"model result as a fraction of this, not of 1.0)")


def build_modelling_table():
    banner("1. LOADING GDSC1 / GDSC2 AND FINDING 5-FU")
    g1 = pd.read_csv(C.GDSC1_FILE, low_memory=False)
    g2 = pd.read_csv(C.GDSC2_FILE, low_memory=False)
    fu1, fu2 = find_drug_rows(g1, "GDSC1"), find_drug_rows(g2, "GDSC2")
    report_gdsc_agreement(fu1, fu2)

    banner("2. WHICH CELL LINES DO WE NEED?")
    needed_ids = set(fu1["SANGER_MODEL_ID"]) | set(fu2["SANGER_MODEL_ID"])
    print(f"  union: {len(needed_ids):,} distinct cell lines")

    banner("3. EXPRESSION")
    expr = load_expression_cached(needed_ids)
    v = expr.to_numpy()
    print(f"  value range: {np.nanmin(v):.3f} to {np.nanmax(v):.1f}  "
          f"(median {np.nanmedian(v):.3f}) -- log2(TPM+1) per Cell Model "
          f"Passports; NOT re-logged here")

    banner("4. MODEL METADATA")
    models = pd.read_csv(C.MODEL_LIST_FILE, low_memory=False)
    models = (models.rename(columns={"model_id": C.ID_COL})
              [[C.ID_COL] + C.MODEL_META]
              .drop_duplicates(C.ID_COL)
              .set_index(C.ID_COL))

    banner("5. JOINING")
    C.PROCESSED.mkdir(parents=True, exist_ok=True)
    for label, fu in [("GDSC1", fu1), ("GDSC2", fu2)]:
        f = fu.drop_duplicates("SANGER_MODEL_ID").set_index("SANGER_MODEL_ID")
        shared = sorted(set(f.index) & set(expr.index))
        print(f"\n  {label}: {len(f):,} lines with drug data, "
              f"{len(shared):,} with expression ({len(f) - len(shared):,} lost)")

        y = f.loc[shared, C.DRUG_COLS].join(models, how="left")
        # LN_IC50 above ln(MAX_CONC) means the curve fit extrapolated past
        # the highest dose actually tested -- the line survived every dose
        # screened, so the "IC50" is not really measured.
        y["ic50_extrapolated"] = y["LN_IC50"] > np.log(y["MAX_CONC"])

        X = expr.loc[shared]
        assert (X.index == y.index).all(), "X and y row order diverged"

        print(f"    COREAD: {int((y[C.LINEAGE_COL] == C.CRC).sum()):,}   "
              f"AUC mean {y.AUC.mean():.3f} sd {y.AUC.std():.3f}   "
              f"extrapolated {100 * y.ic50_extrapolated.mean():.1f}%")

        X.to_parquet(C.PROCESSED / f"X_{label}.parquet")
        y.to_parquet(C.PROCESSED / f"y_{label}.parquet")
        print(f"    wrote X_{label}.parquet {X.shape}  y_{label}.parquet {y.shape}")


def score_signatures():
    alias_map = load_alias_map()
    print(f"  HGNC alias map: {len(alias_map):,} entries")

    for label in [C.TRAIN, C.TEST]:
        banner(f"SCORING SIGNATURES -- {label}")
        X, _ = load_screen(label, with_signatures=False)
        print(f"  {label}: {X.shape[0]:,} lines x {X.shape[1]:,} genes")

        sigs = load_signatures(X.columns, alias_map=alias_map)
        rank_df, ss_df = score_all(X, sigs)

        if {"RSC", "CBC"} <= set(rank_df.columns):
            r_rc = rank_df["RSC"].corr(rank_df["CBC"])
            print(f"\n  RSC vs CBC: r = {r_rc:+.3f}  "
                  f"({'FAILS - should be negative' if r_rc > 0 else 'anti-correlated, control passes'})")

        rank_df.to_parquet(C.PROCESSED / f"signature_scores_{label}.parquet")
        ss_df.to_parquet(C.PROCESSED / f"signature_scores_ssgsea_{label}.parquet")
        print(f"  wrote signature_scores_{label}.parquet {rank_df.shape}")
        print(f"  wrote signature_scores_ssgsea_{label}.parquet {ss_df.shape}")


def main():
    build_modelling_table()
    score_signatures()
    banner("DONE")


if __name__ == "__main__":
    main()
