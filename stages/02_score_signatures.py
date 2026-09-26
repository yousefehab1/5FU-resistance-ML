"""
stages/02_score_signatures.py
==============================

Scores every signature module (DTP, RSC, CBC, Fetal, MYC, CellCycle)
against both GDSC1 and GDSC2 expression, writing the per-screen score
tables every downstream stage reads.


INPUTS   data/processed/X_{GDSC1,GDSC2}.parquet
         data/raw/signatures/*.txt, data/raw/hgnc_alias_map.csv
OUTPUTS  data/processed/signature_scores_{GDSC1,GDSC2}.parquet
         data/processed/signature_scores_ssgsea_{GDSC1,GDSC2}.parquet
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config as C
from lib.io import load_screen
from lib.report import banner
from lib.signatures import load_alias_map, load_signatures, score_all


def main():
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

    banner("DONE")


if __name__ == "__main__":
    main()
