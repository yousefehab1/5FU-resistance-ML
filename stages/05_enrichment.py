"""
stages/05_enrichment.py
=========================

Is the final de-confounded model's gene list enriched for any curated
signature, more than chance predicts?

This is NEW code -- the original project only ever printed the raw
gene-name overlap between the model's genes and each signature, with no
statistical test of whether that overlap exceeds chance. Its headline
"DTP_up enrichment 4.7x, p=1.1e-6" number exists nowhere as computed,
persisted output before this; only as a hand-typed dashboard literal.

INPUTS   data/processed/final_model_genes.csv     (stage 04)
         data/processed/X_GDSC1.parquet            (gene universe)
         data/raw/signatures/*.txt
OUTPUTS  data/processed/enrichment_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd

import config as C
from lib.io import exclude_haem, load_screen
from lib.report import banner, write_and_report
from lib.signatures import load_alias_map, load_signatures
from lib.stats import hypergeometric_enrichment


def main():
    banner("1. LOADING MODEL GENES AND SIGNATURES")
    genes = pd.read_csv(C.PROCESSED / "final_model_genes.csv")

    # The universe must be the gene set the model actually chose from --
    # i.e. AFTER the missingness filter and haem exclusion stage 04 applies,
    # not the raw unfiltered expression matrix. Getting this wrong changes
    # every expected-overlap and p-value below.
    X, y, _ = load_screen(C.TRAIN)
    _, _, X = exclude_haem(y, X)
    universe = X.columns
    sigs = load_signatures(universe, alias_map=load_alias_map())

    query = set(genes["gene"])
    print(f"  model genes: {len(query):,}   universe: {len(universe):,}")

    banner("2. HYPERGEOMETRIC ENRICHMENT")
    rows = []
    for name, sig_genes in sigs.items():
        r = hypergeometric_enrichment(query, sig_genes, universe, name=name)
        rows.append(r)
        flag = "  <-- p<0.05" if r["p_value"] < 0.05 else ""
        print(f"  {name:<10} {r['observed_overlap']:>3}/{r['signature_size']:<4} "
              f"(expected {r['expected_overlap']:.2f})  "
              f"fold={r['fold_enrichment']:.2f}x  p={r['p_value']:.2e}{flag}")

    write_and_report(pd.DataFrame(rows), C.PROCESSED / "enrichment_results.csv",
                      "enrichment_results.csv")
    banner("DONE")


if __name__ == "__main__":
    main()
