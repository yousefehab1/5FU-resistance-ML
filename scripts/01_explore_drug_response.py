"""Extract 5-FU from GDSC1 and GDSC2 and measure how well the two screens agree.

That agreement (r = 0.60 on AUC) is the ceiling any expression model is judged
against, so it is set before any modelling.

Inputs:  data/raw/GDSC{1,2}_fitted_dose_response_24Jul22.csv
Outputs: data/processed/fu_gdsc{1,2}_raw.csv
Run:     python scripts/01_explore_drug_response.py
"""

# ----------------------------------------------------------------------------
# IMPORTS
# ----------------------------------------------------------------------------
# `import x as y` gives a module a short nickname. `pd` and `np` are the
# universal conventions -- essentially everyone writes them this way.

import sys                    # used below to put src/ on the import path
from pathlib import Path      # modern way to handle file paths; avoids "/" string-joining

import numpy as np            # numeric arrays
import pandas as pd           # tables ("DataFrames") -- the R data.frame equivalent
from scipy import stats       # we need pearsonr / spearmanr for the agreement check


# ----------------------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------------------
# Keeping paths and constants at the top means you change them in one place.

# Path(__file__) is this script's own location. `.resolve()` makes it absolute,
# `.parent` goes up one level. So: scripts/01_...py -> scripts/ -> project root,
# and src/ under it goes on sys.path so the fivefu package imports without
# being installed. The root itself then comes from fivefu.paths, which honours
# FIVEFU_ROOT: a run pointed at a scratch copy reads AND writes that copy.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

PROJECT_ROOT = project_root()

RAW = PROJECT_ROOT / "data" / "raw"              # the "/" operator joins paths
PROCESSED = PROJECT_ROOT / "data" / "processed"

GDSC1_FILE = RAW / "GDSC1_fitted_dose_response_24Jul22.csv"
GDSC2_FILE = RAW / "GDSC2_fitted_dose_response_24Jul22.csv"

# We search case-insensitively for this substring rather than hard-coding an
# exact drug name, because we don't yet know whether GDSC calls it
# "Fluorouracil", "5-Fluorouracil", or "5-FU".
DRUG_SEARCH = "fluorouracil"

# GDSC's TCGA-style code for colorectal adenocarcinoma.
CRC_CODE = "COREAD"

# The cell line our own 5-FU time-course was generated in.
OUR_CELL_LINE = "HCT116"


# ----------------------------------------------------------------------------
# SMALL HELPERS
# ----------------------------------------------------------------------------

from fivefu.report import banner


def load_gdsc(path, label):
    """
    Load one GDSC fitted-dose-response CSV and report what's in it.

    A 'docstring' like this one is the triple-quoted text at the top of a
    function. Python stores it, and it's the standard way to document code.
    """
    if not path.exists():
        # Fail loudly and usefully rather than crashing with a confusing error.
        print(f"  !! MISSING: {path}")
        print(f"     Download it first -- see SETUP_phase1.md Part 3.")
        return None

    # low_memory=False stops pandas guessing column types from the first chunk
    # only, which on big files can produce inconsistent types and a warning.
    df = pd.read_csv(path, low_memory=False)

    # An f-string is a string prefixed with f, where {expressions} get inserted.
    # df.shape is a (rows, columns) tuple; :, formats numbers with thousands commas.
    print(f"  {label}: {df.shape[0]:,} rows x {df.shape[1]} columns")

    return df


def find_drug_rows(df, label):
    """
    Pull out just the rows for our drug, checking the column exists first.

    Returns a DataFrame (possibly empty).
    """
    if "DRUG_NAME" not in df.columns:
        print(f"  !! {label} has no DRUG_NAME column. Columns are: {list(df.columns)}")
        return pd.DataFrame()

    # Reading this from the inside out:
    #   df["DRUG_NAME"]           -> one column (a "Series")
    #   .astype(str)              -> force to text, so missing values don't crash .str
    #   .str.contains(..., case=False, na=False)
    #                             -> True/False for each row; na=False treats
    #                                missing as "no match" instead of NaN
    # The result is a boolean mask, and df[mask] keeps only the True rows.
    # This is the pandas equivalent of R's df[grepl(...), ].
    mask = df["DRUG_NAME"].astype(str).str.contains(DRUG_SEARCH, case=False, na=False)
    hits = df[mask]

    print(f"  {label}: {len(hits):,} rows matching '{DRUG_SEARCH}'")

    if len(hits) > 0:
        # .unique() returns the distinct values in a column.
        print(f"    drug names found : {sorted(hits['DRUG_NAME'].unique())}")
        print(f"    DRUG_IDs found   : {sorted(hits['DRUG_ID'].unique())}")

    return hits


# ----------------------------------------------------------------------------
# MAIN
# ----------------------------------------------------------------------------

def main():

    # ------------------------------------------------------------------
    banner("1. LOADING RAW FILES")
    # ------------------------------------------------------------------
    g1 = load_gdsc(GDSC1_FILE, "GDSC1")
    g2 = load_gdsc(GDSC2_FILE, "GDSC2")

    if g2 is None:
        print("\nGDSC2 is required. Stopping.")
        return   # `return` exits the function early

    # Always look at the real columns before assuming anything about them.
    print("\n  GDSC2 columns:")
    for c in g2.columns:
        print(f"    - {c}")

    # ------------------------------------------------------------------
    banner("2. FINDING 5-FU")
    # ------------------------------------------------------------------
    fu2 = find_drug_rows(g2, "GDSC2")
    fu1 = find_drug_rows(g1, "GDSC1") if g1 is not None else pd.DataFrame()

    if len(fu2) == 0:
        print("\n  No match in GDSC2. Listing a sample of available drug names:")
        print(sorted(g2["DRUG_NAME"].astype(str).unique())[:60])
        return

    # ------------------------------------------------------------------
    banner("3. HOW MANY CELL LINES?")
    # ------------------------------------------------------------------
    # .nunique() counts distinct values -- "how many different cell lines".
    for label, fu in [("GDSC1", fu1), ("GDSC2", fu2)]:
        if len(fu) == 0:
            continue
        n_lines = fu["SANGER_MODEL_ID"].nunique()
        print(f"  {label}: {n_lines:,} distinct cell lines with 5-FU data")

        # If rows > lines, some lines were screened more than once
        # (multiple DRUG_IDs). Worth knowing about now.
        if len(fu) > n_lines:
            print(f"         ({len(fu):,} rows -> {len(fu) - n_lines:,} duplicate "
                  f"line/drug combinations, from multiple DRUG_IDs)")

    # ------------------------------------------------------------------
    banner("4. HOW MANY ARE COLORECTAL?")
    # ------------------------------------------------------------------
    # The cancer-type column has been named differently across releases,
    # so find whichever one is present rather than assuming.
    type_col = None
    for candidate in ["TCGA_DESC", "CANCER_TYPE", "TISSUE"]:
        if candidate in fu2.columns:
            type_col = candidate
            break

    if type_col is None:
        print("  !! No cancer-type column found. Will need Cell_Lines_Details.xlsx.")
    else:
        print(f"  Using column: {type_col}")

        for label, fu in [("GDSC1", fu1), ("GDSC2", fu2)]:
            if len(fu) == 0:
                continue
            crc = fu[fu[type_col] == CRC_CODE]
            print(f"  {label}: {crc['SANGER_MODEL_ID'].nunique():,} colorectal "
                  f"({CRC_CODE}) cell lines")

        # Show the biggest cancer types so we can see how COREAD compares.
        # .value_counts() tabulates frequencies; .head(12) takes the top 12.
        print(f"\n  Top cancer types in GDSC2 5-FU data:")
        counts = fu2.drop_duplicates("SANGER_MODEL_ID")[type_col].value_counts()
        for name, n in counts.head(12).items():
            marker = "  <-- ours" if name == CRC_CODE else ""
            print(f"    {str(name):<12} {n:>4}{marker}")

    # ------------------------------------------------------------------
    banner("5. IS HCT116 THERE?")
    # ------------------------------------------------------------------
    # Cell line names are written inconsistently (HCT116 / HCT-116 / HCT 116),
    # so strip out hyphens and spaces before comparing. This is exactly the
    # name-matching problem flagged in the framing document.
    for label, fu in [("GDSC1", fu1), ("GDSC2", fu2)]:
        if len(fu) == 0:
            continue
        # .str.replace with regex=True removes any hyphen or whitespace
        clean = fu["CELL_LINE_NAME"].astype(str).str.replace(r"[-\s]", "", regex=True)
        found = fu[clean.str.upper() == OUR_CELL_LINE.upper()]
        if len(found) > 0:
            for _, row in found.iterrows():     # iterrows gives (index, row) pairs
                print(f"  {label}: FOUND {row['CELL_LINE_NAME']} "
                      f"({row['SANGER_MODEL_ID']})  LN_IC50 = {row['LN_IC50']:.3f}")
        else:
            print(f"  {label}: HCT116 not found in the 5-FU rows")

    # ------------------------------------------------------------------
    banner("6. WHAT DOES LN_IC50 LOOK LIKE?")
    # ------------------------------------------------------------------
    # .describe() gives count/mean/std/min/quartiles/max in one go.
    print("  GDSC2 LN_IC50 across all lines:")
    print(fu2["LN_IC50"].describe().to_string())

    # Now the extrapolation check flagged in the setup notes.
    # LN_IC50 is the natural log of IC50 in micromolar, and MAX_CONC is the
    # highest concentration actually tested (in micromolar). So any row where
    # LN_IC50 > log(MAX_CONC) describes a curve whose IC50 was never actually
    # reached -- the cell line survived every dose tested, and the number is
    # extrapolated by the curve fit rather than measured.
    if "MAX_CONC" in fu2.columns:
        ln_max = np.log(fu2["MAX_CONC"])
        above = fu2["LN_IC50"] > ln_max
        pct = 100 * above.mean()      # mean of a True/False column = proportion True
        print(f"\n  Screened concentration range: "
              f"{fu2['MIN_CONC'].min():.4g} to {fu2['MAX_CONC'].max():.4g} uM")
        print(f"  Rows with LN_IC50 ABOVE the max tested dose: "
              f"{above.sum():,} of {len(fu2):,}  ({pct:.1f}%)")
        print("    -> these are extrapolated, not measured. If this percentage is")
        print("       large, that is itself a finding: most lines are simply not")
        print("       killed by 5-FU at screenable doses, which caps how much real")
        print("       signal there is to predict.")

    # ------------------------------------------------------------------
    banner("7. DO GDSC1 AND GDSC2 AGREE?  (the predictability ceiling)")
    # ------------------------------------------------------------------
    if len(fu1) > 0 and len(fu2) > 0:
        # Collapse to one value per cell line. Where a line has several rows
        # (multiple DRUG_IDs) take the mean -- a defensible simple choice we
        # can revisit later.
        # groupby(...)[col].mean() is R's aggregate() / dplyr's group_by+summarise.
        a = fu1.groupby("SANGER_MODEL_ID")["LN_IC50"].mean()
        b = fu2.groupby("SANGER_MODEL_ID")["LN_IC50"].mean()

        # pd.concat with join="inner" keeps only IDs present in BOTH.
        both = pd.concat([a, b], axis=1, join="inner", keys=["GDSC1", "GDSC2"])
        both = both.dropna()          # drop any row with a missing value

        print(f"  Cell lines screened in both: {len(both):,}")

        if len(both) >= 10:
            r, p_r = stats.pearsonr(both["GDSC1"], both["GDSC2"])
            rho, p_s = stats.spearmanr(both["GDSC1"], both["GDSC2"])
            print(f"  Pearson  r   = {r:.3f}   (p = {p_r:.2e})")
            print(f"  Spearman rho = {rho:.3f}   (p = {p_s:.2e})")
            print()
            print("  >>> THIS IS THE NUMBER TO REMEMBER. <<<")
            print("  Two independent experimental screens of the same drug on the")
            print("  same cell lines agree only this well. No model predicting")
            print("  IC50 from expression can reasonably be expected to beat it,")
            print("  because the measurement itself is this noisy.")
            print(f"  Judge every model result later as a fraction of r = {r:.2f},")
            print("  not against a mental benchmark of 1.0.")
        else:
            print("  Too few shared lines to compute a meaningful correlation.")
    else:
        print("  Need both GDSC1 and GDSC2 for this comparison.")

    # ------------------------------------------------------------------
    banner("8. SAVING")
    # ------------------------------------------------------------------
    # mkdir with parents=True creates intermediate folders;
    # exist_ok=True means "don't error if it's already there".
    PROCESSED.mkdir(parents=True, exist_ok=True)

    out = PROCESSED / "fu_gdsc2_raw.csv"
    fu2.to_csv(out, index=False)     # index=False stops pandas writing row numbers
    print(f"  Wrote GDSC2 5-FU rows -> {out}")

    if len(fu1) > 0:
        out1 = PROCESSED / "fu_gdsc1_raw.csv"
        fu1.to_csv(out1, index=False)
        print(f"  Wrote GDSC1 5-FU rows -> {out1}")

    banner("DONE -- paste this output back to Claude")


# This guard means the code only runs when you execute the file directly,
# not when another script imports it. Standard Python boilerplate.
if __name__ == "__main__":
    main()
