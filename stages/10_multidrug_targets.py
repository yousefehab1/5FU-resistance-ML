"""Build and QC response targets for oxaliplatin, SN-38, irinotecan and cisplatin.

Stops if a drug has fewer than min_n_stop lines (config.MIN_N_STOP).

Inputs:  data/raw/GDSC{1,2}_fitted_dose_response_24Jul22.csv,
         screened_compounds_rel_8.4.csv, model_list_20260724.csv
Outputs: data/processed/targets/<drug>_<screen>.parquet, multidrug_qc.csv,
         reports/10_multidrug_targets.md
Run:     python stages/10_multidrug_targets.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config as C

ROOT = C.PROJECT_ROOT
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
DOCS = ROOT / "data" / "processed" / "reports"  # generated reports
TARGETS_DIR = PROC / "targets"

GDSC1_FILE = RAW / "GDSC1_fitted_dose_response_24Jul22.csv"
GDSC2_FILE = RAW / "GDSC2_fitted_dose_response_24Jul22.csv"
COMPOUNDS_FILE = RAW / "screened_compounds_rel_8.4.csv"
MODELS_FILE = RAW / "model_list_20260724.csv"

ID_COL = "SANGER_MODEL_ID"

# Analysis parameters live in config.py; nothing here re-declares one.
from config import (DRUGS, HAEM, CEILING_AUC, MIN_N_TASK,
                    MIN_N_STOP)
from lib.report import banner

SCREENS = {"GDSC1": GDSC1_FILE, "GDSC2": GDSC2_FILE}


DRUG_COLS = ["SANGER_MODEL_ID", "CELL_LINE_NAME", "TCGA_DESC", "DRUG_ID", "DRUG_NAME",
             "LN_IC50", "AUC", "RMSE", "Z_SCORE", "MIN_CONC", "MAX_CONC"]
MODEL_META = ["cancer_type", "tissue", "cancer_type_detail", "growth_properties",
              "msi_status", "mutational_burden", "model_name", "gender", "age_at_sampling"]


# ============================================================================
# LOADING
# ============================================================================

def read_response_csv(path, label):
    if not path.exists():
        print(f"  !! MISSING: {path}")
        sys.exit(1)
    d = pd.read_csv(path, low_memory=False, usecols=DRUG_COLS)
    print(f"  {label}: {len(d):,} rows, {d.DRUG_NAME.nunique()} drugs, "
          f"{d[ID_COL].nunique():,} cell lines")
    return d


def check_multi_id_drugs(screens, compounds):
    """
    Stop-and-report check before deduplicating: for every drug with >1 DRUG_ID in any
    screen, confirm the IDs share DRUG_NAME, TARGET and TARGET_PATHWAY in
    screened_compounds_rel_8.4.csv. If they don't, they are not simply
    re-screens and must not be silently averaged.
    """
    for label, d in screens.items():
        for drug in DRUGS:
            ids = sorted(d.loc[d.DRUG_NAME == drug, "DRUG_ID"].unique())
            if len(ids) <= 1:
                continue
            rows = compounds[compounds.DRUG_ID.isin(ids)]
            targets = rows[["TARGET", "TARGET_PATHWAY"]].drop_duplicates()
            names = rows["DRUG_NAME"].unique()
            ok = len(targets) <= 1 and len(names) <= 1
            flag = "" if ok else "   <-- MISMATCH, do not average blindly"
            print(f"  {label} {drug:16} DRUG_IDs={ids}  "
                  f"target(s)={targets.TARGET.tolist()}{flag}")
            if not ok:
                print(f"     !! {drug} DRUG_IDs disagree on name/target/pathway. "
                      f"Stopping (expected column/consistency missing).")
                sys.exit(1)


# ============================================================================
# PER-DRUG EXTRACTION
# ============================================================================

def extract_drug(d, drug, screen_label):
    """
    One drug from one screen: drop haem lineages, average across DRUG_IDs
    (checked safe above), return a one-row-per-cell-line table plus a QC dict.
    Returns (None, None) if the drug is absent from this screen.
    """
    sub = d[d.DRUG_NAME == drug].copy()
    if sub.empty:
        return None, None

    ids = sorted(int(i) for i in sub.DRUG_ID.unique())
    id_agreement = None
    if len(ids) > 1:
        wide = sub.pivot_table(index=ID_COL, columns="DRUG_ID", values="AUC")
        shared = wide.dropna()
        if len(shared) >= 10 and len(ids) == 2:
            id_agreement = stats.pearsonr(shared[ids[0]], shared[ids[1]])[0]

    sub = sub[~sub.TCGA_DESC.isin(HAEM)]
    n_rows, n_lines = len(sub), sub[ID_COL].nunique()

    agg = sub.groupby(ID_COL).agg(
        CELL_LINE_NAME=("CELL_LINE_NAME", "first"),
        TCGA_DESC=("TCGA_DESC", "first"),
        LN_IC50=("LN_IC50", "mean"),
        AUC=("AUC", "mean"),
        RMSE=("RMSE", "mean"),
        Z_SCORE=("Z_SCORE", "mean"),
        MIN_CONC=("MIN_CONC", "mean"),
        MAX_CONC=("MAX_CONC", "mean"),
    )
    agg["ic50_extrapolated"] = agg.LN_IC50 > np.log(agg.MAX_CONC)

    qc = dict(
        drug=drug, screen=screen_label, drug_ids=ids, id_agreement_r=id_agreement,
        n_rows_before_dedup=n_rows, n=n_lines,
        auc_mean=agg.AUC.mean(), auc_sd=agg.AUC.std(),
        pct_extrapolated=100 * agg.ic50_extrapolated.mean(),
        pct_auc_gt_090=100 * (agg.AUC > 0.90).mean(),
        pct_auc_gt_095=100 * (agg.AUC > CEILING_AUC).mean(),
        pct_auc_gt_098=100 * (agg.AUC > 0.98).mean(),
        conc_min=agg.MIN_CONC.min(), conc_max=agg.MAX_CONC.max(),
    )
    return agg, qc


# ============================================================================
# MAIN
# ============================================================================

def main():

    banner("1. LOADING GDSC1 AND GDSC2")
    screens = {label: read_response_csv(path, label) for label, path in SCREENS.items()}
    compounds = pd.read_csv(COMPOUNDS_FILE, low_memory=False)

    banner("2. CHECKING MULTI-DRUG_ID DRUGS BEFORE DEDUPLICATING")
    check_multi_id_drugs(screens, compounds)

    banner("3. PER-DRUG x SCREEN EXTRACTION AND QC")
    TARGETS_DIR.mkdir(parents=True, exist_ok=True)
    models = (pd.read_csv(MODELS_FILE, low_memory=False)
                .rename(columns={"model_id": ID_COL})[[ID_COL] + MODEL_META]
                .drop_duplicates(ID_COL).set_index(ID_COL))

    tables, qc_rows = {}, []
    for drug in DRUGS:
        for screen_label, d in screens.items():
            agg, qc = extract_drug(d, drug, screen_label)
            if agg is None:
                print(f"  {drug:16} {screen_label}: absent from this screen")
                continue
            flag = "" if qc["n"] >= MIN_N_TASK else "  <-- below n=300"
            agree = (f"  DRUG_ID agreement r={qc['id_agreement_r']:.3f}"
                     if qc["id_agreement_r"] is not None else "")
            print(f"  {drug:16} {screen_label}  n={qc['n']:>4}  "
                  f"AUC mean={qc['auc_mean']:.3f} sd={qc['auc_sd']:.3f}  "
                  f"extrap={qc['pct_extrapolated']:.1f}%  "
                  f"ceiling(>{CEILING_AUC})={qc['pct_auc_gt_095']:.1f}%"
                  f"{agree}{flag}")
            if qc["n"] < MIN_N_STOP:
                print(f"  !! {drug} {screen_label}: n={qc['n']} < {MIN_N_STOP}. "
                      f"Stopping.")
                sys.exit(1)

            joined = agg.join(models, how="left")
            out_path = TARGETS_DIR / f"{drug.replace(' ', '_')}_{screen_label}.parquet"
            joined.to_parquet(out_path)
            tables[(drug, screen_label)] = agg
            qc_rows.append(qc)

    banner("4. MEASUREMENT CEILING (GDSC1 vs GDSC2 on shared lines, AUC)")
    ceilings = {}
    for drug in DRUGS:
        a, b = tables.get((drug, "GDSC1")), tables.get((drug, "GDSC2"))
        if a is None or b is None:
            ceilings[drug] = None
            print(f"  {drug:16} not computable (screened in one arm only)")
            continue
        both = pd.concat([a.AUC, b.AUC], axis=1, keys=["GDSC1", "GDSC2"]).dropna()
        if len(both) < 10:
            ceilings[drug] = None
            print(f"  {drug:16} not computable (only {len(both)} shared lines)")
            continue
        r, p = stats.pearsonr(both.GDSC1, both.GDSC2)
        ceilings[drug] = r
        print(f"  {drug:16} r={r:.3f}  (n={len(both):,} shared lines, p={p:.1e})")

    banner("5. SCREEN CHOICE PER DRUG (dynamic range = AUC SD)")
    chosen = {}
    for drug in DRUGS:
        options = {lbl: tables[(drug, lbl)] for lbl in SCREENS
                   if (drug, lbl) in tables}
        if len(options) == 1:
            lbl = next(iter(options))
            chosen[drug] = (lbl, f"only screen available (AUC sd={options[lbl].AUC.std():.3f})")
        else:
            sds = {lbl: float(t.AUC.std()) for lbl, t in options.items()}
            best = max(sds, key=sds.get)
            others = ", ".join(f"{k}={v:.3f}" for k, v in sds.items() if k != best)
            chosen[drug] = (best, f"higher AUC sd ({sds[best]:.3f} vs {others})")
        print(f"  {drug:16} -> {chosen[drug][0]:8} ({chosen[drug][1]})")

    banner("6. IRINOTECAN vs SN-38 (prodrug / active-metabolite divergence check)")
    iri, sn38 = tables.get(("Irinotecan", "GDSC2")), tables.get(("SN-38", "GDSC2"))
    iri_sn38_r = None
    if iri is not None and sn38 is not None:
        both = pd.concat([iri.AUC, sn38.AUC], axis=1, keys=["iri", "sn38"]).dropna()
        iri_sn38_r, p = stats.pearsonr(both.iri, both.sn38)
        print(f"  Irinotecan vs SN-38 AUC: r={iri_sn38_r:.3f}  (n={len(both):,}, p={p:.1e})")
        print(f"  {'Carried as separate drugs.' if iri_sn38_r < 0.7 else 'Track closely, as expected for a prodrug/metabolite pair.'}")
    else:
        print("  one or both unavailable")

    banner("7. WRITING QC TABLE AND FINDINGS")
    qc_df = pd.DataFrame(qc_rows)
    qc_df["ceiling_r"] = qc_df["drug"].map(
        lambda dr: ceilings[dr] if ceilings.get(dr) is not None else np.nan)
    qc_df["ceiling_status"] = qc_df["drug"].map(
        lambda dr: f"{ceilings[dr]:.3f}" if ceilings.get(dr) is not None else "not computable")
    qc_df["chosen_screen"] = qc_df["drug"].map(lambda dr: chosen[dr][0])
    qc_df["is_chosen"] = qc_df["screen"] == qc_df["chosen_screen"]
    qc_df.to_csv(PROC / "multidrug_qc.csv", index=False)
    print(f"  -> {PROC / 'multidrug_qc.csv'}")

    dropped = qc_df[qc_df.n < MIN_N_TASK]

    lines = []
    lines.append("# Multi-drug QC\n")
    lines.append("Run against GDSC1/GDSC2, repeating 5-FU's drug-row and "
                  "screen-agreement checks per drug. Full numbers in `data/processed/multidrug_qc.csv`.\n")

    lines.append("## Screen chosen per drug\n")
    for drug in DRUGS:
        lbl, reason = chosen[drug]
        lines.append(f"- **{drug}**: {lbl}. Reason: {reason}.")
    lines.append("")

    lines.append("## Sample size (n >= 300 acceptance criterion)\n")
    if dropped.empty:
        lines.append("All drug x screen combinations meet n >= 300; nothing dropped.\n")
    else:
        for _, r in dropped.iterrows():
            lines.append(f"- **{r.drug} ({r.screen})**: n={r.n}, below 300. "
                          f"Reason: {'only screened in one arm' if pd.isna(r.get('id_agreement_r')) else 'small screened cohort'}.")
        lines.append("")

    lines.append("## Measurement ceiling (GDSC1 vs GDSC2 AUC, shared lines)\n")
    for drug in DRUGS:
        c = ceilings.get(drug)
        if c is not None:
            lines.append(f"- **{drug}**: r = {c:.3f}.")
        else:
            lines.append(f"- **{drug}**: not computable (screened in one arm only, "
                          f"no other drug's ceiling is substituted).")
    lines.append("")
    fu_r = ceilings.get("5-Fluorouracil")
    fu_r_str = f"{fu_r:.3f}" if fu_r is not None else "n/a"
    lines.append(f"5-FU's own ceiling here (r={fu_r_str}, solid tumours only) "
                  "is lower than the pan-cancer r=0.604 (n=902 shared lines including "
                  "haem lineages). Both are legitimate: 0.604 is the whole-panel number; "
                  "the value here restricts to the solid cohort this project models on. "
                  "Do not treat the gap as an error, and do not substitute "
                  "either number for another drug's ceiling.\n")

    lines.append("## AUC distribution and extrapolation, per drug x screen\n")
    lines.append("| Drug | Screen | n | AUC mean | AUC sd | % extrapolated | "
                  f"% >0.90 | % >{CEILING_AUC} (ceiling) | % >0.98 | "
                  "Conc range (uM) |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for _, r in qc_df.sort_values(["drug", "screen"]).iterrows():
        lines.append(f"| {r.drug} | {r.screen} | {r.n} | {r.auc_mean:.3f} | {r.auc_sd:.3f} | "
                      f"{r.pct_extrapolated:.1f}% | {r.pct_auc_gt_090:.1f}% | "
                      f"{r.pct_auc_gt_095:.1f}% | {r.pct_auc_gt_098:.1f}% | "
                      f"{r.conc_min:.4g}-{r.conc_max:.4g} |")
    lines.append("")

    lines.append("## Multi-DRUG_ID re-screens, averaged\n")
    for _, r in qc_df.iterrows():
        if len(r.drug_ids) > 1:
            ag = f"r={r.id_agreement_r:.3f}" if r.id_agreement_r is not None else "not computable"
            lines.append(f"- **{r.drug} ({r.screen})**: DRUG_IDs {r.drug_ids}, confirmed same "
                          f"DRUG_NAME/TARGET/TARGET_PATHWAY in screened_compounds_rel_8.4.csv, "
                          f"agreement {ag}, averaged per cell line.")
    lines.append("")

    lines.append("## Irinotecan vs SN-38\n")
    if iri_sn38_r is not None:
        lines.append(f"AUC correlation on shared GDSC2 lines: r = {iri_sn38_r:.3f}. "
                      f"Both carried forward as separate drugs; this is a "
                      f"QC-stage flag; 12_drug_specificity.py makes the full comparison.\n")
    else:
        lines.append("Could not be computed (one or both tables unavailable).\n")

    lines.append("## Traps checked\n")
    lines.append("- 5-FU's parameters (dose range, ceiling, screen choice) were not "
                  "inherited; every drug ran its own extraction, QC and ceiling above.")
    lines.append("- DRUG_ID multiplicity checked against screened_compounds_rel_8.4.csv "
                  "before averaging, not assumed.")
    lines.append("- Irinotecan and SN-38 carried as distinct targets given the prodrug "
                  "relationship, not merged.")

    DOCS.mkdir(parents=True, exist_ok=True)
    (DOCS / "10_multidrug_targets.md").write_text("\n".join(lines) + "\n")
    print(f"  -> {DOCS / '10_multidrug_targets.md'}")

    banner("DONE")
    print(f"  Wrote {len(tables)} target tables -> {TARGETS_DIR}")
    print("  Next: stages/11_multidrug_models.py")


if __name__ == "__main__":
    main()
