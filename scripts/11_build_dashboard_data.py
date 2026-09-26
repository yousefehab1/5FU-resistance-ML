"""Build the small tables app.py reads. Runs last.

Every number is looked up by row label in the CSV the producing script wrote;
a label that matches nothing stops the build. Predicted AUCs are calibrated
with the constants from 09 (a = 0.514, b = 0.482).

Inputs:  data/processed/*.csv from 03, 05, 07, 08, 10, 13
Outputs: data/processed/dashboard/{cell_lines,results,findings,summary,timecourse}.csv
Run:     python scripts/11_build_dashboard_data.py
"""

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.model_selection import KFold

warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from fivefu.paths import project_root

ROOT = project_root()
RAW, PROC = ROOT / "data" / "raw", ROOT / "data" / "processed"
OUT = PROC / "dashboard"

# Analysis parameters are defined once, in config/*.yaml, and read
# through fivefu.config. Nothing here may re-declare one: a second
# copy is how the config file quietly stops being what decides.
from fivefu.config import N_FOLDS, SEED, HAEM, ALL_MODULES as MODULES
from fivefu.io import load_screen
from fivefu.modeling import elasticnet_pipeline
from fivefu.report import banner
from fivefu import signatures as S

TIMECOURSE = RAW / "Sup_Table_2_HCT116_5FU_timecourse_treatment.txt"

SAMPLES = {
    "PN0129B_P_0h_Ctrl_black": 0, "PN0129B_P_0h_Ctrl_blue": 0, "PN0129B_P_0h_Ctrl_red": 0,
    "PN0129B_P_6h_5FU_black": 6, "PN0129B_P_6h_5FU_blue": 6, "PN0129B_P_6h_5FU_red": 6,
    "PN0129B_P_24h_5FU_black": 24, "PN0129B_P_24h_5FU_blue": 24, "PN0129B_P_24h_5FU_red": 24,
    "PN0129B_P_48h_5FU_black": 48, "PN0129B_P_48h_5FU_blue": 48, "PN0129B_P_48h_5FU_red": 48,
}


def load(label):
    """Every line, haematological ones included.

    The dashboard describes the screens as they are, so the solid-tumour filter
    is not applied at load time. main() derives its own `solid` mask from HAEM
    afterwards, which is why the rows have to still be here.
    """
    return load_screen(label, drop_haem=False)


def one(frame, **where):
    """The single row of `frame` matching every column=value pair.

    Zero or several matches raise. A row renamed upstream must stop the build
    here, not reach the dashboard as an empty or stale cell.
    """
    hit = np.ones(len(frame), dtype=bool)
    for col, value in where.items():
        hit &= (frame[col] == value).to_numpy()
    if hit.sum() != 1:
        raise LookupError(f"expected exactly one row where {where}, found {hit.sum()}")
    return frame[hit].iloc[0]


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    banner("1. CELL LINE TABLE")
    X1, y1, s1 = load("GDSC1")
    X2, y2, s2 = load("GDSC2")
    genes = sorted(set(X1.columns) & set(X2.columns))
    X1, X2 = X1[genes], X2[genes]
    solid = (~y1.TCGA_DESC.isin(HAEM)).to_numpy()
    t1 = y1.AUC.to_numpy()

    # Cross-validated predictions, so displayed values are out-of-sample rather
    # than the model scoring its own training data.
    pipe = elasticnet_pipeline()
    Xs, ts = X1[solid], t1[solid]
    oof = np.full(len(ts), np.nan)
    for tr, te in KFold(N_FOLDS, shuffle=True, random_state=SEED).split(Xs):
        oof[te] = pipe.fit(Xs.iloc[tr], ts[tr]).predict(Xs.iloc[te])
    print(f"  out-of-fold predictions for {len(ts):,} solid lines "
          f"(r = {stats.pearsonr(ts, oof)[0]:.3f})")

    # Calibration (item 5): predictions are over-dispersed, slope ~0.5.
    a, b = np.polyfit(oof, ts, 1)
    cal = a * oof + b
    print(f"  calibration a={a:.3f}, b={b:.3f} -> "
          f"bias {cal.mean() - ts.mean():+.4f} (was {oof.mean() - ts.mean():+.4f})")

    mut = pd.read_csv(RAW / "mutations_summary_20260724.csv", low_memory=False)
    tp53 = set(mut[mut.gene_symbol == "TP53"].model_id)

    ys = y1[solid]
    tab = pd.DataFrame({
        "model_id": ys.index,
        "cell_line": ys.CELL_LINE_NAME.values,
        "lineage": ys.TCGA_DESC.values,
        "msi": ys.msi_status.fillna("Unknown").values,
        "tp53": ["mutant" if i in tp53 else "wild-type/unknown" for i in ys.index],
        "auc_gdsc1": ts,
        "predicted_auc": cal,
        "auc_gdsc2": [y2.AUC.get(i, np.nan) for i in ys.index],
    })
    for m_ in MODULES:
        if m_ in s1.columns:
            tab[m_] = s1[solid][m_].values
    tab.to_csv(OUT / "cell_lines.csv", index=False)
    print(f"  wrote cell_lines.csv  ({tab.shape[0]:,} x {tab.shape[1]})")

    banner("2. TIME-COURSE")
    if TIMECOURSE.exists():
        tc = pd.read_csv(TIMECOURSE, sep="\t", low_memory=False)
        tc = tc.drop(columns=[c for c in tc.columns if str(c).startswith("Unnamed")])
        cnt = tc.set_index("NAME")[list(SAMPLES)].apply(pd.to_numeric, errors="coerce")
        cnt = cnt.groupby(level=0).sum()
        cnt = cnt[(cnt > 0).sum(axis=1) >= 6].T
        sigs = S.load_signatures(cnt.columns, S.load_alias_map(required=False), verbose=False)
        sc, _ = S.score_all(cnt, sigs, verbose=False)
        sc["timepoint_h"] = [SAMPLES[i] for i in sc.index]
        sc.reset_index().rename(columns={"index": "sample"}).to_csv(
            OUT / "timecourse.csv", index=False)
        print(f"  wrote timecourse.csv  ({sc.shape[0]} samples x {sc.shape[1] - 1} features)")
    else:
        print("  time-course file not found -- tab will be hidden")

    banner("3. RESULTS SUMMARY")
    # Every number below is read from the file the analysis wrote, never typed
    # in: a typed copy goes stale the first time a script changes, and nothing
    # would notice. Values are rounded to the 3 places the tables display.
    base = pd.read_csv(PROC / "baseline_results.csv")
    refined = pd.read_csv(PROC / "refined_model_results.csv")
    final = pd.read_csv(PROC / "final_model_results.csv")
    multi = pd.read_csv(PROC / "multidrug_results.csv")
    conf = pd.read_csv(PROC / "dtp_confounders.csv")
    spec = pd.read_csv(PROC / "dtp_specificity.csv")
    induction = pd.read_csv(PROC / "armB_induction_stats.csv")
    enrich = pd.read_csv(PROC / "multidrug_enrichment.csv")

    def ci_row(label, frame, key, value, r, lo, hi, note):
        x = one(frame, **{key: value})
        return (label, round(x[r], 3), round(x[lo], 3), round(x[hi], 3), note)

    def point_row(label, frame, key, value, r, note):
        return (label, round(one(frame, **{key: value})[r], 3), np.nan, np.nan, note)

    def coread(screen, test):
        return one(conf, screen=screen, test=test)

    n1, n2 = coread("GDSC1", "DTP vs AUC").n, coread("GDSC2", "DTP vs AUC").n
    rows = [
        point_row("Lineage only (all lineages)", base, "baseline", "B2 lineage only",
                  "pearson_r", "the original bar"),
        point_row("Lineage only (solid tumours)", refined, "model", "lineage only, solid tumours",
                  "pearson_r", "bar after removing blood cancers"),
        ci_row("Transcriptome, raw AUC", final, "model", "transcriptome -> raw AUC",
               "r_mean", "r_lo", "r_hi", "most of this is tissue identity"),
        ci_row("Transcriptome, lineage de-confounded", final, "model",
               "transcriptome -> AUC, lineage de-confounded",
               "r_mean", "r_lo", "r_hi", "within-tissue signal"),
        ci_row("Transcriptome, 5-FU specific", multi, "model",
               "expression -> 5-FU SPECIFIC (general removed)",
               "r", "lo", "hi", "generic drug-sensitivity removed"),
        point_row("Transcriptome -> GDSC2 (external)", refined, "model",
                  "transcriptome -> GDSC2 solid", "pearson_r", "independent screen"),
        ci_row("Modules only", final, "model", "modules -> raw AUC",
               "r_mean", "r_lo", "r_hi", ""),
        ci_row("Pan-solid model -> CRC (transfer)", final, "model", "transfer pan-solid -> CRC",
               "r_mean", "r_lo", "r_hi", "beats training on CRC directly"),
        ci_row("CRC-only model", final, "model", "CRC-only modules",
               "r_mean", "r_lo", "r_hi", f"n={n1} too small"),
        point_row("Permuted labels (leakage check)", refined, "model", "PERMUTED LABELS",
                  "pearson_r", "must be ~0"),
    ]
    pd.DataFrame(rows, columns=["model", "r", "ci_lo", "ci_hi", "note"]).to_csv(
        OUT / "results.csv", index=False)

    def r_p(label, x, note, r="r", p="p"):
        return (label, f"{x[r]:+.3f}", f"{x[p]:.3f}", note)

    dtp_up = one(enrich, drug="5-Fluorouracil", screen="GDSC1", signature="DTP_up")
    findings = [
        r_p("DTP vs 5-FU resistance, GDSC1 COREAD", coread("GDSC1", "DTP vs AUC"), f"n={n1}"),
        r_p("DTP vs 5-FU resistance, GDSC2 COREAD", coread("GDSC2", "DTP vs AUC"),
            f"n={n2}, replicates"),
        r_p("DTP adjusted for MSI", coread("GDSC1", "DTP vs AUC | MSI"), "MSI is a confounder"),
        r_p("DTP adjusted for TP53", coread("GDSC1", "DTP vs AUC | TP53"), "TP53 is not"),
        r_p("DTP vs generic drug sensitivity", one(spec, target="general chemosensitivity"),
            "not generic fragility"),
        r_p("DTP vs 5-FU-specific component",
            one(spec, target="5-FU specific (general removed)"), "specific to 5-FU"),
        r_p("DTP induction over time-course", one(induction, feature="DTP"),
            "exceeds compositional null", r="trend_r", p="trend_p"),
        ("DTP_up enrichment in model genes", f"{dtp_up.fold_enrichment:.1f}x",
         f"{dtp_up.p_hypergeom:.1e}", "unsupervised rediscovery"),
    ]
    pd.DataFrame(findings, columns=["finding", "effect", "p", "note"]).to_csv(
        OUT / "findings.csv", index=False)
    print(f"  wrote results.csv and findings.csv")

    # The app's prose quotes numbers too. They come from here, one row, so
    # app.py formats rather than types them.
    similarity = pd.read_csv(PROC / "fu_drug_similarity.csv")
    closest = similarity.loc[similarity.r.idxmax()]
    null = pd.read_csv(PROC / "armB_compositional_control.csv").z.abs()
    msi_adj = coread("GDSC1", "DTP vs AUC | MSI")
    summary = dict(
        n_genes=len(genes), n_solid_lines=len(ts), oof_r=stats.pearsonr(ts, oof)[0],
        coread_n=n1,
        dtp_r=coread("GDSC1", "DTP vs AUC").r, dtp_p=coread("GDSC1", "DTP vs AUC").p,
        dtp_msi_r=msi_adj.r, dtp_msi_p=msi_adj.p,
        dtp_cellcycle_r=one(pd.read_csv(PROC / "armB_cellcycle_control.csv"),
                            feature="DTP").partial_r,
        null_z_lo=null.min(), null_z_hi=null.max(),
        deconfounded_r=one(final, model="transcriptome -> AUC, lineage de-confounded").r_mean,
        specific_r=one(multi, model="expression -> 5-FU SPECIFIC (general removed)").r,
        dtp_up_fold=dtp_up.fold_enrichment, dtp_up_p=dtp_up.p_hypergeom,
        closest_drug=closest.drug, closest_drug_r=closest.r,
    )
    pd.DataFrame([summary]).to_csv(OUT / "summary.csv", index=False)
    print(f"  wrote summary.csv  ({len(summary)} values the app's prose quotes)")

    banner("DONE")
    print(f"  dashboard data -> {OUT}")
    print(f"  now run:  streamlit run app.py")


if __name__ == "__main__":
    main()
