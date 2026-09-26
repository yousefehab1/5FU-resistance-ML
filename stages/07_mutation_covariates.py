"""
stages/07_mutation_covariates.py
==================================

The MSI confound: microsatellite instability is mechanistically tied to
5-FU handling (mismatch-repair deficiency changes how cells process
5-FU-induced DNA damage), and in COREAD it is associated with BOTH DTP and
AUC -- the definition of a confounder. This computes how much of the raw
within-COREAD DTP-resistance association (r=+0.344, stages/03_baselines.py)
survives after adjusting for it, and runs the same adjustment for TP53
mutation status (the strongest remaining mechanistic prior: HCT116 is TP53
wild-type and p53 mediates the 5-FU stem-cell response).

MSI adjustment is done here rather than in the main model (which uses
fold-internal lineage de-confounding) because MSI/
mutation status are clinical covariates, not features the final model
uses) and its driver-gene panel from `09_calibration_and_mutations.py`.

INPUTS   data/processed/{y,signature_scores}_{GDSC1,GDSC2}.parquet
         data/raw/mutations_summary_20260724.csv
OUTPUTS  data/processed/driver_mutation_correlations.csv
         data/processed/mutation_covariate_results.csv
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pandas as pd
from scipy import stats

import config as C
from lib.io import load_mutation_flags, load_screen
from lib.report import banner, write_and_report
from lib.stats import partial_corr


def main():
    banner("1. DRIVER MUTATIONS vs AUC")
    driver_rows = []
    for label in [C.TRAIN, C.TEST]:
        _, y, _ = load_screen(label)
        flags = load_mutation_flags(C.DRIVER_GENES, model_ids=y.index)
        auc = y[C.TARGET].to_numpy()
        crc = (y[C.LINEAGE_COL] == C.CRC).to_numpy()

        print(f"\n  --- {label}: {int(flags.to_numpy().sum())} mutation calls "
              f"across {len(C.DRIVER_GENES)} driver genes")
        print(f"      {'gene':<8} {'n mut':>6}  {'all solid':>18}  {'COREAD':>18}")
        for gene in [g for g in C.DRIVER_GENES if g in flags.columns]:
            v = flags[gene].to_numpy()
            r, p = stats.pearsonr(v, auc)
            row = dict(screen=label, gene=gene, n_mutant=int(v.sum()),
                       r_all_solid=r, p_all_solid=p, r_coread=float("nan"), p_coread=float("nan"))
            if crc.sum() > 10 and v[crc].std() > 0:
                rc, pc = stats.pearsonr(v[crc], auc[crc])
                row["r_coread"], row["p_coread"] = rc, pc
                crc_s = f"r={rc:+.3f} p={pc:.3f}"
            else:
                crc_s = "n/a"
            print(f"      {gene:<8} {int(v.sum()):>6}  r={r:+.3f} p={p:.3f}  {crc_s:>18}")
            driver_rows.append(row)
    write_and_report(pd.DataFrame(driver_rows), C.PROCESSED / "driver_mutation_correlations.csv",
                      "driver_mutation_correlations.csv")

    banner("2. THE MSI CONFOUND")
    print("  MMR deficiency is mechanistically linked to 5-FU handling. In COREAD")
    print("  MSI status is associated with both AUC and DTP -- the definition of")
    print("  a confounder. This is the adjusted version of the r=+0.344 headline")
    print("  DTP-resistance association (stages/03_baselines.py).\n")

    rows = []
    for label in [C.TRAIN, C.TEST]:
        _, y, s = load_screen(label)
        m = (y[C.LINEAGE_COL] == C.CRC) & y["msi_status"].notna()
        yy, ss = y[m], s[m.to_numpy()]
        auc = yy[C.TARGET].to_numpy()
        msi = (yy["msi_status"] == "MSI").astype(float).to_numpy()

        print(f"  --- {label} COREAD, n={int(m.sum())} "
              f"(MSI {int(msi.sum())}, MSS {int(len(msi) - msi.sum())})")
        for name, a, b in [("MSI vs AUC", msi, auc),
                            ("DTP vs AUC", ss["DTP"].to_numpy(), auc),
                            ("MSI vs DTP", msi, ss["DTP"].to_numpy())]:
            r, p = stats.pearsonr(a, b)
            print(f"      {name:<26} r={r:+.3f}  p={p:.4f}")
            rows.append(dict(screen=label, test=name, r=r, p=p))

        r, p = partial_corr(ss["DTP"].to_numpy(), auc, [msi])
        print(f"      {'DTP vs AUC | MSI':<26} r={r:+.3f}  p={p:.4f}   <-- adjusted")
        rows.append(dict(screen=label, test="DTP vs AUC | MSI", r=r, p=p))

        other_modules = [c for c in ss.columns if c != "DTP"]
        r, p = partial_corr(ss["DTP"].to_numpy(), auc,
                            [msi] + [ss[c].to_numpy() for c in other_modules])
        print(f"      {'DTP vs AUC | MSI + modules':<26} r={r:+.3f}  p={p:.4f}")
        rows.append(dict(screen=label, test="DTP vs AUC | MSI + modules", r=r, p=p))

        tp53 = load_mutation_flags(["TP53"], model_ids=yy.index)["TP53"].to_numpy()
        r, p = partial_corr(ss["DTP"].to_numpy(), auc, [tp53])
        print(f"      {'DTP vs AUC | TP53':<26} r={r:+.3f}  p={p:.4f}   <-- adjusted")
        rows.append(dict(screen=label, test="DTP vs AUC | TP53", r=r, p=p))
        print()

    write_and_report(pd.DataFrame(rows), C.PROCESSED / "mutation_covariate_results.csv",
                      "mutation_covariate_results.csv")
    banner("DONE")


if __name__ == "__main__":
    main()
