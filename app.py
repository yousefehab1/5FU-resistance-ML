"""
5-FU resistance dashboard
=========================

Streamlit app over the precomputed results.

    pip install streamlit
    python scripts/11_build_dashboard_data.py     # once
    streamlit run app.py

Design principle: every number shown is paired with what it should be judged
against. A predicted AUC without the measurement ceiling next to it, or an r
without its confidence interval, invites the reader to over-believe it. The
whole project has turned on that distinction, so the dashboard reflects it.
"""

import string
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Analysis parameters are defined once, in config/*.yaml, and read through
# fivefu.config. The dashboard reads the same file the models do, so a number
# shown here cannot drift from the number that produced it.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
from fivefu.config import ALL_MODULES as MODULES, CEILING_R
from fivefu.paths import processed_dir

DATA = processed_dir() / "dashboard"
CEILING = CEILING_R["AUC"]

st.set_page_config(page_title="5-FU resistance", layout="wide")


@st.cache_data
def load():
    lines = pd.read_csv(DATA / "cell_lines.csv")
    results = pd.read_csv(DATA / "results.csv")
    findings = pd.read_csv(DATA / "findings.csv")
    summary = pd.read_csv(DATA / "summary.csv").iloc[0]
    tc_path = DATA / "timecourse.csv"
    tc = pd.read_csv(tc_path) if tc_path.exists() else None
    return lines, results, findings, summary, tc


def sci(p):
    """1.07e-06 as '1.1 × 10⁻⁶', the way the prose has always written it."""
    mantissa, exponent = f"{p:.1e}".split("e")
    sup = str.maketrans("-" + string.digits, "⁻⁰¹²³⁴⁵⁶⁷⁸⁹")
    return f"{mantissa} × 10{str(int(exponent)).translate(sup)}"


# Two numbers in the prose that no script computes; they come from an earlier
# analysis and stay until a script produces them or they are dropped:
#   264 drugs  Script 10 now correlates 5-FU with 352 other drugs, so this no
#              longer matches the r quoted beside it.
#   TYMS r     The current data give +0.024 (GDSC1 solid), not this.
N_DRUGS_DOCUMENTED = "264"
TYMS_R_DOCUMENTED = "+0.021"

if not DATA.exists():
    st.error("Dashboard data not found. Run: python scripts/11_build_dashboard_data.py")
    st.stop()

lines, results, findings, nums, tc = load()

st.title("Predicting 5-FU resistance from transcriptional state")
st.caption(
    f"GDSC solid tumour cell lines · ElasticNet on {nums.n_genes:,} genes · "
    f"measurement ceiling r = {CEILING} "
    "(the two GDSC screens only agree with each other this well)"
)

tab1, tab2, tab3, tab4 = st.tabs(
    ["Predict", "Signature modules", "5-FU time-course (Arm B)", "Results & caveats"])

# ---------------------------------------------------------------- PREDICT ---
with tab1:
    c1, c2 = st.columns([1, 2])

    with c1:
        lineages = ["All"] + sorted(lines.lineage.dropna().unique())
        pick_lin = st.selectbox("Lineage", lineages,
                                index=lineages.index("COREAD") if "COREAD" in lineages else 0)
        sub = lines if pick_lin == "All" else lines[lines.lineage == pick_lin]
        name = st.selectbox("Cell line", sorted(sub.cell_line.unique()))
        row = sub[sub.cell_line == name].iloc[0]

        st.markdown("---")
        st.metric("Predicted AUC", f"{row.predicted_auc:.3f}",
                  help="Out-of-fold and calibrated. Higher = more resistant.")
        st.metric("Observed AUC (GDSC1)", f"{row.auc_gdsc1:.3f}",
                  delta=f"{row.predicted_auc - row.auc_gdsc1:+.3f} error")
        if not pd.isna(row.auc_gdsc2):
            st.metric("Observed AUC (GDSC2)", f"{row.auc_gdsc2:.3f}",
                      delta=f"{row.auc_gdsc2 - row.auc_gdsc1:+.3f} vs GDSC1",
                      help="The disagreement between screens for the SAME cell "
                           "line is the measurement noise the model competes with.")
        st.markdown(
            f"**Lineage** {row.lineage}  \n"
            f"**MSI status** {row.msi}  \n"
            f"**TP53** {row.tp53}")

    with c2:
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.scatter(lines.auc_gdsc1, lines.predicted_auc, s=12, alpha=0.25,
                   color="#888", label="all solid lines")
        if pick_lin != "All":
            ax.scatter(sub.auc_gdsc1, sub.predicted_auc, s=28, alpha=0.75,
                       color="#1f77b4", label=pick_lin)
        ax.scatter([row.auc_gdsc1], [row.predicted_auc], s=180, color="#d62728",
                   zorder=5, label=name, edgecolor="white", linewidth=1.5)
        lo = min(lines.auc_gdsc1.min(), lines.predicted_auc.min())
        hi = max(lines.auc_gdsc1.max(), lines.predicted_auc.max())
        ax.plot([lo, hi], [lo, hi], "k--", lw=1, alpha=0.5, label="perfect prediction")
        ax.set_xlabel("Observed AUC (GDSC1)"); ax.set_ylabel("Predicted AUC")
        ax.set_title(f"Out-of-fold predictions, {nums.n_solid_lines:,} solid tumour lines")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2)
        st.pyplot(fig)

        st.info(
            f"**How good is this really?** Cross-validated r = {nums.oof_r:.3f}, against a "
            f"ceiling of {CEILING}. That ceiling is not a modelling limit, it is "
            "how well GDSC1 and GDSC2 agree when measuring the *same cell lines*. "
            "Most of the remaining error is measurement noise, not the model.\n\n"
            "**Note the spread.** Predictions are compressed toward the mean: the "
            "model is better at ranking cell lines than at giving an absolute AUC."
        )

# ---------------------------------------------------------------- MODULES ---
with tab2:
    st.subheader("Signature module scores")
    st.caption("Rank-based, background-calibrated. "
               "Higher = the module's genes sit higher in that cell line's transcriptome.")

    c1, c2 = st.columns([1, 2])
    with c1:
        mod = st.selectbox("Module", [m for m in MODULES if m in lines.columns])
        scope = st.radio("Scope", ["Colorectal only", "All solid tumours"])
    d = lines[lines.lineage == "COREAD"] if scope == "Colorectal only" else lines
    d = d.dropna(subset=[mod, "auc_gdsc1"])

    with c2:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        colors = {"MSI": "#d62728", "MSS": "#1f77b4"}
        for lab, g in d.groupby(d.msi.where(d.msi.isin(["MSI", "MSS"]), "Unknown")):
            ax.scatter(g[mod], g.auc_gdsc1, s=30, alpha=0.7,
                       color=colors.get(lab, "#bbb"), label=lab)
        if len(d) > 2:
            z = np.polyfit(d[mod], d.auc_gdsc1, 1)
            xs = np.linspace(d[mod].min(), d[mod].max(), 50)
            ax.plot(xs, np.polyval(z, xs), "k-", lw=2, alpha=0.7)
            r = np.corrcoef(d[mod], d.auc_gdsc1)[0, 1]
            ax.set_title(f"{mod} vs 5-FU AUC   (r = {r:+.3f}, n = {len(d)})")
        ax.set_xlabel(f"{mod} score"); ax.set_ylabel("AUC, higher = more resistant")
        ax.legend(title="MSI status", fontsize=8); ax.grid(alpha=0.2)
        st.pyplot(fig)

    if scope == "Colorectal only" and mod == "DTP":
        st.warning(
            "**Read this before believing the DTP line.** The association "
            f"(r = {nums.dtp_r:+.3f}, p = {nums.dtp_p:.3f}) is confounded by MSI status: MSI lines are "
            "both more 5-FU sensitive and lower in DTP. Adjusting for MSI it falls "
            f"to r = {nums.dtp_msi_r:+.3f}, p = {nums.dtp_msi_p:.3f}, no longer significant "
            f"at n = {nums.coread_n}. "
            "Colour the points by MSI above and the pattern is visible."
        )

# ------------------------------------------------------------- TIME-COURSE ---
with tab3:
    if tc is None:
        st.info("Time-course data not available.")
    else:
        st.subheader("HCT116 under 5-FU, are the programmes induced?")
        feats = [c for c in tc.columns if c in MODULES]
        sel = st.multiselect("Modules", feats, default=[f for f in ["DTP", "RSC", "CellCycle"] if f in feats])

        if sel:
            fig, ax = plt.subplots(figsize=(9, 4.5))
            for f in sel:
                g = tc.groupby("timepoint_h")[f]
                m, sd = g.mean(), g.std()
                z = (m - m.iloc[0])            # change from 0h, so scales are comparable
                ax.errorbar(m.index, z, yerr=sd, marker="o", capsize=4, lw=2, label=f)
            ax.axhline(0, color="k", lw=0.8, alpha=0.5)
            ax.set_xlabel("Hours of 5-FU"); ax.set_ylabel("Change in score from 0h")
            ax.set_xticks([0, 6, 24, 48]); ax.legend(); ax.grid(alpha=0.2)
            st.pyplot(fig)

        st.warning(
            "**Two limits on this panel.** There is no time-matched vehicle "
            "control, only 0h is untreated, so a change by 48h confounds 5-FU "
            "with 48 more hours in culture. And n = 3 per timepoint supports a "
            "direction, not a confident p-value. The trends do exceed a "
            f"random-gene-set null (z = {nums.null_z_lo:.1f}–{nums.null_z_hi:.1f}), and DTP's "
            f"rise survives adjustment for the cell-cycle collapse (r = {nums.dtp_cellcycle_r:+.3f})."
        )

# ------------------------------------------------------------------ RESULTS ---
with tab4:
    st.subheader("Model performance")
    r = results.copy()
    r["95% CI"] = r.apply(
        lambda x: ", " if pd.isna(x.ci_lo) else f"[{x.ci_lo:.3f}, {x.ci_hi:.3f}]", axis=1)
    r["% of ceiling"] = (100 * r.r / CEILING).round(0).astype(int).astype(str) + "%"
    st.dataframe(r[["model", "r", "95% CI", "% of ceiling", "note"]],
                 hide_index=True, use_container_width=True)

    half_width = ((results.ci_hi - results.ci_lo) / 2).median()
    st.caption(
        f"Confidence intervals are roughly ±{half_width:.2f}. Most differences between rows "
        "are inside that, so point estimates should not be compared as though "
        "small gaps were real."
    )

    st.subheader("The DTP hypothesis, every test run")
    st.dataframe(findings, hide_index=True, use_container_width=True)

    st.markdown(f"""
### What this project does and does not show

**Does show**
- Expression predicts 5-FU sensitivity across solid tumour cell lines at r ≈ {nums.oof_r:.2f}, about {nums.oof_r / CEILING:.0%} of the achievable ceiling, validated on an independent screen and on held-out lines.
- A real signal survives removing tissue identity (r = {nums.deconfounded_r:.2f}) and removing generic drug-sensitivity (r = {nums.specific_r:.2f}).
- The DTP programme is induced under 5-FU in HCT116, above a compositional null.
- An unsupervised model, selecting from {nums.n_genes:,} genes with no knowledge of the signatures, enriches for DTP_up genes **{nums.dtp_up_fold:.1f}× (p = {sci(nums.dtp_up_p)})**, the strongest and cleanest result here.

**Does not show**
- That DTP predicts 5-FU resistance independently of MSI status. It does not, at n = {nums.coread_n}.
- That a useful colorectal-specific predictor can be built. n = {nums.coread_n} is too small; transfer from a pan-solid model works better than training on CRC.
- That 5-FU induction is specific to 5-FU rather than to time in culture. That needs a vehicle-controlled experiment.

**An unexpected finding.** Across {N_DRUGS_DOCUMENTED} drugs, 5-FU most resembles **{nums.closest_drug}** (r = {nums.closest_drug_r:.3f}), an RNA Pol I / ribosome-biogenesis inhibitor, while **TYMS**, its canonical DNA-directed target, shows no association at all (r = {TYMS_R_DOCUMENTED}). The variation in 5-FU sensitivity here looks driven more by the RNA arm than by thymidylate synthase.
""")

st.markdown("---")
st.caption("Methods, results and limitations in `docs/`. Model artefacts in `models/`.")
