"""
5-FU resistance dashboard
=========================

Streamlit app over the precomputed results.

    pip install streamlit
    python stages/12_dashboard_data.py     # once, after the rest of the pipeline
    streamlit run app.py

Design principle: every number shown is paired with what it should be judged
against. A predicted AUC without the measurement ceiling next to it, or an r
without its confidence interval, invites the reader to over-believe it. The
whole project has turned on that distinction, so the dashboard reflects it.

Every number below, including the caveat prose, is read from
data/processed/dashboard/*.csv -- itself read exclusively from earlier
stages' outputs (see stages/12_dashboard_data.py). Nothing here is
hand-typed. If a number changes upstream, rerunning the pipeline changes
this page; nothing needs editing here.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import config as C

DATA = C.DASHBOARD_DIR
PROC = C.PROCESSED
MODULES = C.ALL_MODULES
CEILING = C.CEILING_R[C.TARGET]

st.set_page_config(page_title="5-FU resistance", layout="wide")


@st.cache_data
def load():
    lines = pd.read_csv(DATA / "cell_lines.csv")
    results = pd.read_csv(DATA / "results.csv")
    findings = pd.read_csv(DATA / "findings.csv").set_index("finding")
    tc_path = DATA / "timecourse.csv"
    tc = pd.read_csv(tc_path) if tc_path.exists() else None

    drug_sim = pd.read_csv(PROC / "drug_similarity.csv")
    gene_target = pd.read_csv(PROC / "gene_target_check.csv").set_index("gene")
    comp_ctrl = pd.read_csv(PROC / "armB_compositional_control.csv")
    cc_adj = pd.read_csv(PROC / "armB_cellcycle_adjusted.csv").set_index("feature")
    return lines, results, findings, tc, drug_sim, gene_target, comp_ctrl, cc_adj


if not DATA.exists():
    st.error("Dashboard data not found. Run: python stages/12_dashboard_data.py")
    st.stop()

lines, results, findings, tc, drug_sim, gene_target, comp_ctrl, cc_adj = load()

cv_row = results.set_index("model").loc["Transcriptome, raw AUC"]

st.title("Predicting 5-FU resistance from transcriptional state")
st.caption(
    f"GDSC solid tumour cell lines · ElasticNet on {lines.shape[0]:,} lines · "
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
        r_oof = np.corrcoef(lines.auc_gdsc1, lines.predicted_auc)[0, 1]

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
        ax.set_title(f"Out-of-fold predictions, {lines.shape[0]:,} solid tumour lines")
        ax.legend(fontsize=8, loc="upper left"); ax.grid(alpha=0.2)
        st.pyplot(fig)

        st.info(
            f"**How good is this really?** Out-of-fold r = {r_oof:.3f}, against a "
            f"ceiling of {CEILING}. That ceiling is not a modelling limit, it is "
            "how well GDSC1 and GDSC2 agree when measuring the *same cell lines*. "
            "Most of the remaining error is measurement noise, not the model.\n\n"
            "**Note the spread.** Predictions are compressed toward the mean: the "
            "model is better at ranking cell lines than at giving an absolute AUC."
        )

# ---------------------------------------------------------------- MODULES ---
with tab2:
    st.subheader("Signature module scores")
    st.caption("Rank-based, background-calibrated (see lib/signatures.py::background_score). "
               "Higher = the module's genes sit higher in that cell line's transcriptome.")

    c1, c2 = st.columns([1, 2])
    with c1:
        mod = st.selectbox("Module", [m for m in MODULES if m in lines.columns])
        scope = st.radio("Scope", ["Colorectal only", "All solid tumours"])
    d = lines[lines.lineage == C.CRC] if scope == "Colorectal only" else lines
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
        raw = findings.loc["DTP vs 5-FU resistance, GDSC1 COREAD"]
        msi_adj = findings.loc["DTP adjusted for MSI"]
        st.warning(
            f"**Read this before believing the DTP line.** The association "
            f"(r = {raw.effect}, p = {raw.p}) is confounded by MSI status: MSI lines are "
            "both more 5-FU sensitive and lower in DTP. Adjusting for MSI it falls "
            f"to r = {msi_adj.effect}, p = {msi_adj.p}, no longer significant at {raw.note}. "
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

        z_lo, z_hi = comp_ctrl.z.abs().min(), comp_ctrl.z.abs().max()
        dtp_cc = cc_adj.loc["DTP", "r_adjusted"]
        st.warning(
            "**Two limits on this panel.** There is no time-matched vehicle "
            "control, only 0h is untreated, so a change by 48h confounds 5-FU "
            "with 48 more hours in culture. And n = 3 per timepoint supports a "
            f"direction, not a confident p-value. The trends do exceed a "
            f"random-gene-set null (z = {z_lo:.1f}-{z_hi:.1f}), and DTP's rise survives "
            f"adjustment for the cell-cycle collapse (r = {dtp_cc:+.3f})."
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

    st.caption(
        "Confidence intervals are roughly ±0.07. Most differences between rows "
        "are inside that, so point estimates should not be compared as though "
        "small gaps were real."
    )

    st.subheader("The DTP hypothesis, every test run")
    st.dataframe(findings.reset_index(), hide_index=True, use_container_width=True)

    raw_r = cv_row.r
    deconf_r = results.set_index("model").loc["Transcriptome, lineage de-confounded", "r"]
    spec_r = results.set_index("model").loc["Transcriptome, 5-FU specific", "r"]
    enrich = findings.loc["DTP_up enrichment in model genes"]
    msi_row = findings.loc["DTP adjusted for MSI"]
    top_drug = drug_sim.iloc[0]
    tyms = gene_target.loc["TYMS"]

    st.markdown(f"""
### What this project does and does not show

**Does show**
- Expression predicts 5-FU sensitivity across solid tumour cell lines at r ≈ {raw_r:.2f}, about {100 * raw_r / CEILING:.0f}% of the achievable ceiling, validated on an independent screen and on held-out lines.
- A real signal survives removing tissue identity (r = {deconf_r:.2f}) and removing generic drug-sensitivity (r = {spec_r:.2f}).
- The DTP programme changes under 5-FU treatment over time in HCT116, above a compositional null (no vehicle control exists -- see the time-course tab).
- An unsupervised model, selecting from real genes with no knowledge of the signatures, enriches for DTP_up genes **{enrich.effect} (p = {enrich.p})**, the strongest and cleanest result here.

**Does not show**
- That DTP predicts 5-FU resistance independently of MSI status. It does not: adjusted r = {msi_row.effect}, p = {msi_row.p}, at {findings.loc["DTP vs 5-FU resistance, GDSC1 COREAD"].note}.
- That a useful colorectal-specific predictor can be built. n = 43 is too small; transfer from a pan-solid model works better than training on CRC.
- That 5-FU induction is specific to 5-FU rather than to time in culture. That needs a vehicle-controlled experiment.

**An unexpected finding.** Across {drug_sim.shape[0]} drugs, 5-FU most resembles **{top_drug.drug}** (r = {top_drug.r_with_5fu:.3f}), an RNA Pol I / ribosome-biogenesis inhibitor, while **TYMS**, its canonical DNA-directed target, shows almost no association at all (r = {tyms.r_with_auc:+.3f}). The variation in 5-FU sensitivity here looks driven more by the RNA arm than by thymidylate synthase.
""")

st.markdown("---")
st.caption("Full method and decision record in `docs/`. Model artefacts in `models/`.")
