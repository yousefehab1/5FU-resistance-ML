# Predicting 5-FU resistance from regenerative transcriptional state

Youssef Elabd, 2026

**Question.** Do the drug-tolerant persister (DTP) and regenerative stem-cell programmes from my MSc work predict baseline 5-fluorouracil resistance across cancer cell lines, and does 5-FU treatment induce those same programmes?

**Answer, in short.**

- Expression predicts 5-FU sensitivity at r = 0.43 across 786 solid-tumour lines (about 70% of the r ≈ 0.60 ceiling set by the two screens' own agreement). That rises to r = 0.47 among lines the assay can actually resolve.
- A model that knew nothing about the signatures picked 413 genes, and those genes are 4.7x enriched for DTP_up (p = 1e-6).
- The DTP programme is induced over a 48 h 5-FU time course in HCT116 (r = 0.82), beyond a compositional null.
- The DTP to resistance link in colorectal lines weakens once MSI status is adjusted for (r 0.34 to 0.23, no longer significant at n = 43), so it is hypothesis-generating, not established.

Full results: [`docs/RESULTS.md`](docs/RESULTS.md). What they cannot be stretched to claim: [`docs/LIMITATIONS.md`](docs/LIMITATIONS.md).

## Run it

```bash
python3.12 -m venv .venv
make setup        # pinned dependencies + the fivefu package
make features     # once: raw expression -> modelling matrices (slow)
make run          # every analysis, writes data/processed/
make test         # behaviour tests + golden gate
make dashboard    # Streamlit app
```

The input files and where to download them are listed in [`docs/METHODS.md`](docs/METHODS.md#data). The R steps (17a, 18a, 23a and the `export_*.R` helpers) need R with Bioconductor.

## Layout

| Path | Contents |
|---|---|
| `scripts/` | The pipeline. Numbered scripts, run in the order the `Makefile` lists. |
| `src/fivefu/` | Shared library: loading, signature scoring, modelling, statistics. |
| `config/` | Every analysis parameter (seed, CV folds, ElasticNet settings, drugs, signatures). |
| `data/raw/` | Downloaded inputs. Never written to. Not in git. |
| `data/processed/` | Everything the pipeline writes, including per-script markdown reports in `reports/`. Safe to delete and regenerate. Not in git. |
| `tests/` | `test_library.py` (behaviour) and `test_golden.py` (every result table must match `tests/golden/`). |
| `app.py` | Dashboard. Reads only `data/processed/dashboard/`. |
| `docs/` | `METHODS.md`, `RESULTS.md`, `LIMITATIONS.md`. |

## Rules the code keeps

1. Feature selection and scaling happen inside each cross-validation fold. A permuted-label run accompanies every model and must give r ≈ 0.
2. Lineage de-confounding estimates tissue means on training rows only.
3. Results are quoted against the measurement ceiling and next to trivial baselines (lineage, proliferation), not against r = 1.
4. `data/raw/` is never modified; symbol repairs happen at load time.
5. Reproducing `tests/golden/` exactly needs `requirements.lock.txt`.
