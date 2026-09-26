# Limitations

What a reader should know before trusting a number here.

1. **GDSC2 is not independent validation.** 882 of its 943 lines are also in GDSC1, and expression comes from one shared matrix. Agreement with GDSC2 measures assay reproducibility. The genuinely new samples are the held-out 25% split in `06`, the colorectal transfer in `08`, and the clinical cohorts.

2. **29% of lines sit at the assay ceiling.** Moving from IC50 to AUC did not remove censoring: lines the top dose never kills have AUC near 1.0. The model has no signal there (r = -0.09). Quote the pooled r only next to the responsive-line figure (0.467).

3. **The DTP association in colorectal lines does not survive MSI adjustment** at n = 43. There are only about 60 to 70 colorectal cell lines in existence, so this cannot be fixed by collecting more lines. Treat it as hypothesis-generating.

4. **Arm B has no vehicle control and one cell line.** Induction could reflect time in culture rather than 5-FU. A vehicle-controlled time course is the single most valuable next experiment.

5. **Clinical cohorts are prognostic only.** Every patient received FOLFOX, so an association means "did better on FOLFOX", not "benefits from FOLFOX". Gene recovery on the Agilent platform is low.

6. **Hyperparameters were tuned once, on GDSC1 5-FU.** `alpha = 0.02, l1_ratio = 0.5` come from nested CV in `06` and are reused for every other drug and for the methylation models without retuning. `06_model.py --quick` skips the search entirely; do not cite a quick run. Two call sites differ on purpose to keep published numbers: `06`'s permuted arm (`alpha = 0.01, max_iter = 5000`) and `07`'s (no `l1_ratio` or `random_state`).

7. **Median imputation happens before the split**, in `fivefu.io.load_expression`. It never runs on expression (the 10% missingness filter removes every gene with any gap), so no expression result is affected. On methylation it fills about 0.01% of cells. Raising `max_gene_missing` above 0.10 would switch it on silently for expression too.

8. **Exact reproduction needs the pinned environment.** Full-data ElasticNet fits run on float32 input, and the result shifts by about 5e-7 between scikit-learn versions. Install `requirements.lock.txt` to reproduce `tests/golden/`. Casting to float64 would fix this and move numbers at the seventh decimal; that belongs in its own commit with a re-baseline.

9. **Signature scores depend on the alias map.** Without `data/raw/hgnc_alias_map.csv` the pipeline warns and scores without alias resolution, which quietly scores a different gene set under the same name.

10. **The rank-score null does not remove expression-level bias.** Within a sample, percentile ranks have a fixed mean, so the z-scored signature score is almost a linear rescaling of the plain mean rank. The exact null removed Monte Carlo noise; it does not correct for curated genes tending to be highly expressed.
