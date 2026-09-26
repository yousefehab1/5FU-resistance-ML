# Pipeline order is the order the scripts are listed here.
# R steps (17a, 18a, 23a) need Bioconductor; run them only when their inputs change.

PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
S  := scripts

.PHONY: help setup test run features core methylation clinical dashboard golden-update clean

help:
	@echo "setup          install pinned dependencies and the fivefu package"
	@echo "test           behaviour tests + golden gate"
	@echo "features       rebuild X/y matrices from raw expression (slow, run when raw data changes)"
	@echo "run            core, then methylation, then clinical"
	@echo "dashboard      launch the Streamlit app"
	@echo "golden-update  re-freeze tests/golden/ after an intentional change to a result"

setup:
	$(PY) -m pip install -r requirements.lock.txt
	$(PY) -m pip install -e ".[test,dashboard]"

test:
	$(PY) -m pytest -q

features:
	$(PY) $(S)/02_build_modelling_table.py

run: core methylation clinical

core:
	$(PY) $(S)/01_explore_drug_response.py
	$(PY) $(S)/03_baselines.py
	$(PY) $(S)/05_armB_induction.py
	$(PY) $(S)/06_model.py
	$(PY) $(S)/07_refined_model.py
	$(PY) $(S)/08_final_model.py
	$(PY) $(S)/09_calibration_and_mutations.py
	$(PY) $(S)/10_multidrug.py
	$(PY) $(S)/12_multidrug_targets.py
	$(PY) $(S)/13_multidrug_models.py
	$(PY) $(S)/14_drug_specificity.py
	$(PY) $(S)/11_build_dashboard_data.py   # last: reads 13's output

methylation:
	Rscript $(S)/18a_methylation_prep.R
	$(PY) $(S)/18_methylation_ingest.py
	$(PY) $(S)/19_mlh1_msi.py
	$(PY) $(S)/20_methylation_models.py
	$(PY) $(S)/21_tf_activity.py
	$(PY) $(S)/22_regulatory_architecture.py
	Rscript $(S)/23a_methylation_context_prep.R
	$(PY) $(S)/23_methylation_context.py

clinical:
	$(PY) $(S)/16_score_external.py
	$(PY) $(S)/17_clinical_validation.py

dashboard:
	streamlit run app.py

golden-update:
	for f in tests/golden/*.csv; do cp data/processed/$$(basename $$f) $$f; done

clean:
	rm -rf .pytest_cache build dist src/*.egg-info
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
