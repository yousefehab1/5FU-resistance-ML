# Stages run in numeric order. The three R steps in stages/prep/ need Bioconductor
# and download from GEO; run `make r-prep` once before `make run`, then only when
# their inputs change.

PY ?= $(shell test -x .venv/bin/python && echo .venv/bin/python || echo python3)
PY_STAGES := $(sort $(wildcard stages/*.py))

.PHONY: help setup run r-prep test dashboard golden-update clean

help:
	@echo "setup          install pinned dependencies"
	@echo "run            every Python stage, in order"
	@echo "r-prep         the R preparation steps (clinical, methylation)"
	@echo "test           library tests + golden gate"
	@echo "dashboard      launch the Streamlit app"
	@echo "golden-update  re-freeze tests/golden/ after an intentional change"

setup:
	$(PY) -m pip install -r requirements.txt

run:
	@set -e; for s in $(PY_STAGES); do echo ">>> $$s"; $(PY) $$s; done

r-prep:
	Rscript stages/prep/clinical_prep.R
	Rscript stages/prep/methylation_prep.R
	Rscript stages/prep/methylation_context_prep.R

test:
	$(PY) -m pytest -q

dashboard:
	streamlit run app.py

golden-update:
	for f in tests/golden/*.csv; do cp data/processed/$$(basename $$f) $$f; done

clean:
	rm -rf .pytest_cache
	find . -name '__pycache__' -type d -prune -exec rm -rf {} +
