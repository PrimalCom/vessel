VENV      := .venv
PYTHON    := $(VENV)/bin/python
PIP       := $(VENV)/bin/pip
PYTEST    := $(VENV)/bin/pytest
CONFIG    := config.yaml

.PHONY: help venv install clean demo pipeline evaluate test lint

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*##' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*##"}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

venv: $(VENV)/bin/activate ## Create virtual environment

$(VENV)/bin/activate:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip
	@touch $@

install: venv ## Install dependencies into venv
	$(PIP) install -r requirements.txt

demo: install ## Run quick demo with synthetic phantom
	$(PYTHON) scripts/run_demo.py

pipeline: install ## Run full pipeline (use CONFIG= or INPUT= to override)
ifdef INPUT
	$(PYTHON) scripts/run_pipeline.py --input $(INPUT) --config $(CONFIG)
else
	$(PYTHON) scripts/run_pipeline.py --config $(CONFIG)
endif

evaluate: install ## Evaluate centerline (set EXTRACTED= and GT=)
	$(PYTHON) scripts/evaluate.py --extracted $(EXTRACTED) --ground-truth $(GT)

test: install ## Run test suite
	$(PYTEST) tests/ -v

clean: ## Remove venv and generated outputs
	rm -rf $(VENV) outputs/ .pytest_cache __pycache__
	find . -type d -name __pycache__ -exec rm -rf {} +
