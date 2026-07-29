# ---------------------------------------------------------------------------
# Wind Turbine Predictive Maintenance Platform - Failure Prediction Module
#
# Every target has an equivalent direct command documented in the README, for
# environments without `make`. Run `make help` for the list.
# ---------------------------------------------------------------------------
.DEFAULT_GOAL := help
.PHONY: help install install-dev data prepare train evaluate pipeline test test-fast \
        lint format typecheck api dashboard docs-figures docker-build docker-up docker-down \
        clean clean-all prepare-health train-health health-pipeline health-pipeline-force \
        pipeline-all

PYTHON ?= python
PIP    ?= $(PYTHON) -m pip
API_PORT       ?= 8000
DASHBOARD_PORT ?= 8501

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Setup -----------------------------------------------------------------
install:  ## Install the package and its pinned runtime dependencies
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	# --no-deps keeps the pinned versions in requirements.txt authoritative.
	$(PIP) install -e . --no-deps

install-dev: install  ## Install development extras (pre-commit; test tools are pinned already)
	-pre-commit install

# --- Pipeline --------------------------------------------------------------
data:  ## Generate the synthetic SCADA dataset
	$(PYTHON) scripts/generate_synthetic_data.py

prepare:  ## Validate, clean, label and feature-engineer the dataset
	$(PYTHON) scripts/prepare_data.py

train:  ## Train candidates, optimise the threshold and publish the model
	$(PYTHON) scripts/train_failure_model.py

evaluate:  ## Evaluate the published model and build figures + SHAP artifacts
	$(PYTHON) scripts/evaluate_failure_model.py

pipeline:  ## Run the full pipeline end to end (data -> prepare -> train -> evaluate)
	$(PYTHON) scripts/run_failure_pipeline.py

pipeline-force:  ## Rebuild everything from scratch, ignoring caches
	$(PYTHON) scripts/run_failure_pipeline.py --force

# --- Turbine Health Monitoring ---------------------------------------------
# The health module reuses the same raw dataset, so `health-pipeline` does not
# regenerate it unless --force is passed. That is deliberate: both modules must
# describe the same fleet.
prepare-health:  ## Validate, label and feature-engineer the health dataset
	$(PYTHON) scripts/prepare_health_data.py

train-health:  ## Train the health-score model and fit the drift detectors
	$(PYTHON) scripts/train_health_model.py

health-pipeline:  ## Run the health pipeline end to end (prepare -> train)
	$(PYTHON) scripts/run_health_pipeline.py

health-pipeline-force:  ## Rebuild the health data and model from scratch
	$(PYTHON) scripts/run_health_pipeline.py --force

pipeline-all:  ## Run both module pipelines against one shared dataset
	$(PYTHON) scripts/run_failure_pipeline.py
	$(PYTHON) scripts/run_health_pipeline.py

# --- Quality ---------------------------------------------------------------
test:  ## Run the full test suite
	$(PYTHON) -m pytest

test-fast:  ## Run the test suite, skipping slow tests
	$(PYTHON) -m pytest -m "not slow"

lint:  ## Lint with ruff
	$(PYTHON) -m ruff check .

format:  ## Auto-format and auto-fix with ruff
	$(PYTHON) -m ruff format .
	$(PYTHON) -m ruff check --fix .

docs-figures:  ## Refresh the result figures committed under docs/images/
	@mkdir -p docs/images
	@for f in pr_curve_test confusion_matrix_test threshold_curve model_comparison \
	          global_feature_importance shap_summary local_explanation_high_risk; do \
		cp artifacts/figures/$$f.png docs/images/$$f.png 2>/dev/null || \
			echo "  missing: artifacts/figures/$$f.png (run make pipeline)"; \
	done
	@echo "docs/images refreshed"

# --- Services --------------------------------------------------------------
api:  ## Serve the FastAPI app with reload (http://localhost:8000/docs)
	$(PYTHON) -m uvicorn wind_turbine_pm.api.main:app --reload --port $(API_PORT)

dashboard:  ## Serve the Streamlit dashboard (http://localhost:8501)
	$(PYTHON) -m streamlit run dashboard/app.py --server.port $(DASHBOARD_PORT)

# --- Docker ----------------------------------------------------------------
docker-build:  ## Build the Docker image
	docker compose build

docker-up:  ## Start the API and dashboard containers
	docker compose up --build

docker-down:  ## Stop and remove the containers
	docker compose down

docker-pipeline:  ## Run the training pipeline inside Docker
	docker compose --profile pipeline run --rm pipeline

# --- Cleaning --------------------------------------------------------------
clean:  ## Remove caches and temporary files
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -name "*.tmp" -delete

clean-all: clean  ## Also remove generated data and model artifacts
	rm -rf data/raw/* data/interim/* data/processed/*
	rm -rf artifacts/models/* artifacts/metrics/* artifacts/figures/* artifacts/metadata/*
	@echo "Generated data and artifacts removed. Rebuild with: make pipeline-all"
