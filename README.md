# Evaluation Agent

This repository contains an automated agent that evaluates AI models against configurable
datasets and metrics. It now ships with a production-ready workflow for sentiment
classification, a FastAPI backend that stores runs in SQLite, and a React dashboard for
exploring evaluation results.

## Features

- Modular registries for datasets, models, metrics, and tasks.
- JSON-based configuration system with automatic path resolution and SQLite persistence.
- Keyword baseline **and** a trained scikit-learn TF-IDF + logistic regression model.
- Accuracy, macro/weighted precision/recall/F1, confusion matrix, and label distribution metrics.
- FastAPI service with REST endpoints for triggering runs and browsing historic executions.
- React + Material UI dashboard with interactive charts and a prediction explorer.
- Command line interface for both single-run execution and launching the API server.

## Project Layout

```
configs/                  # Evaluation configurations for keyword and sklearn models
data/                     # Training and evaluation datasets (JSONL)
frontend/                 # Vite + React dashboard for the evaluation API
scripts/                  # Utility scripts (e.g., training the scikit-learn model)
src/eval_agent/           # Evaluation agent source code
tests/                    # Pytest integration tests, including API coverage
```

## Getting Started

1. **Install Python dependencies in editable mode:**

   ```bash
   pip install -e .
   ```

2. **(Optional) Re-train the bundled scikit-learn sentiment model:**

   ```bash
   python scripts/train_sentiment_model.py
   ```

   This reads `data/sentiment_train.jsonl` and writes
   `artifacts/sentiment_pipeline.joblib` used by the production configuration.

3. **Run an evaluation using either of the provided configurations:**

   ```bash
   # Keyword baseline
   python -m eval_agent.cli run configs/sentiment_keyword.json

   # Trained scikit-learn model
   python -m eval_agent.cli run configs/sentiment_sklearn.json
   ```

   The agent emits JSON artifacts in `runs/` (ignored by git) and records each run in
   `runs/evaluations.db` for the API/dashboard.

4. **Launch the API server:**

   ```bash
   python -m eval_agent.cli serve --host 0.0.0.0 --port 8000
   ```

   The API exposes endpoints under `/api/*` and persists runs to SQLite. See
   `src/eval_agent/api/app.py` for the schema.

5. **Start the React dashboard (in a separate terminal):**

   ```bash
   cd frontend
   npm install
   npm run dev
   ```

   Override the API base URL by setting `VITE_API_BASE_URL` if the backend runs on a
   different host/port.

6. **Run the Python test suite:**

   ```bash
   pytest
   ```

   Frontend builds are also covered by running `npm run build` inside the `frontend/`
   directory.

## Extending the Agent

- Register new datasets by creating a subclass of `eval_agent.datasets.base.Dataset` and
  decorating it with `@DATASET_REGISTRY.register("your-dataset-name")`.
- Add new model integrations by inheriting from `eval_agent.models.base.ModelAdapter` and
  registering via `@MODEL_REGISTRY.register("your-model")`.
- Implement additional metrics by extending `eval_agent.metrics.base.Metric` and registering
  them with `@METRIC_REGISTRY.register("metric-name")`.
- Create new tasks by subclassing `eval_agent.tasks.base.Task` and registering with
  `@TASK_REGISTRY.register("task-name")`.
- Surface new presets via the API by adding entries to `PRESET_CONFIGS` in
  `src/eval_agent/api/app.py`; the React UI automatically lists these options.

Each component becomes available for use in configuration files and through the API
immediately after registration.
