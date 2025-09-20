# Evaluation Agent

This repository contains an end-to-end evaluation stack for text-classification models. It ships with

- a modular Python evaluation agent capable of orchestrating datasets, model adapters, and metrics,
- a FastAPI service with a SQLite-backed run history, and
- a Vite + React dashboard for visualising metrics and predictions.

The default workflow benchmarks sentiment analysis models on small JSONL datasets. Two presets are
provided out of the box: a deterministic keyword baseline and a scikit-learn TF-IDF + logistic
regression model that can be re-trained locally.

## Project Layout

```
configs/                  # Evaluation configurations (keyword baseline & sklearn)
data/                     # Training and evaluation datasets (JSONL)
frontend/                 # Vite + React dashboard that consumes the FastAPI API
scripts/                  # Utility scripts (e.g., training the sklearn model)
src/eval_agent/           # Python evaluation agent, API service, and persistence
tests/                    # Pytest-based regression tests
```

## Quickstart

### 1. Install dependencies

```bash
pip install -e .
```

The Python package depends on FastAPI, uvicorn, and scikit-learn. Frontend dependencies are managed
with `npm` inside the `frontend/` directory.

### 2. (Optional) Train the scikit-learn sentiment model

The serialized pipeline is intentionally excluded from version control. Run the helper script whenever
you need to refresh the artifact:

```bash
python scripts/train_sentiment_model.py
```

This reads `data/sentiment_train.jsonl`, evaluates on `data/sentiment_eval.jsonl`, and writes
`artifacts/sentiment_pipeline.joblib` used by the `sentiment-sklearn` preset.

### 3. Execute an evaluation from the CLI

Two configuration files live under `configs/`:

```bash
# Keyword baseline
python -m eval_agent.cli run configs/sentiment_keyword.json

# Trained scikit-learn pipeline (requires the artifact from step 2)
python -m eval_agent.cli run configs/sentiment_sklearn.json
```

Results are written to `runs/` as JSON (ignored by git). The CLI will also print a summary and, by
default, the individual predictions.

### 4. Launch the API service

```bash
python -m eval_agent.cli serve --host 0.0.0.0 --port 8000
```

Available endpoints include:

- `GET /api/configs` — preset configurations exposed by the dashboard
- `GET /api/runs` — list of historical runs stored in `runs/evaluations.db`
- `POST /api/runs` — execute a configuration immediately
- `GET /api/runs/{id}` — retrieve metrics and predictions for a single run

All runs triggered through the API are persisted to SQLite with links to the JSON artifacts on disk.

### 5. Start the React dashboard

In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

By default the dashboard talks to `http://localhost:8000/api`. Override the target by exporting
`VITE_API_BASE_URL` before running `npm run dev` or `npm run build`.

### 6. Run the test suite

```bash
pytest
```

The tests cover the keyword baseline configuration to guard against regressions in the evaluation
engine.

## Extending the agent

- Register new datasets by creating a subclass of `eval_agent.datasets.base.Dataset` and decorating it
  with `@DATASET_REGISTRY.register("your-dataset-name")`.
- Add model adapters by subclassing `eval_agent.models.base.ModelAdapter` and registering via
  `@MODEL_REGISTRY.register("your-model")`.
- Implement additional metrics by extending `eval_agent.metrics.base.Metric` and registering with
  `@METRIC_REGISTRY.register("metric-name")`.
- Surface new presets in the API/dashboard by updating `PRESET_CONFIGS` in
  `src/eval_agent/api/app.py`.

Each component becomes available in configuration files, through the CLI, and via the HTTP API as soon
as it is registered.
