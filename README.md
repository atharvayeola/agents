# Evaluation Agent

This repository contains an automated agent that evaluates AI models against configurable
datasets and metrics. It now ships with a production-ready workflow for sentiment
classification, a FastAPI backend that stores runs in SQLite, and a React dashboard for
exploring evaluation results.
This repository contains an end-to-end evaluation stack for text-classification models. It ships with

## Features
- a modular Python evaluation agent capable of orchestrating datasets, model adapters, and metrics,
- a FastAPI service with a SQLite-backed run history, and
- a Vite + React dashboard for visualising metrics and predictions.

- Modular registries for datasets, models, metrics, and tasks.
- JSON-based configuration system with automatic path resolution and SQLite persistence.
- Keyword baseline **and** a trained scikit-learn TF-IDF + logistic regression model.
- Accuracy, macro/weighted precision/recall/F1, confusion matrix, and label distribution metrics.
- FastAPI service with REST endpoints for triggering runs and browsing historic executions.
- React + Material UI dashboard with interactive charts and a prediction explorer.
- Command line interface for both single-run execution and launching the API server.
The default workflow benchmarks sentiment analysis models on small JSONL datasets. Two presets are
provided out of the box: a deterministic keyword baseline and a scikit-learn TF-IDF + logistic
regression model that can be re-trained locally.

## Project Layout

```
configs/                  # Evaluation configurations for keyword and sklearn models
configs/                  # Evaluation configurations (keyword baseline & sklearn)
data/                     # Training and evaluation datasets (JSONL)
frontend/                 # Vite + React dashboard for the evaluation API
scripts/                  # Utility scripts (e.g., training the scikit-learn model)
src/eval_agent/           # Evaluation agent source code
tests/                    # Pytest integration tests, including API coverage
frontend/                 # Vite + React dashboard that consumes the FastAPI API
scripts/                  # Utility scripts (e.g., training the sklearn model)
src/eval_agent/           # Python evaluation agent, API service, and persistence
tests/                    # Pytest-based regression tests
```

## Getting Started
## Quickstart

1. **Install Python dependencies in editable mode:**
### 1. Install dependencies

   ```bash
   pip install -e .
   ```
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

2. **(Optional) Re-train the bundled scikit-learn sentiment model:**
```bash
# Keyword baseline
python -m eval_agent.cli run configs/sentiment_keyword.json

   ```bash
   python scripts/train_sentiment_model.py
   ```
# Trained scikit-learn pipeline (requires the artifact from step 2)
python -m eval_agent.cli run configs/sentiment_sklearn.json
```

Results are written to `runs/` as JSON (ignored by git). The CLI will also print a summary and, by
default, the individual predictions.

   This reads `data/sentiment_train.jsonl` and writes
   `artifacts/sentiment_pipeline.joblib` used by the production configuration.
### 4. Launch the API service

3. **Run an evaluation using either of the provided configurations:**
```bash
python -m eval_agent.cli serve --host 0.0.0.0 --port 8000
```

   ```bash
   # Keyword baseline
   python -m eval_agent.cli run configs/sentiment_keyword.json
Available endpoints include:

   # Trained scikit-learn model
   python -m eval_agent.cli run configs/sentiment_sklearn.json
   ```
- `GET /api/configs` — preset configurations exposed by the dashboard
- `GET /api/runs` — list of historical runs stored in `runs/evaluations.db`
- `POST /api/runs` — execute a configuration immediately
- `GET /api/runs/{id}` — retrieve metrics and predictions for a single run

   The agent emits JSON artifacts in `runs/` (ignored by git) and records each run in
   `runs/evaluations.db` for the API/dashboard.
All runs triggered through the API are persisted to SQLite with links to the JSON artifacts on disk.

4. **Launch the API server:**
### 5. Start the React dashboard

   ```bash
   python -m eval_agent.cli serve --host 0.0.0.0 --port 8000
   ```
In a separate terminal:

```bash
cd frontend
npm install
npm run dev
```

By default the dashboard talks to `http://localhost:8000/api`. Override the target by exporting
`VITE_API_BASE_URL` (either the bare API origin or a full `/api` URL) before running `npm run dev`
or `npm run build`.

### 6. Run the test suite

```bash
pytest
```

   The API exposes endpoints under `/api/*` and persists runs to SQLite. See
   `src/eval_agent/api/app.py` for the schema.
The tests cover the keyword baseline configuration to guard against regressions in the evaluation
engine.

5. **Start the React dashboard (in a separate terminal):**
## Deployment

   ```bash
   cd frontend
   npm install
   npm run dev
   ```
### Render (free tier)

   Override the API base URL by setting `VITE_API_BASE_URL` if the backend runs on a
   different host/port.
The repository ships with a [`render.yaml`](render.yaml) blueprint that provisions a free-tier FastAPI
service together with a static build of the React dashboard.

6. **Run the Python test suite:**
1. Push the repository to your own GitHub (or GitLab/Bitbucket) account.
2. Sign in to [Render](https://render.com) and choose **Blueprint** > **New Blueprint Instance**.
3. Point Render at your fork and select the `render.yaml` file when prompted.
4. Deploy the services:
   - **`eval-agent-api`** — Python web service that runs `uvicorn eval_agent.api.app:app`.
     The build step installs this package via `pip install .`, and the service exposes the HTTP
     API on the free tier.
   - **`eval-agent-dashboard`** — Static site that runs `npm install && npm run build` inside
     `frontend/` and publishes the compiled assets from `frontend/dist`.

   ```bash
   pytest
   ```
The blueprint automatically wires the dashboard's `VITE_API_BASE_URL` environment variable to the
API service URL, and the frontend code appends `/api` if necessary. Runs executed via the hosted API
write to `runs/evaluations.db`, which lives on the instance's ephemeral disk—data will be cleared
whenever the service is redeployed.

   Frontend builds are also covered by running `npm run build` inside the `frontend/`
   directory.
Adjust service names, regions, or plans in `render.yaml` if you need to scale beyond the free tier.

## Extending the Agent
## Extending the agent

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
- Register new datasets by creating a subclass of `eval_agent.datasets.base.Dataset` and decorating it
  with `@DATASET_REGISTRY.register("your-dataset-name")`.
- Add model adapters by subclassing `eval_agent.models.base.ModelAdapter` and registering via
  `@MODEL_REGISTRY.register("your-model")`.
- Implement additional metrics by extending `eval_agent.metrics.base.Metric` and registering with
  `@METRIC_REGISTRY.register("metric-name")`.
- Surface new presets in the API/dashboard by updating `PRESET_CONFIGS` in
  `src/eval_agent/api/app.py`.

Each component becomes available for use in configuration files and through the API
immediately after registration.
Each component becomes available in configuration files, through the CLI, and via the HTTP API as soon
as it is registered.
