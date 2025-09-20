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
`VITE_API_BASE_URL` (either the bare API origin or a full `/api` URL) before running `npm run dev`
or `npm run build`. When you build for production against the co-hosted FastAPI service, set
`VITE_API_BASE_URL=/` so the generated bundle issues relative requests to `/api`.

### 6. Run the test suite

```bash
pytest
```

The tests cover the keyword baseline configuration to guard against regressions in the evaluation
engine.

## Deployment

### Render (free tier)

The repository ships with a [`render.yaml`](render.yaml) blueprint that provisions a free-tier FastAPI
service together with a static build of the React dashboard.

1. Push the repository to your own GitHub (or GitLab/Bitbucket) account.
2. Sign in to [Render](https://render.com) and choose **Blueprint** > **New Blueprint Instance**.
3. Point Render at your fork and select the `render.yaml` file when prompted.
4. Deploy the services:
   - **`eval-agent-api`** — Python web service that runs `uvicorn eval_agent.api.app:app`.
     The build step installs this package via `pip install .`, and the service exposes the HTTP
     API on the free tier.
   - **`eval-agent-dashboard`** — Static site that runs `npm install && npm run build` inside
     `frontend/` and publishes the compiled assets from `frontend/dist`.

The blueprint automatically wires the dashboard's `VITE_API_BASE_URL` environment variable to the
API service URL, and the frontend code appends `/api` if necessary. Runs executed via the hosted API
write to `runs/evaluations.db`, which lives on the instance's ephemeral disk—data will be cleared
whenever the service is redeployed.

Adjust service names, regions, or plans in `render.yaml` if you need to scale beyond the free tier.

### Heroku (container deploy)

The repository now includes a multi-stage [`Dockerfile`](Dockerfile) and a [`heroku.yml`](heroku.yml)
definition that bundle the API and the compiled React dashboard into a single container. Heroku's
free tier can run this setup via the container stack:

1. Install the [Heroku CLI](https://devcenter.heroku.com/articles/heroku-cli) and authenticate with
   `heroku login`.
2. Create an app that uses the container stack: `heroku create <your-app-name> --stack container`.
3. Push the Docker image: `heroku container:push web --app <your-app-name>`.
4. Release the image: `heroku container:release web --app <your-app-name>`.
5. Visit the deployed dashboard with `heroku open --app <your-app-name>` or check health via
   `https://<your-app-name>.herokuapp.com/api/health`.

The Docker image builds the dashboard with `VITE_API_BASE_URL=/`, so the static assets served by the
FastAPI application automatically speak to the same origin. No additional environment variables are
required; evaluation runs are persisted in the container's ephemeral `runs/evaluations.db` SQLite
database just like the local setup.

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
