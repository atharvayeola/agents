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

Two configuration files live under `configs/`:

```bash
# Keyword baseline
python -m eval_agent.cli run configs/sentiment_keyword.json

# Trained scikit-learn pipeline (requires the artifact from step 2)
python -m eval_agent.cli run configs/sentiment_sklearn.json
```


Each component becomes available in configuration files, through the CLI, and via the HTTP API as soon
as it is registered.
