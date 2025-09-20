# Evaluation Agent

This repository contains an automated agent that evaluates AI models against configurable
datasets and metrics. The initial implementation focuses on text classification tasks and
provides a simple keyword-based baseline model along with reusable abstractions for future
model, dataset, and metric plugins.

## Features

- Modular registries for datasets, models, metrics, and tasks.
- JSON-based configuration system with automatic path resolution.
- Keyword-matching baseline classifier for sentiment analysis data.
- Accuracy and label distribution metrics with JSON artifact export.
- Command line interface for running evaluations and inspecting predictions.

## Project Layout

```
configs/                  # Example evaluation configurations
data/                     # Sample evaluation datasets
src/eval_agent/           # Evaluation agent source code
tests/                    # Pytest integration tests
```

## Getting Started

1. **Install the project in editable mode (optional but recommended):**

   ```bash
   pip install -e .
   ```

2. **Run an evaluation using the provided sample configuration:**

   ```bash
   python -m eval_agent.cli configs/sentiment_keyword.json
   ```

   The agent will produce a JSON artifact in the configured output directory (default is
   `runs/`) that captures metrics and individual predictions.

3. **Run the test suite:**

   ```bash
   pytest
   ```

## Extending the Agent

- Register new datasets by creating a subclass of `eval_agent.datasets.base.Dataset` and
  decorating it with `@DATASET_REGISTRY.register("your-dataset-name")`.
- Add new model integrations by inheriting from `eval_agent.models.base.ModelAdapter` and
  registering via `@MODEL_REGISTRY.register("your-model")`.
- Implement additional metrics by extending `eval_agent.metrics.base.Metric` and registering
  them with `@METRIC_REGISTRY.register("metric-name")`.
- Create new tasks by subclassing `eval_agent.tasks.base.Task` and registering with
  `@TASK_REGISTRY.register("task-name")`.

Each component becomes available for use in configuration files immediately after
registration.
