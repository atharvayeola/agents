"""Command line interface for the evaluation agent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

from eval_agent import EvaluationAgent, load_config


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation agent workflows.")
    parser.add_argument("config", type=Path, help="Path to the evaluation configuration JSON file.")
    parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Do not print individual predictions to stdout.",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for the run.",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(level=getattr(logging, args.log_level), format="%(levelname)s: %(message)s")
    config = load_config(args.config)
    # Override saving predictions flag if requested.
    if args.no_predictions:
        config.output.save_predictions = False

    agent = EvaluationAgent(config)
    result = agent.run()

    summary = {
        "name": result.name,
        "task": result.task,
        "metrics": [metric.to_dict() for metric in result.metrics],
        "output_path": str(result.output_path) if result.output_path else None,
    }

    print(json.dumps(summary, indent=2))

    if not args.no_predictions:
        print("\nPredictions:")
        for prediction in result.predictions:
            print(json.dumps(prediction.to_dict(), ensure_ascii=False))


if __name__ == "__main__":
    main()
