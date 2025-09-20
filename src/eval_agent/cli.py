"""Command line interface for the evaluation agent."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Callable

from eval_agent import EvaluationAgent, load_config


def _configure_logging(level: str) -> None:
    logging.basicConfig(level=getattr(logging, level), format="%(levelname)s: %(message)s")


def _run_command(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)
    config = load_config(args.config)
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


def _serve_command(args: argparse.Namespace) -> None:
    _configure_logging(args.log_level)
    try:
        import uvicorn
    except ImportError as exc:  # pragma: no cover - defensive
        raise SystemExit("uvicorn must be installed to use the serve command") from exc

    uvicorn.run(
        "eval_agent.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        log_level=args.log_level.lower(),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run evaluation agent workflows and services.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Execute an evaluation configuration once.")
    run_parser.add_argument("config", type=Path, help="Path to the evaluation configuration JSON file.")
    run_parser.add_argument(
        "--no-predictions",
        action="store_true",
        help="Do not print individual predictions to stdout.",
    )
    run_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for the run.",
    )
    run_parser.set_defaults(func=_run_command)

    serve_parser = subparsers.add_parser("serve", help="Launch the FastAPI service.")
    serve_parser.add_argument("--host", default="0.0.0.0", help="Host interface to bind the API server.")
    serve_parser.add_argument("--port", type=int, default=8000, help="Port for the API server.")
    serve_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable autoreload (useful during development).",
    )
    serve_parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["CRITICAL", "ERROR", "WARNING", "INFO", "DEBUG"],
        help="Logging level for the API server.",
    )
    serve_parser.set_defaults(func=_serve_command)

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    func: Callable[[argparse.Namespace], None] = getattr(args, "func")
    func(args)


if __name__ == "__main__":
    main()
