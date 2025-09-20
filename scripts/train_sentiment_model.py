"""Train a scikit-learn sentiment analysis pipeline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable, Tuple

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN_PATH = BASE_DIR / "data" / "sentiment_train.jsonl"
DEFAULT_EVAL_PATH = BASE_DIR / "data" / "sentiment_eval.jsonl"
DEFAULT_ARTIFACT_PATH = BASE_DIR / "artifacts" / "sentiment_pipeline.joblib"


def read_jsonl(path: Path) -> Tuple[list[str], list[str]]:
    texts: list[str] = []
    labels: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            text = payload.get("text") or payload.get("input")
            label = payload.get("label")
            if text is None or label is None:
                raise ValueError(f"Example missing 'text'/'label' fields in {path} -> {payload}")
            texts.append(text)
            labels.append(str(label))
    return texts, labels


def build_pipeline() -> Pipeline:
    return Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=1, max_features=5000)),
            (
                "clf",
                LogisticRegression(
                    multi_class="auto",
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def train(
    train_path: Path,
    eval_path: Path,
    artifact_path: Path,
) -> None:
    train_texts, train_labels = read_jsonl(train_path)
    eval_texts, eval_labels = read_jsonl(eval_path)

    pipeline = build_pipeline()
    pipeline.fit(train_texts, train_labels)

    preds = pipeline.predict(eval_texts)
    report = classification_report(eval_labels, preds, digits=3)

    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, artifact_path)

    print("Saved model artifact to", artifact_path)
    print("Evaluation report on validation split:\n")
    print(report)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the bundled sentiment classifier.")
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN_PATH, help="Path to JSONL training data")
    parser.add_argument("--eval", dest="eval_path", type=Path, default=DEFAULT_EVAL_PATH, help="Path to JSONL evaluation data")
    parser.add_argument("--output", type=Path, default=DEFAULT_ARTIFACT_PATH, help="Where to write the trained pipeline artifact")
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> None:
    args = parse_args(argv)
    train_path = args.train if args.train.is_absolute() else (BASE_DIR / args.train)
    eval_path = args.eval_path if args.eval_path.is_absolute() else (BASE_DIR / args.eval_path)
    artifact_path = args.output if args.output.is_absolute() else (BASE_DIR / args.output)

    train(train_path=train_path, eval_path=eval_path, artifact_path=artifact_path)


if __name__ == "__main__":
    main()
