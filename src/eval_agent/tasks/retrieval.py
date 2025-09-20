"""Task implementation for retrieval-augmented question answering."""

from __future__ import annotations

from typing import List

from eval_agent.registry import TASK_REGISTRY
from eval_agent.tasks.base import Task
from eval_agent.types import Example, ModelResponse


@TASK_REGISTRY.register("retrieval-qa")
class RetrievalQuestionAnsweringTask(Task):
    """Execute retrieval + generation models over question answering datasets."""

    def run(self) -> List[ModelResponse]:
        batch_size = getattr(self.model, "batch_size", None)
        if not isinstance(batch_size, int) or batch_size <= 1:
            return [self.model.predict(example) for example in self.dataset]

        responses: List[ModelResponse] = []
        batch: list[Example] = []
        for example in self.dataset:
            batch.append(example)
            if len(batch) >= batch_size:
                responses.extend(self.model.predict_batch(batch))
                batch = []

        if batch:
            responses.extend(self.model.predict_batch(batch))

        return responses
