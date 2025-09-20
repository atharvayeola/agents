"""LangChain-powered retrieval-augmented generation model adapter."""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough

from eval_agent.models.base import ModelAdapter
from eval_agent.registry import MODEL_REGISTRY
from eval_agent.types import Example, ModelResponse


def _load_documents(path: Path) -> List[Document]:
    if not path.exists():
        raise FileNotFoundError(f"Knowledge base file not found at {path}")

    def _from_mapping(mapping: Dict[str, Any]) -> Document:
        identifier = mapping.get("id")
        text = mapping.get("text") or mapping.get("content")
        if text is None:
            raise ValueError("Context entries must include a 'text' field")
        metadata = {key: value for key, value in mapping.items() if key not in {"text", "content"}}
        if identifier is None:
            identifier = str(len(documents))
        metadata.setdefault("id", str(identifier))
        return Document(page_content=str(text), metadata=metadata)

    documents: List[Document] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle):
                line = line.strip()
                if not line:
                    continue
                payload = json.loads(line)
                if not isinstance(payload, dict):
                    raise ValueError("Each JSONL context entry must be an object")
                documents.append(_from_mapping(payload))
    else:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, list):
            for entry in payload:
                if isinstance(entry, dict):
                    documents.append(_from_mapping(entry))
                elif isinstance(entry, str):
                    metadata = {"id": str(len(documents))}
                    documents.append(Document(page_content=entry, metadata=metadata))
                else:
                    raise ValueError("List-based contexts must contain strings or objects with 'text'")
        elif isinstance(payload, dict):
            for key, value in payload.items():
                if not isinstance(value, str):
                    raise ValueError("Dictionary-based contexts must map identifiers to strings")
                documents.append(Document(page_content=value, metadata={"id": str(key)}))
        else:
            raise ValueError("Unsupported knowledge base format. Use JSON, JSONL, or a mapping of texts.")

    if not documents:
        raise ValueError(f"No documents were loaded from {path}")
    return documents


class ContextualAnswerLLM(LLM):
    """A lightweight, deterministic LLM that selects the most relevant context snippet."""

    model_config = {"extra": "allow"}
    default_response: str = "I'm not sure."

    def __init__(self, *, default_response: str = "I'm not sure.") -> None:
        super().__init__()
        self.default_response = default_response

    @property
    def _llm_type(self) -> str:
        return "contextual-answer"

    def _call(self, prompt: str, stop: Sequence[str] | None = None) -> str:
        context_block = ""
        question = ""
        if "Context:" in prompt:
            after_context = prompt.split("Context:", 1)[1]
            if "\n\nQuestion:" in after_context:
                context_block, question_section = after_context.split("\n\nQuestion:", 1)
            else:
                context_block = after_context
                question_section = ""
        else:
            question_section = prompt

        if "Question:" in question_section:
            question = question_section.split("Question:", 1)[1]
        if "Answer:" in question:
            question = question.split("Answer:", 1)[0]
        question = question.strip()

        raw_contexts = [segment.strip() for segment in context_block.split("\n---\n") if segment.strip()]
        contexts = [self._clean_context(entry) for entry in raw_contexts if entry]
        if not contexts:
            response = self.default_response
        else:
            response = self._select_best_context(contexts, question)

        if stop:
            for token in stop:
                if token in response:
                    response = response.split(token)[0]
        return response.strip() or self.default_response

    def _clean_context(self, text: str) -> str:
        if text.startswith("["):
            closing = text.find("]")
            if closing != -1:
                text = text[closing + 1 :]
        return text.strip()

    def _select_best_context(self, contexts: Sequence[str], question: str) -> str:
        if not question:
            return contexts[0]
        tokens = [token.lower() for token in re.findall(r"\w+", question)]
        best_score = -1
        best_context = contexts[0]
        for context in contexts:
            lower = context.lower()
            score = sum(1 for token in tokens if token and token in lower)
            if score > best_score:
                best_score = score
                best_context = context
        return best_context


@MODEL_REGISTRY.register("langchain-rag")
class LangChainRagModel(ModelAdapter):
    """Model adapter that orchestrates LangChain components for RAG workflows."""

    def __init__(
        self,
        *,
        documents_path: str | Path,
        retriever_top_k: int = 3,
        embedding_size: int = 768,
        prompt_template: str | None = None,
        default_response: str = "I'm not sure.",
        batch_size: int | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        self.documents_path = Path(documents_path)
        self.retriever_top_k = max(1, int(retriever_top_k))
        self.embedding_size = max(8, int(embedding_size))
        self.prompt_template = (
            prompt_template
            or "You are a helpful assistant.\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        self.default_response = default_response
        self.batch_size = batch_size

        self._vectorstore: FAISS | None = None
        self._pipeline = None
        self._generator_chain = None

    def warmup(self, examples: Iterable[Example] | None = None) -> None:
        _ = examples
        documents = _load_documents(self.documents_path)
        embeddings = BagOfWordsEmbeddings(dimension=self.embedding_size)
        self._vectorstore = FAISS.from_documents(documents, embeddings)

        prompt = PromptTemplate.from_template(self.prompt_template)
        llm = ContextualAnswerLLM(default_response=self.default_response)
        parser = StrOutputParser()
        self._generator_chain = prompt | llm | parser

        retrieval_step = RunnableLambda(self._retrieve_documents)
        self._pipeline = retrieval_step | RunnablePassthrough.assign(answer=self._generator_chain)

    def _retrieve_documents(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        if self._vectorstore is None:
            raise RuntimeError("The LangChain RAG model must be warmed up before predicting")

        question = str(inputs.get("question") or inputs.get("text") or "").strip()
        documents: List[Document] = []
        scores: List[float] = []
        if hasattr(self._vectorstore, "similarity_search_with_score"):
            docs_with_scores = self._vectorstore.similarity_search_with_score(
                question,
                k=self.retriever_top_k,
            )
            for doc, distance in docs_with_scores:
                documents.append(doc)
                distance_value = float(distance)
                similarity = 1.0 / (1.0 + max(distance_value, 0.0))
                scores.append(similarity)
        else:
            docs_with_scores = self._vectorstore.similarity_search_with_relevance_scores(
                question,
                k=self.retriever_top_k,
            )
            for doc, relevance in docs_with_scores:
                documents.append(doc)
                scores.append(float(relevance))
        context = self._combine_documents(documents)
        return {
            "question": question,
            "documents": documents,
            "scores": scores,
            "context": context,
        }

    def _combine_documents(self, documents: Sequence[Document]) -> str:
        if not documents:
            return "No context retrieved."
        formatted = []
        for idx, doc in enumerate(documents, 1):
            identifier = doc.metadata.get("id") or doc.metadata.get("source") or f"doc-{idx}"
            formatted.append(f"[{identifier}] {doc.page_content.strip()}")
        return "\n---\n".join(formatted)

    def _extract_question(self, example: Example) -> str:
        for key in ("question", "text", "input"):
            value = example.inputs.get(key)
            if isinstance(value, str) and value.strip():
                return value
        raise ValueError(f"Example {example.uid} does not contain a question input")

    def _build_metadata(
        self,
        example: Example,
        documents: Sequence[Document],
        scores: Sequence[float],
    ) -> Dict[str, Any]:
        retrieved: List[Dict[str, Any]] = []
        for idx, (doc, score) in enumerate(zip(documents, scores), 1):
            identifier = doc.metadata.get("id") or doc.metadata.get("source") or f"doc-{idx}"
            retrieved.append(
                {
                    "id": str(identifier),
                    "score": float(score),
                    "text": doc.page_content,
                }
            )

        expected_ids_raw = example.metadata.get("context_ids") if example.metadata else None
        expected_ids = {str(item) for item in expected_ids_raw} if isinstance(expected_ids_raw, (list, tuple, set)) else set()
        matched_ids = [item["id"] for item in retrieved if item["id"] in expected_ids]
        context_recall = (len(matched_ids) / len(expected_ids)) if expected_ids else None

        metadata: Dict[str, Any] = {
            "model_name": self.name,
            "retrieved_documents": retrieved,
            "confidence": float(scores[0]) if scores else 0.0,
        }
        if matched_ids:
            metadata["matched_context_ids"] = matched_ids
        if context_recall is not None:
            metadata["context_recall"] = context_recall
        return metadata

    def predict(self, example: Example) -> ModelResponse:
        if self._pipeline is None:
            raise RuntimeError("The LangChain RAG model must be warmed up before predicting")

        question = self._extract_question(example)
        result = self._pipeline.invoke({"question": question})
        documents = result.get("documents", [])
        scores = result.get("scores", [])
        answer = result.get("answer", self.default_response)

        metadata = self._build_metadata(example, documents, scores)
        return ModelResponse(uid=example.uid, output=answer, metadata=metadata)
class BagOfWordsEmbeddings(Embeddings):
    """Deterministic hashing-based embeddings for small synthetic corpora."""

    def __init__(self, *, dimension: int = 256) -> None:
        self.dimension = max(32, dimension)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed(text) for text in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed(text)

    def _embed(self, text: str) -> List[float]:
        tokens = re.findall(r"\w+", text.lower())
        vector = [0.0] * self.dimension
        for token in tokens:
            index = hash(token) % self.dimension
            vector[index] += 1.0
        norm = math.sqrt(sum(value * value for value in vector)) or 1.0
        return [value / norm for value in vector]

