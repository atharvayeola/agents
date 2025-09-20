"""Adapter that proxies model requests over the Model Context Protocol."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Any, AsyncContextManager, Callable, Iterable, Optional

from mcp import McpError
import mcp.types as mcp_types
from mcp.client.session import ClientSession
from mcp.client.sse import sse_client

from eval_agent.models.base import ModelAdapter
from eval_agent.registry import MODEL_REGISTRY
from eval_agent.types import Example, ModelResponse

logger = logging.getLogger(__name__)

SessionFactory = Callable[[], AsyncContextManager[ClientSession]]


def _to_json_serialisable(value: Any) -> Any:
    """Ensure complex values are converted into JSON-friendly structures."""

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, (list, tuple)):
        return [_to_json_serialisable(item) for item in value]
    if isinstance(value, dict):
        return {key: _to_json_serialisable(item) for key, item in value.items()}
    return json.loads(json.dumps(value, default=str))


@MODEL_REGISTRY.register("mcp")
class MCPModelAdapter(ModelAdapter):
    """Model adapter that connects to an MCP server via the SSE transport."""

    def __init__(
        self,
        *,
        endpoint: str,
        model_id: str,
        auth: str | dict[str, Any] | None = None,
        instruction: str | None = None,
        headers: dict[str, str] | None = None,
        transport: str = "sse",
        http_timeout: float = 10.0,
        sse_read_timeout: float = 300.0,
        request_timeout: float | None = None,
        session_factory: SessionFactory | None = None,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if not endpoint:
            raise ValueError("'endpoint' must be provided for the MCP adapter")
        if not model_id:
            raise ValueError("'model_id' must be provided for the MCP adapter")
        if transport.lower() != "sse":
            raise ValueError("Only the 'sse' transport is currently supported")

        self.endpoint = endpoint
        self.model_id = model_id
        self.instruction = instruction
        self.http_timeout = float(http_timeout)
        self.sse_read_timeout = float(sse_read_timeout)
        self.request_timeout = float(request_timeout) if request_timeout is not None else None
        self._custom_session_factory = session_factory
        self._server_info: dict[str, Any] | None = None
        self._ready = False

        self._headers = self._build_headers(auth, headers)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def warmup(self, examples: Iterable[Example] | None = None) -> None:
        try:
            self._run(self._warmup_async())
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise RuntimeError("Unexpected failure during MCP warmup") from exc

    def predict(self, example: Example) -> ModelResponse:
        if not self._ready:
            self._ensure_ready()

        try:
            return self._run(self._predict_async(example))
        except RuntimeError:
            raise
        except Exception as exc:  # pragma: no cover - defensive guardrail
            raise RuntimeError(f"Unexpected MCP failure for example {example.uid}") from exc

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _run(self, coroutine: Any) -> Any:
        return asyncio.run(coroutine)

    def _ensure_ready(self) -> None:
        try:
            self._run(self._warmup_async())
        except RuntimeError:
            raise

    async def _warmup_async(self) -> None:
        async with self._session() as session:
            try:
                tools_result = await session.list_tools()
            except McpError as exc:  # pragma: no cover - network/transport errors
                raise RuntimeError(f"Failed to list tools from MCP server: {exc}") from exc

            available = {tool.name for tool in tools_result.tools}
            if self.model_id not in available:
                raise RuntimeError(
                    f"MCP server at {self.endpoint} does not expose tool '{self.model_id}'."
                )

        self._ready = True

    async def _predict_async(self, example: Example) -> ModelResponse:
        arguments = self._format_arguments(example)

        async with self._session() as session:
            try:
                result = await session.call_tool(self.model_id, arguments)
            except McpError as exc:
                message = f"MCP tool call failed for example {example.uid}: {exc}"
                logger.error(message)
                raise RuntimeError(message) from exc

        if result.isError:
            message = self._format_error_message(result)
            logger.error(message)
            raise RuntimeError(message)

        output, metadata = self._parse_result(result)
        metadata.setdefault("example_uid", example.uid)
        return ModelResponse(uid=example.uid, output=output, metadata=metadata)

    def _format_arguments(self, example: Example) -> dict[str, Any]:
        input_text = self._render_example_text(example)
        message = mcp_types.SamplingMessage(
            role="user",
            content=mcp_types.TextContent(type="text", text=input_text),
        )

        metadata: dict[str, Any] = {"uid": example.uid}
        if example.metadata:
            metadata["example"] = _to_json_serialisable(example.metadata)

        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [message.model_dump(mode="json", by_alias=True)],
            "input": _to_json_serialisable(example.inputs),
            "metadata": metadata,
        }

        return payload

    def _render_example_text(self, example: Example) -> str:
        raw_input = example.inputs.get("text")
        if isinstance(raw_input, str):
            text = raw_input
        else:
            text = json.dumps(_to_json_serialisable(example.inputs), ensure_ascii=False)

        if self.instruction:
            return f"{self.instruction.strip()}\n\n{text}"
        return text

    def _parse_result(self, result: mcp_types.CallToolResult) -> tuple[Any, dict[str, Any]]:
        text_blocks = [
            block.text
            for block in result.content
            if isinstance(block, mcp_types.TextContent)
        ]
        output: Any
        if text_blocks:
            output = "\n".join(text_blocks)
        elif result.structuredContent is not None:
            output = result.structuredContent
        else:
            output = ""

        metadata: dict[str, Any] = {
            "tool": self.model_id,
            "content": [
                block.model_dump(mode="json", by_alias=True)
                for block in result.content
            ],
        }
        if result.structuredContent is not None:
            metadata["structured"] = result.structuredContent
        if self._server_info is not None:
            metadata.setdefault("server_info", self._server_info)

        return output, metadata

    def _format_error_message(self, result: mcp_types.CallToolResult) -> str:
        fragments = [
            block.text
            for block in result.content
            if isinstance(block, mcp_types.TextContent)
        ]
        detail = "\n".join(fragments) if fragments else "(no error message provided)"
        if result.structuredContent:
            detail = f"{detail}\nStructured payload: {result.structuredContent}"
        return f"MCP tool '{self.model_id}' returned an error response: {detail}"

    def _build_headers(
        self,
        auth: str | dict[str, Any] | None,
        headers: dict[str, str] | None,
    ) -> dict[str, str]:
        resolved: dict[str, str] = {}
        if headers:
            resolved.update(headers)

        if auth is None:
            return resolved

        if isinstance(auth, str):
            resolved["Authorization"] = f"Bearer {auth}"
            return resolved

        scheme = str(auth.get("type") or auth.get("scheme") or auth.get("kind") or "bearer").lower()
        if scheme == "bearer":
            token = self._extract_secret(auth, primary="token")
            resolved["Authorization"] = f"Bearer {token}"
            return resolved
        if scheme in {"header", "api_key"}:
            name = auth.get("name")
            if not name:
                raise ValueError("Auth configuration for 'header' requires a 'name'.")
            value = self._extract_secret(auth, primary="value", fallbacks=("token",))
            resolved[str(name)] = value
            return resolved

        raise ValueError(f"Unsupported auth scheme '{scheme}' for MCP adapter")

    def _extract_secret(
        self,
        source: dict[str, Any],
        *,
        primary: str,
        fallbacks: tuple[str, ...] = (),
    ) -> str:
        keys = (primary, *fallbacks)
        for key in keys:
            if key in source:
                return self._coerce_secret(source[key])
            env_key = f"{key}_env"
            if env_key in source:
                return self._read_env_var(str(source[env_key]))
        if "env" in source:
            return self._read_env_var(str(source["env"]))
        raise ValueError(f"No value provided for '{primary}' in auth configuration")

    def _coerce_secret(self, value: Any) -> str:
        if isinstance(value, dict) and "env" in value:
            return self._read_env_var(str(value["env"]))
        return str(value)

    def _read_env_var(self, name: str) -> str:
        try:
            return os.environ[name]
        except KeyError as exc:
            raise RuntimeError(f"Environment variable '{name}' is required for MCP auth") from exc

    @asynccontextmanager
    async def _session(self) -> AsyncIterator[ClientSession]:
        factory = self._custom_session_factory or self._default_session_factory
        async with factory() as session:
            init_result = await session.initialize()
            self._server_info = init_result.serverInfo.model_dump(mode="json", by_alias=True)
            yield session

    def _default_session_factory(self) -> AsyncIterator[ClientSession]:
        return self._sse_session()

    @asynccontextmanager
    async def _sse_session(self) -> AsyncIterator[ClientSession]:
        headers = self._headers or None
        timeout_delta: Optional[timedelta]
        timeout_delta = timedelta(seconds=self.request_timeout) if self.request_timeout else None

        async with sse_client(
            self.endpoint,
            headers=headers,
            timeout=self.http_timeout,
            sse_read_timeout=self.sse_read_timeout,
        ) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream, read_timeout_seconds=timeout_delta) as session:
                yield session
