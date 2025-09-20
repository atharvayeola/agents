"""Tests for the MCP model adapter."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import Any

import pytest
from mcp import McpError
import mcp.types as mcp_types
from mcp.client.session import SUPPORTED_PROTOCOL_VERSIONS

from eval_agent.models.mcp import MCPModelAdapter
from eval_agent.types import Example


class FakeSessionFactory:
    """Lightweight stand-in for an MCP client session."""

    def __init__(
        self,
        *,
        tool_name: str,
        call_tool_result: mcp_types.CallToolResult,
        side_effect: Exception | None = None,
    ) -> None:
        self.tool_name = tool_name
        self.call_tool_result = call_tool_result
        self.side_effect = side_effect
        self.initialize_calls = 0
        self.list_tools_calls = 0
        self.call_tool_calls = 0
        self.last_tool_name: str | None = None
        self.last_arguments: dict[str, Any] | None = None

    @asynccontextmanager
    async def __call__(self):
        factory = self

        class _Session:
            async def initialize(self) -> mcp_types.InitializeResult:
                factory.initialize_calls += 1
                return mcp_types.InitializeResult(
                    protocolVersion=SUPPORTED_PROTOCOL_VERSIONS[0],
                    capabilities=mcp_types.ServerCapabilities(),
                    serverInfo=mcp_types.Implementation(name="stub-server", version="0.1.0"),
                    instructions=None,
                )

            async def list_tools(self, cursor: str | None = None) -> mcp_types.ListToolsResult:
                factory.list_tools_calls += 1
                tool = mcp_types.Tool(name=factory.tool_name, inputSchema={})
                return mcp_types.ListToolsResult(tools=[tool])

            async def call_tool(self, name: str, arguments: dict[str, Any]) -> mcp_types.CallToolResult:
                factory.call_tool_calls += 1
                factory.last_tool_name = name
                factory.last_arguments = arguments
                if factory.side_effect is not None:
                    raise factory.side_effect
                return factory.call_tool_result

        session = _Session()
        yield session


def _text_result(text: str, *, structured: dict[str, Any] | None = None, is_error: bool = False) -> mcp_types.CallToolResult:
    content = [mcp_types.TextContent(type="text", text=text)]
    return mcp_types.CallToolResult(content=content, structuredContent=structured, isError=is_error)


def _example(text: str) -> Example:
    return Example(uid="example-1", inputs={"text": text}, expected_output="positive")


def test_mcp_adapter_warmup_and_predict() -> None:
    factory = FakeSessionFactory(
        tool_name="sentiment-classifier",
        call_tool_result=_text_result("positive", structured={"score": 0.94}),
    )
    adapter = MCPModelAdapter(
        endpoint="http://stub",  # not used by fake session
        model_id="sentiment-classifier",
        instruction="Classify the review as positive, negative, or neutral.",
        session_factory=factory,
    )

    adapter.warmup(None)
    assert factory.initialize_calls == 1
    assert factory.list_tools_calls == 1

    response = adapter.predict(_example("I absolutely loved it."))

    assert factory.initialize_calls == 2  # warmup + predict sessions
    assert factory.call_tool_calls == 1
    assert factory.last_tool_name == "sentiment-classifier"

    arguments = factory.last_arguments
    assert arguments is not None
    assert arguments["model"] == "sentiment-classifier"
    assert arguments["metadata"]["uid"] == "example-1"
    assert "Classify the review" in arguments["messages"][0]["content"]["text"]

    assert response.output == "positive"
    assert response.metadata["structured"] == {"score": 0.94}
    assert response.metadata["tool"] == "sentiment-classifier"
    assert response.metadata["server_info"]["name"] == "stub-server"


def test_mcp_adapter_predict_triggers_warmup() -> None:
    factory = FakeSessionFactory(
        tool_name="demo-tool",
        call_tool_result=_text_result("neutral"),
    )
    adapter = MCPModelAdapter(
        endpoint="http://stub",
        model_id="demo-tool",
        session_factory=factory,
    )

    response = adapter.predict(_example("It's fine."))

    assert response.output == "neutral"
    # Warmup should have been performed automatically.
    assert factory.list_tools_calls == 1
    assert factory.call_tool_calls == 1


def test_mcp_adapter_raises_on_server_error() -> None:
    factory = FakeSessionFactory(
        tool_name="demo-tool",
        call_tool_result=_text_result("failure", structured={"reason": "invalid"}, is_error=True),
    )
    adapter = MCPModelAdapter(
        endpoint="http://stub",
        model_id="demo-tool",
        session_factory=factory,
    )

    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict(_example("bad input"))

    message = str(excinfo.value)
    assert "returned an error" in message
    assert "invalid" in message


def test_mcp_adapter_wraps_transport_errors() -> None:
    error = McpError(mcp_types.ErrorData(code=mcp_types.INTERNAL_ERROR, message="transport boom"))
    factory = FakeSessionFactory(
        tool_name="demo-tool",
        call_tool_result=_text_result("ignored"),
        side_effect=error,
    )
    adapter = MCPModelAdapter(
        endpoint="http://stub",
        model_id="demo-tool",
        session_factory=factory,
    )

    with pytest.raises(RuntimeError) as excinfo:
        adapter.predict(_example("hello"))

    assert "MCP tool call failed" in str(excinfo.value)


def test_mcp_adapter_auth_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("MCP_TOKEN", "secret-token")
    factory = FakeSessionFactory(
        tool_name="demo-tool",
        call_tool_result=_text_result("ok"),
    )

    adapter = MCPModelAdapter(
        endpoint="http://stub",
        model_id="demo-tool",
        auth={"type": "bearer", "token_env": "MCP_TOKEN"},
        session_factory=factory,
    )

    assert adapter._headers["Authorization"] == "Bearer secret-token"
