from __future__ import annotations

import pytest
from autogen_core.tools import FunctionTool

from infolang_autogen import tools as tools_mod
from infolang_autogen.tools import (
    create_forget_tool,
    create_infolang_tools,
    create_investigate_tool,
    create_recall_tool,
    create_remember_tool,
)
from tests.conftest import FakeAsyncInfoLang, run_tool


@pytest.mark.asyncio
async def test_recall_tool_runs_and_formats(fake_client: FakeAsyncInfoLang) -> None:
    tool = create_recall_tool(fake_client, namespace="ns")  # type: ignore[arg-type]
    out = await run_tool(tool, query="how does auth work?", top_k=3)
    assert "1. [0.91] alpha fact" in out
    name, kwargs = fake_client.calls[-1]
    assert name == "recall"
    assert kwargs["namespace"] == "ns"
    assert kwargs["top_k"] == 3


def test_recall_tool_defaults(fake_client: FakeAsyncInfoLang) -> None:
    tool = create_recall_tool(fake_client, default_top_k=7)  # type: ignore[arg-type]
    assert tool.name == "infolang_recall"
    assert "search" in tool.description.lower()
    props = tool.schema["parameters"]["properties"]  # type: ignore[index]
    assert props["top_k"]["default"] == 7
    assert set(props) == {"query", "top_k"}


@pytest.mark.asyncio
async def test_recall_tool_uses_default_top_k(fake_client: FakeAsyncInfoLang) -> None:
    tool = create_recall_tool(fake_client, default_top_k=9)  # type: ignore[arg-type]
    await run_tool(tool, query="q")
    assert fake_client.calls[-1][1]["top_k"] == 9


@pytest.mark.asyncio
async def test_investigate_tool(fake_client: FakeAsyncInfoLang) -> None:
    tool = create_investigate_tool(fake_client, namespace="ns", top_k=4)  # type: ignore[arg-type]
    out = await run_tool(tool, query="what shipped?")
    assert out.startswith("1. [0.91]")
    name, kwargs = fake_client.calls[-1]
    assert name == "investigate"
    assert kwargs["namespace_hint"] == "ns"
    assert kwargs["top_k"] == 4


def test_investigate_schema_has_only_query(fake_client: FakeAsyncInfoLang) -> None:
    tool = create_investigate_tool(fake_client)  # type: ignore[arg-type]
    assert set(tool.schema["parameters"]["properties"]) == {"query"}  # type: ignore[index]


@pytest.mark.asyncio
async def test_remember_tool_returns_id_and_defaults(
    fake_client: FakeAsyncInfoLang,
) -> None:
    tool = create_remember_tool(fake_client, namespace="ns")  # type: ignore[arg-type]
    out = await run_tool(tool, text="a durable fact")
    assert out == "Stored memory mem_1."
    stored = fake_client.remembered[-1]
    assert stored["text"] == "a durable fact"
    assert stored["namespace"] == "ns"
    assert stored["source"] == "autogen"
    assert stored["tags"] is None


@pytest.mark.asyncio
async def test_remember_tool_passes_tags_and_source(
    fake_client: FakeAsyncInfoLang,
) -> None:
    tool = create_remember_tool(fake_client)  # type: ignore[arg-type]
    await run_tool(tool, text="x", tags="a,b", source="docs/y.md")
    stored = fake_client.remembered[-1]
    assert stored["tags"] == "a,b"
    assert stored["source"] == "docs/y.md"


@pytest.mark.asyncio
async def test_remember_tool_without_id() -> None:
    fake = FakeAsyncInfoLang(remember_id=None)
    tool = create_remember_tool(fake)  # type: ignore[arg-type]
    out = await run_tool(tool, text="x")
    assert out == "Stored memory."


@pytest.mark.asyncio
async def test_remember_tool_custom_default_source() -> None:
    fake = FakeAsyncInfoLang()
    tool = create_remember_tool(fake, default_source="crew")  # type: ignore[arg-type]
    await run_tool(tool, text="x")
    assert fake.remembered[-1]["source"] == "crew"


@pytest.mark.asyncio
async def test_forget_tool(fake_client: FakeAsyncInfoLang) -> None:
    tool = create_forget_tool(fake_client, namespace="ns")  # type: ignore[arg-type]
    out = await run_tool(tool, memory_id="mem_9")
    assert out == "Forgot memory mem_9."
    assert fake_client.forgotten[-1] == {"memory_id": "mem_9", "namespace": "ns"}


def test_create_infolang_tools_returns_four(fake_client: FakeAsyncInfoLang) -> None:
    tools = create_infolang_tools(fake_client, namespace="ns")  # type: ignore[arg-type]
    assert [t.name for t in tools] == [
        "infolang_recall",
        "infolang_investigate",
        "infolang_remember",
        "infolang_forget",
    ]
    assert all(isinstance(t, FunctionTool) for t in tools)


def test_create_infolang_tools_via_api_key(
    monkeypatch: pytest.MonkeyPatch, fake_client: FakeAsyncInfoLang
) -> None:
    monkeypatch.setattr(
        tools_mod, "resolve_async_client", lambda *a, **k: fake_client
    )
    tools = create_infolang_tools(api_key="il_live_test", namespace="ns")
    assert len(tools) == 4


@pytest.mark.asyncio
async def test_include_scores_false_propagates(
    fake_client: FakeAsyncInfoLang,
) -> None:
    tools = create_infolang_tools(
        fake_client, namespace="ns", include_scores=False  # type: ignore[arg-type]
    )
    recall_tool = tools[0]
    out = await run_tool(recall_tool, query="q")
    assert "[0.91]" not in out
    assert "1. alpha fact" in out


def test_remember_and_forget_schemas(fake_client: FakeAsyncInfoLang) -> None:
    remember = create_remember_tool(fake_client)  # type: ignore[arg-type]
    forget = create_forget_tool(fake_client)  # type: ignore[arg-type]
    assert set(remember.schema["parameters"]["properties"]) == {  # type: ignore[index]
        "text",
        "tags",
        "source",
    }
    assert set(forget.schema["parameters"]["properties"]) == {"memory_id"}  # type: ignore[index]
