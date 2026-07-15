from __future__ import annotations

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType
from autogen_core.model_context import BufferedChatCompletionContext
from autogen_core.models import AssistantMessage, SystemMessage, UserMessage

from infolang_autogen import memory as memory_mod
from infolang_autogen.memory import InfoLangMemory, _message_text
from tests.conftest import FakeAsyncInfoLang


def _content(text: str, **metadata: object) -> MemoryContent:
    return MemoryContent(
        content=text,
        mime_type=MemoryMimeType.TEXT,
        metadata=metadata or None,
    )


def test_name_property(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client, name="agent-mem")  # type: ignore[arg-type]
    assert mem.name == "agent-mem"


@pytest.mark.asyncio
async def test_add_stores_and_tracks_id(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client, namespace="ns")  # type: ignore[arg-type]
    await mem.add(_content("a fact"))
    stored = fake_client.remembered[-1]
    assert stored["text"] == "a fact"
    assert stored["namespace"] == "ns"
    assert stored["source"] == "autogen"
    assert mem._added_ids == ["mem_1"]


@pytest.mark.asyncio
async def test_add_uses_metadata_overrides(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client)  # type: ignore[arg-type]
    await mem.add(_content("x", tags="t1,t2", source="notes.md"))
    stored = fake_client.remembered[-1]
    assert stored["tags"] == "t1,t2"
    assert stored["source"] == "notes.md"


@pytest.mark.asyncio
async def test_add_without_id_is_not_tracked() -> None:
    fake = FakeAsyncInfoLang(remember_id=None)
    mem = InfoLangMemory(fake)  # type: ignore[arg-type]
    await mem.add(_content("x"))
    assert mem._added_ids == []


@pytest.mark.asyncio
async def test_query_returns_memory_contents(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client, namespace="ns")  # type: ignore[arg-type]
    result = await mem.query("auth?")
    assert [c.content for c in result.results] == ["alpha fact", "weak beta"]
    first = result.results[0]
    assert first.metadata == {"id": "a", "score": 0.91, "tags": "x"}
    assert fake_client.calls[-1][1]["namespace"] == "ns"


@pytest.mark.asyncio
async def test_query_accepts_memory_content_input(
    fake_client: FakeAsyncInfoLang,
) -> None:
    mem = InfoLangMemory(fake_client)  # type: ignore[arg-type]
    await mem.query(_content("query as content"))
    assert fake_client.calls[-1][1]["query"] == "query as content"


@pytest.mark.asyncio
async def test_query_score_threshold_filters(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client, score_threshold=0.85)  # type: ignore[arg-type]
    result = await mem.query("auth?")
    assert [c.content for c in result.results] == ["alpha fact"]


@pytest.mark.asyncio
async def test_query_top_k_override(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client, top_k=5)  # type: ignore[arg-type]
    await mem.query("auth?", top_k=2)
    assert fake_client.calls[-1][1]["top_k"] == 2


@pytest.mark.asyncio
async def test_update_context_injects_system_message(
    fake_client: FakeAsyncInfoLang,
) -> None:
    mem = InfoLangMemory(fake_client)  # type: ignore[arg-type]
    ctx = BufferedChatCompletionContext(buffer_size=10)
    await ctx.add_message(UserMessage(content="how does auth work?", source="user"))
    result = await mem.update_context(ctx)

    assert len(result.memories.results) == 2
    messages = await ctx.get_messages()
    system = [m for m in messages if isinstance(m, SystemMessage)]
    assert len(system) == 1
    assert "Relevant InfoLang memory" in system[0].content
    assert "1. alpha fact" in system[0].content


@pytest.mark.asyncio
async def test_update_context_no_user_message_is_noop(
    fake_client: FakeAsyncInfoLang,
) -> None:
    mem = InfoLangMemory(fake_client)  # type: ignore[arg-type]
    ctx = BufferedChatCompletionContext(buffer_size=10)
    await ctx.add_message(AssistantMessage(content="hi", source="assistant"))
    result = await mem.update_context(ctx)
    assert result.memories.results == []
    messages = await ctx.get_messages()
    assert not any(isinstance(m, SystemMessage) for m in messages)


@pytest.mark.asyncio
async def test_update_context_no_results_adds_nothing() -> None:
    from infolang import RecallResult

    fake = FakeAsyncInfoLang(recall_result=RecallResult(chunks=[]))
    mem = InfoLangMemory(fake)  # type: ignore[arg-type]
    ctx = BufferedChatCompletionContext(buffer_size=10)
    await ctx.add_message(UserMessage(content="q", source="user"))
    result = await mem.update_context(ctx)
    assert result.memories.results == []
    messages = await ctx.get_messages()
    assert not any(isinstance(m, SystemMessage) for m in messages)


@pytest.mark.asyncio
async def test_update_context_flattens_multimodal_content(
    fake_client: FakeAsyncInfoLang,
) -> None:
    mem = InfoLangMemory(fake_client)  # type: ignore[arg-type]
    ctx = BufferedChatCompletionContext(buffer_size=10)
    await ctx.add_message(UserMessage(content=["hello", "world"], source="user"))
    await mem.update_context(ctx)
    assert fake_client.calls[-1][1]["query"] == "hello world"


@pytest.mark.asyncio
async def test_clear_forgets_tracked_ids(fake_client: FakeAsyncInfoLang) -> None:
    mem = InfoLangMemory(fake_client, namespace="ns")  # type: ignore[arg-type]
    await mem.add(_content("one"))
    await mem.add(_content("two"))
    await mem.clear()
    assert [f["memory_id"] for f in fake_client.forgotten] == ["mem_1", "mem_1"]
    assert mem._added_ids == []


@pytest.mark.asyncio
async def test_close_releases_owned_client(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    fake = FakeAsyncInfoLang()
    monkeypatch.setattr(memory_mod, "resolve_async_client", lambda *a, **k: fake)
    mem = InfoLangMemory(api_key="il_live_test")
    await mem.close()
    assert fake.closed is True


@pytest.mark.asyncio
async def test_close_does_not_release_borrowed_client(
    fake_client: FakeAsyncInfoLang,
) -> None:
    mem = InfoLangMemory(fake_client)  # type: ignore[arg-type]
    await mem.close()
    assert fake_client.closed is False


def test_message_text_helper() -> None:
    assert _message_text("plain") == "plain"
    assert _message_text(["a", "b"]) == "a b"
    # Non-string parts (e.g. images) are dropped.
    assert _message_text(["a", 123, "b"]) == "a b"
