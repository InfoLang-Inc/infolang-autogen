"""An AutoGen :class:`~autogen_core.memory.Memory` provider backed by InfoLang.

Implements the stabilized 0.4+ memory protocol (``add`` / ``query`` /
``update_context`` / ``clear`` / ``close``) so InfoLang can be dropped into an
``AssistantAgent`` as durable, cross-run memory::

    from autogen_agentchat.agents import AssistantAgent
    from infolang_autogen import InfoLangMemory

    memory = InfoLangMemory(api_key="il_live_...", namespace="agent-42")
    agent = AssistantAgent("assistant", model_client=..., memory=[memory])

``add`` maps to :meth:`AsyncInfoLang.remember`, ``query`` to
:meth:`AsyncInfoLang.recall`, and ``update_context`` injects the top matches for
the latest user turn as a system message.
"""

from __future__ import annotations

from typing import Any

from autogen_core import CancellationToken
from autogen_core.memory import (
    Memory,
    MemoryContent,
    MemoryMimeType,
    MemoryQueryResult,
    UpdateContextResult,
)
from autogen_core.model_context import ChatCompletionContext
from autogen_core.models import SystemMessage, UserMessage
from infolang import AsyncInfoLang

from ._client import resolve_async_client

__all__ = ["InfoLangMemory"]


def _message_text(content: str | list[Any]) -> str:
    """Flatten a chat message ``content`` (str or multimodal list) to text."""

    if isinstance(content, str):
        return content
    return " ".join(part for part in content if isinstance(part, str))


class InfoLangMemory(Memory):
    """InfoLang-backed implementation of the AutoGen ``Memory`` protocol."""

    def __init__(
        self,
        client: AsyncInfoLang | None = None,
        *,
        api_key: str | None = None,
        namespace: str | None = None,
        workspace: str | None = None,
        top_k: int = 5,
        score_threshold: float | None = None,
        source: str = "autogen",
        name: str = "infolang",
        **kwargs: Any,
    ) -> None:
        self._client = resolve_async_client(
            client,
            api_key=api_key,
            namespace=namespace,
            workspace=workspace,
            **kwargs,
        )
        # We only close clients we created ourselves.
        self._owns_client = client is None
        self._namespace = namespace
        self._top_k = top_k
        self._score_threshold = score_threshold
        self._source = source
        self._name = name
        self._added_ids: list[str] = []

    @property
    def name(self) -> str:
        """Identifier for this memory instance."""

        return self._name

    async def add(
        self,
        content: MemoryContent,
        cancellation_token: CancellationToken | None = None,
    ) -> None:
        """Persist ``content`` in InfoLang via ``remember``.

        ``content.metadata`` may carry ``tags`` and ``source`` overrides.
        """

        _ = cancellation_token
        metadata = content.metadata or {}
        tags = metadata.get("tags")
        source = metadata.get("source", self._source)
        result = await self._client.remember(
            str(content.content),
            namespace=self._namespace,
            source=source,
            tags=tags,
        )
        if result.memory_id:
            self._added_ids.append(result.memory_id)

    async def query(
        self,
        query: str | MemoryContent = "",
        cancellation_token: CancellationToken | None = None,
        **kwargs: Any,
    ) -> MemoryQueryResult:
        """Semantic ``recall`` over stored memory, returned as memory contents."""

        _ = cancellation_token
        text = query if isinstance(query, str) else str(query.content)
        top_k = kwargs.pop("top_k", self._top_k)
        result = await self._client.recall(
            text, namespace=self._namespace, top_k=top_k, **kwargs
        )
        contents: list[MemoryContent] = []
        for chunk in result.chunks:
            if (
                self._score_threshold is not None
                and chunk.score is not None
                and chunk.score < self._score_threshold
            ):
                continue
            contents.append(
                MemoryContent(
                    content=chunk.text,
                    mime_type=MemoryMimeType.TEXT,
                    metadata={
                        "id": chunk.id,
                        "score": chunk.score,
                        "tags": chunk.tags,
                    },
                )
            )
        return MemoryQueryResult(results=contents)

    async def update_context(
        self,
        model_context: ChatCompletionContext,
    ) -> UpdateContextResult:
        """Inject recalled memory for the latest user turn as a system message."""

        messages = await model_context.get_messages()
        last_user = next(
            (m for m in reversed(messages) if isinstance(m, UserMessage)),
            None,
        )
        if last_user is None:
            return UpdateContextResult(memories=MemoryQueryResult(results=[]))

        query = _message_text(last_user.content)
        recalled = await self.query(query)
        if not recalled.results:
            return UpdateContextResult(memories=recalled)

        lines = [
            f"{i}. {str(item.content)}"
            for i, item in enumerate(recalled.results, 1)
        ]
        block = (
            "\nRelevant InfoLang memory (most similar first):\n"
            + "\n".join(lines)
            + "\n"
        )
        await model_context.add_message(SystemMessage(content=block))
        return UpdateContextResult(memories=recalled)

    async def clear(self) -> None:
        """Forget every memory this instance stored via :meth:`add`.

        Scoped to this instance rather than the whole namespace, which would be
        destructive. Memories written by other clients are left untouched.
        """

        for memory_id in self._added_ids:
            await self._client.forget(memory_id, namespace=self._namespace)
        self._added_ids.clear()

    async def close(self) -> None:
        """Release the underlying client if this instance created it."""

        if self._owns_client:
            await self._client.aclose()
