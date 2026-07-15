"""Shared fixtures: an in-memory fake of the async InfoLang SDK client.

Tests run offline by default. The fake mirrors the subset of the published
``AsyncInfoLang`` surface the adapters use (``recall``, ``investigate``,
``remember``, ``forget``, ``aclose``) and records how it was called.
"""

from __future__ import annotations

from typing import Any

import pytest
from autogen_core import CancellationToken
from autogen_core.tools import FunctionTool
from infolang import Chunk, RecallResult, RememberResult


class FakeAsyncInfoLang:
    """Records calls and returns canned SDK result objects."""

    def __init__(
        self,
        *,
        recall_result: RecallResult | None = None,
        remember_id: str | None = "mem_1",
    ) -> None:
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.remembered: list[dict[str, Any]] = []
        self.forgotten: list[dict[str, Any]] = []
        self.closed = False
        self._recall_result = recall_result
        self._remember_id = remember_id

    def _canned(self) -> RecallResult:
        if self._recall_result is not None:
            return self._recall_result
        return RecallResult(
            chunks=[
                Chunk(i="a", s=0.91, t="alpha fact", g="x"),
                Chunk(i="b", s=0.40, t="weak beta"),
            ],
            namespace="default",
        )

    async def recall(
        self, query: str, **kwargs: Any
    ) -> RecallResult:
        self.calls.append(("recall", {"query": query, **kwargs}))
        return self._canned()

    async def investigate(self, query: str, **kwargs: Any) -> RecallResult:
        self.calls.append(("investigate", {"query": query, **kwargs}))
        return self._canned()

    async def remember(self, text: str, **kwargs: Any) -> RememberResult:
        self.remembered.append({"text": text, **kwargs})
        self.calls.append(("remember", {"text": text, **kwargs}))
        if self._remember_id is None:
            return RememberResult()
        return RememberResult(id=self._remember_id)

    async def forget(self, memory_id: str, **kwargs: Any) -> None:
        self.forgotten.append({"memory_id": memory_id, **kwargs})
        self.calls.append(("forget", {"memory_id": memory_id, **kwargs}))

    async def aclose(self) -> None:
        self.closed = True


@pytest.fixture
def fake_client() -> FakeAsyncInfoLang:
    return FakeAsyncInfoLang()


async def run_tool(tool: FunctionTool, **kwargs: Any) -> Any:
    """Validate args against the tool schema and execute it."""

    args = tool.args_type().model_validate(kwargs)
    return await tool.run(args, CancellationToken())
