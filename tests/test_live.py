"""Live smoke tests against the real InfoLang API.

Deselected by default (``addopts = -m 'not live'``). Run explicitly with
credentials in the environment::

    INFOLANG_API_KEY=il_live_... pytest -m live
"""

from __future__ import annotations

import os
import uuid

import pytest
from autogen_core.memory import MemoryContent, MemoryMimeType

from infolang_autogen import InfoLangMemory, create_infolang_tools
from tests.conftest import run_tool

pytestmark = pytest.mark.live

_HAS_CREDS = bool(os.environ.get("INFOLANG_API_KEY") or os.environ.get("INFOLANG_DEV_KEY"))
_NAMESPACE = os.environ.get("INFOLANG_TEST_NAMESPACE", "autogen-live-test")


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_CREDS, reason="no InfoLang credentials in environment")
async def test_memory_round_trip_live() -> None:
    marker = f"autogen-live {uuid.uuid4()}"
    memory = InfoLangMemory(namespace=_NAMESPACE)
    try:
        await memory.add(
            MemoryContent(content=marker, mime_type=MemoryMimeType.TEXT)
        )
        result = await memory.query(marker, top_k=5)
        assert any(marker in item.content for item in result.results)
    finally:
        await memory.clear()
        await memory.close()


@pytest.mark.asyncio
@pytest.mark.skipif(not _HAS_CREDS, reason="no InfoLang credentials in environment")
async def test_tools_round_trip_live() -> None:
    tools = {t.name: t for t in create_infolang_tools(namespace=_NAMESPACE)}
    marker = f"autogen-tool-live {uuid.uuid4()}"
    stored = await run_tool(tools["infolang_remember"], text=marker)
    assert "Stored memory" in stored
    out = await run_tool(tools["infolang_recall"], query=marker, top_k=5)
    assert isinstance(out, str)
