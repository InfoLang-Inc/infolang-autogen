"""InfoLang memory tools for AutoGen (``autogen-agentchat`` >= 0.4).

Each of the four InfoLang operations — ``recall``, ``investigate``,
``remember`` and ``forget`` — is exposed as an :class:`~autogen_core.tools.FunctionTool`
so it can be handed to an ``AssistantAgent`` (or any tool-using agent) verbatim::

    from autogen_agentchat.agents import AssistantAgent
    from infolang_autogen import create_infolang_tools

    tools = create_infolang_tools(api_key="il_live_...", namespace="agent-42")
    agent = AssistantAgent("assistant", model_client=..., tools=tools)

The tool coroutines call the published :class:`AsyncInfoLang` SDK; they never
touch HTTP or runtime internals.
"""

from __future__ import annotations

from typing import Any

from autogen_core.tools import FunctionTool
from infolang import AsyncInfoLang

from ._client import resolve_async_client
from ._format import format_recall

__all__ = [
    "create_recall_tool",
    "create_investigate_tool",
    "create_remember_tool",
    "create_forget_tool",
    "create_infolang_tools",
]


def create_recall_tool(
    client: AsyncInfoLang,
    *,
    namespace: str | None = None,
    default_top_k: int = 5,
    include_scores: bool = True,
    name: str = "infolang_recall",
    description: str = (
        "Search long-term InfoLang memory for context relevant to a query and "
        "return the most similar stored chunks."
    ),
) -> FunctionTool:
    """Build the ``recall`` tool: semantic search over stored memory."""

    async def recall(query: str, top_k: int = default_top_k) -> str:
        result = await client.recall(query, namespace=namespace, top_k=top_k)
        return format_recall(result, include_scores=include_scores)

    return FunctionTool(recall, description=description, name=name)


def create_investigate_tool(
    client: AsyncInfoLang,
    *,
    namespace: str | None = None,
    top_k: int = 5,
    include_scores: bool = True,
    name: str = "infolang_investigate",
    description: str = (
        "Investigate a question against InfoLang memory (agent-style recall with "
        "a sensible default depth). Use before answering questions about prior work."
    ),
) -> FunctionTool:
    """Build the ``investigate`` tool: agent-style recall with a default depth."""

    async def investigate(query: str) -> str:
        result = await client.investigate(
            query, namespace_hint=namespace, top_k=top_k
        )
        return format_recall(result, include_scores=include_scores)

    return FunctionTool(investigate, description=description, name=name)


def create_remember_tool(
    client: AsyncInfoLang,
    *,
    namespace: str | None = None,
    default_source: str | None = "autogen",
    name: str = "infolang_remember",
    description: str = (
        "Store a fact or note in long-term InfoLang memory so it can be recalled "
        "in future conversations. Returns the new memory id."
    ),
) -> FunctionTool:
    """Build the ``remember`` tool: persist a memory."""

    async def remember(
        text: str, tags: str | None = None, source: str | None = None
    ) -> str:
        result = await client.remember(
            text,
            namespace=namespace,
            source=source or default_source,
            tags=tags,
        )
        if result.memory_id:
            return f"Stored memory {result.memory_id}."
        return "Stored memory."

    return FunctionTool(remember, description=description, name=name)


def create_forget_tool(
    client: AsyncInfoLang,
    *,
    namespace: str | None = None,
    name: str = "infolang_forget",
    description: str = (
        "Delete a memory from InfoLang by its id. Use to remove outdated or "
        "incorrect stored facts."
    ),
) -> FunctionTool:
    """Build the ``forget`` tool: delete a memory by id."""

    async def forget(memory_id: str) -> str:
        await client.forget(memory_id, namespace=namespace)
        return f"Forgot memory {memory_id}."

    return FunctionTool(forget, description=description, name=name)


def create_infolang_tools(
    client: AsyncInfoLang | None = None,
    *,
    api_key: str | None = None,
    namespace: str | None = None,
    workspace: str | None = None,
    default_top_k: int = 5,
    include_scores: bool = True,
    **kwargs: Any,
) -> list[FunctionTool]:
    """Build all four InfoLang tools bound to a single client.

    Pass an existing ``client`` (whose lifecycle you own), an ``api_key``, or
    rely on environment credentials. ``namespace`` scopes the memory bank and
    ``workspace`` the tenant. Extra keyword arguments flow to the SDK client.
    """

    resolved = resolve_async_client(
        client,
        api_key=api_key,
        namespace=namespace,
        workspace=workspace,
        **kwargs,
    )
    return [
        create_recall_tool(
            resolved,
            namespace=namespace,
            default_top_k=default_top_k,
            include_scores=include_scores,
        ),
        create_investigate_tool(
            resolved,
            namespace=namespace,
            top_k=default_top_k,
            include_scores=include_scores,
        ),
        create_remember_tool(resolved, namespace=namespace),
        create_forget_tool(resolved, namespace=namespace),
    ]
