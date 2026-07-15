"""InfoLang integration for AutoGen (``autogen-agentchat`` >= 0.4).

Two adapters, both built on the published ``infolang`` SDK:

* :func:`create_infolang_tools` — the four InfoLang operations as
  :class:`~autogen_core.tools.FunctionTool` instances for tool-using agents.
* :class:`InfoLangMemory` — an implementation of AutoGen's ``Memory`` protocol
  for durable, cross-run agent memory.

Quickstart::

    from autogen_agentchat.agents import AssistantAgent
    from infolang_autogen import InfoLangMemory, create_infolang_tools

    memory = InfoLangMemory(api_key="il_live_...", namespace="agent-42")
    tools = create_infolang_tools(api_key="il_live_...", namespace="agent-42")
    agent = AssistantAgent(
        "assistant", model_client=..., tools=tools, memory=[memory]
    )
"""

from __future__ import annotations

from ._version import __version__
from .memory import InfoLangMemory
from .tools import (
    create_forget_tool,
    create_infolang_tools,
    create_investigate_tool,
    create_recall_tool,
    create_remember_tool,
)

__all__ = [
    "__version__",
    "InfoLangMemory",
    "create_infolang_tools",
    "create_recall_tool",
    "create_investigate_tool",
    "create_remember_tool",
    "create_forget_tool",
]
