"""infolang-autogen quickstart.

Wires InfoLang memory into an AutoGen ``AssistantAgent`` two ways: as tools and
as a ``Memory`` provider. Requires a model client and credentials::

    export INFOLANG_API_KEY=il_live_...
    export OPENAI_API_KEY=sk-...
    python examples/quickstart.py
"""

from __future__ import annotations

import asyncio

from autogen_core.memory import MemoryContent, MemoryMimeType

from infolang_autogen import InfoLangMemory, create_infolang_tools


async def main() -> None:
    namespace = "autogen-quickstart"

    # 1. The four operations as agent tools.
    tools = create_infolang_tools(namespace=namespace)
    print(f"created {len(tools)} tools: {[t.name for t in tools]}")

    # 2. Durable memory the agent reads from and writes to across runs.
    memory = InfoLangMemory(namespace=namespace)
    await memory.add(
        MemoryContent(
            content="The user prefers metric units.",
            mime_type=MemoryMimeType.TEXT,
        )
    )
    hits = await memory.query("what units does the user prefer?")
    for item in hits.results:
        print(item.content)
    await memory.close()

    # Hand `tools` and `[memory]` to an AssistantAgent:
    #   from autogen_agentchat.agents import AssistantAgent
    #   agent = AssistantAgent("assistant", model_client=..., tools=tools,
    #                          memory=[memory])


if __name__ == "__main__":
    asyncio.run(main())
