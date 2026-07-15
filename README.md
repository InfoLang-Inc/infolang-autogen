# infolang-autogen

InfoLang semantic memory for [AutoGen](https://github.com/microsoft/autogen)
(`autogen-agentchat` >= 0.4, the rewritten API). Give your agents durable,
cross-run memory backed by the [InfoLang](https://infolang.ai) runtime.

Built entirely on the published [`infolang`](https://pypi.org/project/infolang/)
Python SDK — no HTTP, no runtime internals.

## Install

```bash
pip install infolang-autogen
```

This pulls in `infolang` and `autogen-agentchat`/`autogen-core`.

## Two ways to use it

### 1. Memory tools (`FunctionTool`s)

Expose the four InfoLang operations — **recall**, **investigate**, **remember**,
**forget** — as tools an `AssistantAgent` can call:

```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient
from infolang_autogen import create_infolang_tools

tools = create_infolang_tools(api_key="il_live_...", namespace="agent-42")

agent = AssistantAgent(
    "assistant",
    model_client=OpenAIChatCompletionClient(model="gpt-4o"),
    tools=tools,
)
```

`create_infolang_tools` returns `[infolang_recall, infolang_investigate,
infolang_remember, infolang_forget]`. You can also build them individually with
`create_recall_tool`, `create_investigate_tool`, `create_remember_tool` and
`create_forget_tool`.

### 2. Memory provider (`Memory` protocol)

`InfoLangMemory` implements AutoGen's `Memory` protocol, so it plugs directly
into an agent's `memory=[...]`. On each turn, `update_context` recalls the most
relevant stored chunks for the latest user message and injects them as a system
message.

```python
from autogen_agentchat.agents import AssistantAgent
from infolang_autogen import InfoLangMemory

memory = InfoLangMemory(api_key="il_live_...", namespace="agent-42")

agent = AssistantAgent(
    "assistant",
    model_client=...,
    memory=[memory],
)
```

| Protocol method | InfoLang operation |
|-----------------|--------------------|
| `add(content)` | `remember` |
| `query(query)` | `recall` |
| `update_context(ctx)` | `recall` (injects a system message) |
| `clear()` | `forget` (only memories added via this instance) |
| `close()` | releases the client (if this instance created it) |

## Authentication

Pass `api_key=` explicitly, hand in a pre-built `AsyncInfoLang` client
(`client=`), or rely on environment credentials (`INFOLANG_API_KEY`,
`INFOLANG_DEV_KEY`, `INFOLANG_NAMESPACE`, `INFOLANG_WORKSPACE`). Credentials are
resolved by the SDK.

## Namespacing (per-agent memory)

`namespace` selects the memory **bank** and `workspace` the **tenant**. For
multi-agent or multi-user deployments, give each agent (or user) its own
namespace so recalls stay isolated:

```python
memory = InfoLangMemory(api_key=key, namespace=f"agent-{agent_id}")
```

Managed API-key requests honor `namespace` on both reads and writes.

## Development

```bash
pip install -e ".[dev]"
ruff check .
mypy
pytest
```

Tests run offline against a mocked SDK client. The live smoke tests are marked
`live` and deselected by default; run them with real credentials:

```bash
INFOLANG_API_KEY=il_live_... pytest -m live
```

## License

Apache-2.0.
