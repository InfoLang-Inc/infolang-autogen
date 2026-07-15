"""Client resolution for the AutoGen integration.

Every adapter in this package talks to InfoLang through the **published public
SDK** (`infolang` on PyPI) and nothing else. We never speak HTTP directly or
reach into runtime/engine internals — the SDK is the contract.

AutoGen is async-first, so the integration is built on :class:`AsyncInfoLang`.
"""

from __future__ import annotations

from typing import Any

from infolang import AsyncInfoLang

__all__ = ["resolve_async_client"]


def resolve_async_client(
    client: AsyncInfoLang | None = None,
    *,
    api_key: str | None = None,
    namespace: str | None = None,
    workspace: str | None = None,
    **kwargs: Any,
) -> AsyncInfoLang:
    """Return an :class:`AsyncInfoLang` client for the adapters to use.

    Resolution order:

    1. An explicit ``client`` instance (caller owns its lifecycle).
    2. An ``api_key`` — constructs ``AsyncInfoLang.from_api_key(...)``.
    3. Environment credentials (``INFOLANG_API_KEY`` / ``INFOLANG_DEV_KEY``),
       resolved by the SDK itself.

    ``namespace`` selects the memory bank and ``workspace`` the tenant, matching
    the SDK's scoping model. Extra keyword arguments are forwarded to the SDK
    constructor unchanged.
    """

    if client is not None:
        return client
    if api_key is not None:
        return AsyncInfoLang.from_api_key(
            api_key, namespace=namespace, workspace=workspace, **kwargs
        )
    return AsyncInfoLang(namespace=namespace, workspace=workspace, **kwargs)
