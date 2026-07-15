from __future__ import annotations

import pytest
from infolang import AsyncInfoLang

from infolang_autogen._client import resolve_async_client
from tests.conftest import FakeAsyncInfoLang


def test_explicit_client_is_returned_unchanged(fake_client: FakeAsyncInfoLang) -> None:
    assert resolve_async_client(fake_client) is fake_client  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_api_key_builds_client_with_scoping() -> None:
    client = resolve_async_client(
        api_key="il_live_test", namespace="ns", workspace="ws"
    )
    assert isinstance(client, AsyncInfoLang)
    assert client.namespace == "ns"
    assert client.workspace == "ws"
    await client.aclose()


@pytest.mark.asyncio
async def test_env_credentials_are_used_when_no_api_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("INFOLANG_API_KEY", "il_live_env")
    client = resolve_async_client(namespace="from-arg")
    assert isinstance(client, AsyncInfoLang)
    assert client.namespace == "from-arg"
    await client.aclose()


@pytest.mark.asyncio
async def test_extra_kwargs_forwarded_to_sdk() -> None:
    client = resolve_async_client(
        api_key="il_live_test", base_url="https://example.test"
    )
    assert client._base_url == "https://example.test"
    await client.aclose()


def test_missing_credentials_raise(monkeypatch: pytest.MonkeyPatch) -> None:
    from infolang import InfoLangConfigError

    for var in ("INFOLANG_API_KEY", "INFOLANG_DEV_KEY", "INFOLANG_BASE_URL"):
        monkeypatch.delenv(var, raising=False)
    with pytest.raises(InfoLangConfigError):
        resolve_async_client()
