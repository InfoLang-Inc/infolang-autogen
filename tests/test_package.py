from __future__ import annotations

import infolang_autogen


def test_version_is_populated() -> None:
    assert isinstance(infolang_autogen.__version__, str)
    assert infolang_autogen.__version__


def test_public_exports_are_importable() -> None:
    for name in infolang_autogen.__all__:
        assert hasattr(infolang_autogen, name)


def test_key_symbols_present() -> None:
    assert infolang_autogen.InfoLangMemory is not None
    assert callable(infolang_autogen.create_infolang_tools)
