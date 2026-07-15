from __future__ import annotations

from infolang import Chunk, RecallResult

from infolang_autogen._format import NO_RESULTS, format_recall


def _result(*chunks: Chunk) -> RecallResult:
    return RecallResult(chunks=list(chunks))


def test_empty_result_returns_placeholder() -> None:
    assert format_recall(_result()) == NO_RESULTS


def test_numbered_lines_with_scores() -> None:
    out = format_recall(
        _result(Chunk(i="a", s=0.91, t="first"), Chunk(i="b", s=0.88, t="second"))
    )
    assert out == "1. [0.91] first\n2. [0.88] second"


def test_include_scores_false_omits_brackets() -> None:
    out = format_recall(
        _result(Chunk(i="a", s=0.91, t="first")), include_scores=False
    )
    assert out == "1. first"


def test_weak_match_is_annotated() -> None:
    out = format_recall(_result(Chunk(i="a", s=0.40, t="weak")))
    assert "weak match" in out
    assert out.startswith("1. [0.40] weak")


def test_strong_match_has_no_weak_annotation() -> None:
    out = format_recall(_result(Chunk(i="a", s=0.95, t="strong")))
    assert "weak match" not in out


def test_chunk_without_score_has_no_bracket() -> None:
    out = format_recall(_result(Chunk(i="a", t="no score")))
    assert out == "1. no score"
