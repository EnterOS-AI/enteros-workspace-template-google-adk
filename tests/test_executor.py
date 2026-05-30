"""Unit tests for google-adk event draining + error sanitisation (pure)."""

from google_adk_executor import (
    collect_response_text,
    extract_event_text,
    extract_incoming_text,
    is_final,
    sanitize_error,
)


class _Ctx:
    """Duck-typed RequestContext whose get_user_input may return text or raise."""
    def __init__(self, value=None, raises=False):
        self._value = value
        self._raises = raises

    def get_user_input(self):
        if self._raises:
            raise RuntimeError("SDK boom")
        return self._value


def _boom(_ctx):
    raise AssertionError("get_user_input fallback must not run when primary has text")


def test_incoming_text_prefers_primary_extractor():
    # primary (extract_message_text) runs first; get_user_input is NOT consulted
    assert extract_incoming_text(_Ctx(raises=True), lambda _c: "from-parts") == "from-parts"


def test_incoming_text_preserves_attachment_manifest():
    # the attachment manifest only the primary emits must survive (regression guard)
    manifest = "What is this?\n\nAttached files:\n- doc.pdf (application/pdf) at /work/doc.pdf"
    assert extract_incoming_text(_Ctx(value="What is this?"), lambda _c: manifest) == manifest


def test_incoming_text_falls_back_to_get_user_input_when_primary_empty():
    # the a2a-sdk-1.1.0 safety net: primary returned "" -> recover text via the SDK
    assert extract_incoming_text(_Ctx(value="  sdk text  "), lambda _c: "") == "sdk text"


def test_incoming_text_fallback_survives_get_user_input_raising():
    assert extract_incoming_text(_Ctx(raises=True), lambda _c: "") == ""


def test_incoming_text_empty_when_both_empty():
    assert extract_incoming_text(_Ctx(value=None), lambda _c: "") == ""


class _Part:
    def __init__(self, text=None):
        self.text = text


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Event:
    def __init__(self, texts=(), final=False, non_text_parts=0):
        parts = [_Part(text=t) for t in texts] + [_Part(text=None) for _ in range(non_text_parts)]
        self.content = _Content(parts)
        self._final = final

    def is_final_response(self):
        return self._final


def test_extract_event_text_skips_non_text_parts():
    assert extract_event_text(_Event(texts=["hello", "world"], non_text_parts=2)) == ["hello", "world"]


def test_extract_event_text_handles_missing_content():
    class _Bare:
        pass
    assert extract_event_text(_Bare()) == []


def test_is_final_detects_terminal_event():
    assert is_final(_Event(final=True)) is True
    assert is_final(_Event(final=False)) is False


def test_collect_returns_final_event_text():
    events = [_Event(texts=["thinking..."]), _Event(texts=["the answer is 42"], final=True)]
    assert collect_response_text(events) == "the answer is 42"


def test_collect_streams_chunks_and_falls_back_when_no_final():
    chunks = []
    out = collect_response_text([_Event(texts=["a"]), _Event(texts=["b"]), _Event(texts=["c"])], on_chunk=chunks.append)
    assert out == "abc" and chunks == ["a", "b", "c"]


def test_collect_final_wins_over_stream():
    assert collect_response_text([_Event(texts=["partial"]), _Event(texts=["FINAL"], final=True)]) == "FINAL"


def test_sanitize_error_is_single_line_and_tagged():
    out = sanitize_error(RuntimeError("boom\nsecret traceback\nmore"))
    assert out.startswith("[A2A_ERROR] ") and "\n" not in out and "secret traceback" not in out


def test_sanitize_error_falls_back_to_class_name():
    class WeirdError(Exception):
        pass
    assert "WeirdError" in sanitize_error(WeirdError(""))
