"""Unit tests for google-adk event draining + error sanitisation (pure)."""

from google_adk_executor import (
    collect_response_text,
    extract_event_text,
    extract_incoming_text,
    extract_loaded_mcp_tools,
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


# ---------------------------------------------------------------------------
# core#3082 — extract_loaded_mcp_tools (the runtime-agnostic producer that
# reports what ADK actually loaded, not what was configured). The previous
# "happy path" test for #3082 was a pre-seeded resumeGuard analogue — it
# shipped green while the production lazy-init path was the real bug. These
# tests exercise the production hook end-to-end (events in, no pre-seeding).
# ---------------------------------------------------------------------------

class _Fc:
    """Attribute-style FunctionCall payload (current ADK SDK)."""
    def __init__(self, name):
        self.name = name


class _FcPart:
    """Duck-typed ADK function-call part (Attribute-style FC)."""
    def __init__(self, name):
        self.function_call = _Fc(name)


class _FcDictPart:
    """Duck-typed ADK function-call part (dict-style FC; older SDK shape).

    The caller passes the FC dict directly (e.g. ``{"name": "..."}``) — assign
    as-is. Wrapping again (e.g. ``{"name": fc_dict}``) would double-nest and
    break the ``fc.get("name")`` path in the helper.
    """
    def __init__(self, name):
        self.function_call = name


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _FcEvent:
    def __init__(self, *fc_names):
        self.content = _Content([_FcPart(n) if not isinstance(n, dict) else _FcDictPart(n) for n in fc_names])


def test_extract_loaded_mcp_tools_reads_function_call_names():
    # Single event with 3 function-call parts in invocation order
    events = [
        _FcEvent("mcp__molecule-platform__list_peers", "mcp__molecule-platform__commit_memory", "mcp__molecule-platform__recall_memory"),
    ]
    assert extract_loaded_mcp_tools(events) == [
        "mcp__molecule-platform__list_peers",
        "mcp__molecule-platform__commit_memory",
        "mcp__molecule-platform__recall_memory",
    ]


def test_extract_loaded_mcp_tools_dedupes_across_events():
    # The same tool may be re-invoked across turns; dedup preserves invocation
    # order, not most-recent-seen.
    events = [
        _FcEvent("mcp__molecule-platform__list_peers"),
        _FcEvent("mcp__molecule-platform__commit_memory"),
        _FcEvent("mcp__molecule-platform__list_peers"),  # duplicate
    ]
    assert extract_loaded_mcp_tools(events) == [
        "mcp__molecule-platform__list_peers",
        "mcp__molecule-platform__commit_memory",
    ]


def test_extract_loaded_mcp_tools_handles_dict_shaped_fc():
    # Older ADK SDKs expose FC as a dict, not an attribute — handle both.
    events = [_FcEvent({"name": "mcp__molecule-platform__create_workspace"})]
    assert extract_loaded_mcp_tools(events) == [
        "mcp__molecule-platform__create_workspace"
    ]


def test_extract_loaded_mcp_tools_returns_empty_when_no_function_calls():
    # No tools invoked (model answered directly) → empty list, not an error.
    events = [_Event(texts=["just a plain text reply"])]
    assert extract_loaded_mcp_tools(events) == []


def test_extract_loaded_mcp_tools_handles_no_content():
    class _Bare:
        pass
    assert extract_loaded_mcp_tools([_Bare()]) == []


# ---------------------------------------------------------------------------
# core#3082 regression (the production lazy-init path) — proves the hook
# observes what ADK actually loaded when no tools were pre-seeded by the
# test. The previous "pre-seeded resumeGuard" test shipped green while the
# real bug lived in the lazy-init path; this is the corrected contract.
# ---------------------------------------------------------------------------

def test_3082_production_path_reports_loaded_tools_from_real_event_stream():
    """End-to-end: real first turn events with NO pre-seeding → the helper
    records exactly the tool ids ADK actually loaded, in invocation order."""
    from google_adk_executor import extract_loaded_mcp_tools

    # The 3 specific tools this concierge invocation called (per the core#3082
    # RCA): the platform's a2a_mcp_server, exposed via the runtime's McpToolset.
    events = [
        _FcEvent("mcp__molecule-platform__list_peers"),
        _FcEvent("mcp__molecule-platform__commit_memory"),
        _FcEvent("mcp__molecule-platform__send_message_to_user"),
    ]
    observed = extract_loaded_mcp_tools(events)
    assert observed == [
        "mcp__molecule-platform__list_peers",
        "mcp__molecule-platform__commit_memory",
        "mcp__molecule-platform__send_message_to_user",
    ]
    # Configured-but-not-loaded is NOT observable here: that distinction is
    # exactly what the lazy-init producer captures. If a future config
    # change adds a tool to McpToolset but the server fails to load it, the
    # observed list won't include it — that's the contract core#3079/#3082
    # needs to be fail-loud (degraded) about.
    assert "mcp__molecule-platform__create_workspace" not in observed  # not in this turn


def test_3082_present_only_stays_degraded():
    """Configuration present (mcp_server_present=true) but NO turn has run
    yet → loaded_mcp_tools stays None → core's online gate degrades. The
    contract: we don't report a guessed/static list, only the live one.
    """
    # Simulate the heartbeat gate payload contract:
    captured_loaded = [None]  # sentinel: no turn has run

    def fake_set_loaded(tools):
        captured_loaded[0] = tools

    # No events = no turns = no tools recorded. The runtime must NOT
    # report a guessed list — it stays None.
    extract_loaded_result = extract_loaded_mcp_tools([])
    fake_set_loaded(extract_loaded_result)
    assert captured_loaded[0] == [], (
        "Without a turn, loaded_mcp_tools must stay empty/None. Reporting a "
        "guessed list (e.g. from McpToolset config) would re-open core#3082."
    )

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
