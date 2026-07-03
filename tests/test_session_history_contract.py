"""Session-id derivation + NO-history-injection regression tests (tenant-agent BUG 3).

``GoogleADKA2AExecutor`` now INHERITS the session contract from the shared
``SubprocessA2AExecutor`` base (``molecule_runtime.subprocess_executor``):

  * the ADK session id (passed to ``Runner.run_async(session_id=...)``) is
    derived from the STABLE workspace identity (``workspace:<WORKSPACE_ID>``),
    NOT the per-request ``context_id`` the a2a-sdk mints fresh each turn — using
    ``context_id`` opened a new ``InMemorySessionService`` session every message;
    ``context_id`` / ``task_id`` / ``"default"`` remain fallbacks only when no
    WORKSPACE_ID is available; and
  * continuity is that native ADK session, resumed via the stable id — the base
    passes ONLY the current user message to ``run_agent`` and does NOT force-inject
    ``metadata.history`` into the ADK ``user`` content. (Older/other history is
    retrieved only if the agent CHOOSES to call the ``get_conversation_history``
    MCP tool — never shoved into every turn.)

These tests drive the REAL base ``execute()`` through a thin fake ADK ``Runner``
that captures the ``run_async`` kwargs — no google-adk, no key, no network. They
mirror the base's own ``test_subprocess_executor_contract.py`` (which asserts the
same stable-id + no-inject contract on a bare stub subclass) and
template-openclaw #139's session test (which captures the ``openclaw agent`` CLI
args instead of the Runner kwargs).

The base ships with molecule-ai-workspace-runtime, so ``importorskip`` it: this
module runs against the real runtime (CI installs it) and skips cleanly when the
runtime is absent or too old to carry the base (pre runtime #222).
"""
from __future__ import annotations

import sys
import types as _t
from types import SimpleNamespace

import pytest

pytest.importorskip(
    "molecule_runtime.subprocess_executor",
    reason=(
        "shared SubprocessA2AExecutor base ships with molecule-ai-workspace-runtime "
        "(runtime #222); skip when the runtime is absent/too old"
    ),
)


@pytest.fixture(autouse=True)
def _isolate_workspace_id(monkeypatch):
    """Deterministic WORKSPACE_ID resolution regardless of test order.

    ``platform_auth.get_workspace_id`` caches the validated WORKSPACE_ID in a
    module global on first read; an EARLIER test (here or in the base suite) can
    populate that cache and the ambient env with a different value. The
    derive_session_id fallback + env-read assertions below both flow through that
    cache, so reset it and clear the ambient env before/after each test — otherwise
    a leaked value shadows the per-test setup (the order-dependent red the base's
    own contract test also guards against, runtime commit 8afd207).
    """
    from molecule_runtime import platform_auth

    platform_auth._reset_workspace_id_cache()
    monkeypatch.delenv("WORKSPACE_ID", raising=False)
    yield
    platform_auth._reset_workspace_id_cache()


# --------------------------------------------------------------------------- #
# google.genai is installed only inside the runtime image (separate PyPI index),
# not in the unit-test env. run_agent imports google.genai.types lazily; stub it
# so the Content/Part the executor builds carry the task text we assert on.
# --------------------------------------------------------------------------- #
def _install_genai_stub() -> None:
    if "google.genai.types" in sys.modules:
        return
    genai = sys.modules.get("google.genai") or _t.ModuleType("google.genai")
    gtypes = _t.ModuleType("google.genai.types")

    class _Content:
        def __init__(self, role=None, parts=None):
            self.role = role
            self.parts = parts or []

    class _Part:
        def __init__(self, text=None):
            self.text = text

    gtypes.Content = _Content
    gtypes.Part = _Part
    genai.types = gtypes
    sys.modules.setdefault("google", _t.ModuleType("google"))
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = gtypes


_install_genai_stub()

from google_adk_executor import GoogleADKA2AExecutor  # noqa: E402
from molecule_runtime.subprocess_executor import SubprocessA2AExecutor  # noqa: E402


# --------------------------------------------------------------------------- #
# Fakes for the ADK Runner / session service / event stream / A2A context+queue.
# --------------------------------------------------------------------------- #
class _ADKPart:
    def __init__(self, text):
        self.text = text


class _ADKContent:
    def __init__(self, parts):
        self.parts = parts


class _ADKEvent:
    """Duck-typed ADK event: ``.content.parts[*].text`` + ``.is_final_response()``."""

    def __init__(self, text, final=True):
        self.content = _ADKContent([_ADKPart(text)])
        self._final = final

    def is_final_response(self):
        return self._final


class _SessionService:
    async def get_session(self, **_k):
        return object()  # session exists → no create

    async def create_session(self, **_k):
        return object()


class _Runner:
    """Captures each ``run_async`` call so tests can assert on session_id + text."""

    def __init__(self, events=None):
        self.session_service = _SessionService()
        self._events = events if events is not None else [_ADKEvent("ok", final=True)]
        self.calls: list[dict] = []

    def run_async(self, *, user_id, session_id, new_message):
        self.calls.append(
            {"user_id": user_id, "session_id": session_id, "new_message": new_message}
        )

        async def _gen():
            for e in self._events:
                yield e

        return _gen()


class _Tool:
    def __init__(self, name):
        self.name = name


class _McpToolset:
    def __init__(self, *names):
        self._names = names

    def get_tools(self):
        return [_Tool(n) for n in self._names]


class _RecordingQueue:
    def __init__(self):
        self.events = []

    async def enqueue_event(self, event):
        self.events.append(event)


def _build_context(*, text="hi", history=None, context_id=None, task_id=None):
    """Minimal A2A RequestContext shape the shared helpers read.

    Mirrors the base's own contract-test fakes: text-only dict parts (so
    ``extract_attached_files`` returns []), and ``history`` on both
    ``request.metadata`` and ``metadata`` (where ``extract_history`` looks).
    """
    msg = SimpleNamespace(parts=[{"text": text}], metadata={})
    return SimpleNamespace(
        message=msg,
        request=SimpleNamespace(metadata={"history": history or []}),
        metadata={"history": history or []},
        context_id=context_id,
        task_id=task_id,
    )


def _history(*pairs):
    return [{"role": role, "parts": [{"text": text}]} for role, text in pairs]


def _make_executor(runner, *, workspace_id="", tools=None):
    return GoogleADKA2AExecutor(
        runner,
        app_name="molecule",
        user_id="u1",
        model="platform:gemini",
        heartbeat=None,
        tools=tools if tools is not None else [_McpToolset("list_peers")],
        workspace_id=workspace_id,
    )


async def _run(ex, ctx):
    q = _RecordingQueue()
    await ex.execute(ctx, q)
    return q


def _session_of(runner, i=0):
    return runner.calls[i]["session_id"]


def _text_of(runner, i=0):
    return runner.calls[i]["new_message"].parts[0].text


# --------------------------------------------------------------------------- #
# CONTRACT #1 — stable, workspace-keyed session id (the core BUG-3 fix).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_session_id_is_workspace_keyed_and_stable():
    """The ADK session id is workspace-keyed and STABLE across fresh context_ids."""
    runner = _Runner()
    ex = _make_executor(runner, workspace_id="ws-stable-1")

    await _run(ex, _build_context(text="turn 1", context_id="ctx-aaaa"))
    await _run(ex, _build_context(text="turn 2", context_id="ctx-bbbb"))

    assert _session_of(runner, 0) == "workspace:ws-stable-1"
    assert _session_of(runner, 1) == "workspace:ws-stable-1"  # stable, not ctx-bbbb


@pytest.mark.asyncio
async def test_session_id_falls_back_to_context_id_without_workspace():
    runner = _Runner()
    ex = _make_executor(runner, workspace_id="")
    ex._workspace_id = ""  # force the no-identity fallback path deterministically
    await _run(ex, _build_context(text="m", context_id="chat-stable", task_id="task-x"))
    assert _session_of(runner, 0) == "chat-stable"


@pytest.mark.asyncio
async def test_session_id_falls_back_to_task_id_then_default():
    runner = _Runner()
    ex = _make_executor(runner, workspace_id="")
    ex._workspace_id = ""
    await _run(ex, _build_context(text="m", context_id=None, task_id="task-only"))
    assert _session_of(runner, 0) == "task-only"

    runner2 = _Runner()
    ex2 = _make_executor(runner2, workspace_id="")
    ex2._workspace_id = ""
    await _run(ex2, _build_context(text="m", context_id=None, task_id=None))
    assert _session_of(runner2, 0) == "default"


@pytest.mark.asyncio
async def test_workspace_id_read_from_env_when_not_passed(monkeypatch):
    monkeypatch.setenv("WORKSPACE_ID", "env-ws-9")
    runner = _Runner()
    ex = _make_executor(runner)  # no explicit workspace_id → base reads env
    await _run(ex, _build_context(text="m", context_id="ctx-z"))
    assert _session_of(runner, 0) == "workspace:env-ws-9"


# --------------------------------------------------------------------------- #
# CONTRACT #2 — NO history injection: only the current message reaches ADK.
# Continuity is the native ADK session resumed via the stable id, NOT a
# force-injected transcript (tenant-agent BUG 3 — the part CORE-A removed).
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_history_is_not_injected_into_adk_message():
    """Even with prior turns in metadata, ONLY the current message reaches ADK."""
    runner = _Runner()
    ex = _make_executor(runner, workspace_id="ws-1")
    history = _history(("user", "my name is Ada"), ("agent", "Hello Ada"))
    await _run(ex, _build_context(text="what did I just say?", history=history))

    text = _text_of(runner, 0)
    # ONLY the current message — verbatim; the prior turns must NOT be prepended,
    # and none of the old build_task_text framing ("Conversation so far:") leaks.
    assert text == "what did I just say?"
    assert "my name is Ada" not in text
    assert "Hello Ada" not in text
    assert "Conversation so far:" not in text


@pytest.mark.asyncio
async def test_bare_message_when_no_history():
    runner = _Runner()
    ex = _make_executor(runner, workspace_id="ws-1")
    await _run(ex, _build_context(text="hello"))
    assert _text_of(runner, 0) == "hello"


# --------------------------------------------------------------------------- #
# core#3082 — the loaded MCP tool INVENTORY is reported, independent of the turn.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_loaded_mcp_inventory_is_reported(monkeypatch):
    import molecule_runtime.platform_agent_identity as pai

    calls: list[list[str]] = []
    monkeypatch.setattr(pai, "set_loaded_mcp_tools", lambda tools: calls.append(list(tools or [])))

    runner = _Runner()
    ex = _make_executor(
        runner,
        workspace_id="ws-1",
        tools=[_McpToolset("list_peers", "commit_memory", "create_workspace")],
    )
    await _run(ex, _build_context(text="hi"))

    assert len(calls) == 1
    assert "mcp__molecule-platform__create_workspace" in calls[0]


@pytest.mark.asyncio
async def test_loaded_mcp_inventory_empty_when_no_toolset(monkeypatch):
    import molecule_runtime.platform_agent_identity as pai

    calls: list[list[str]] = []
    monkeypatch.setattr(pai, "set_loaded_mcp_tools", lambda tools: calls.append(list(tools or [])))

    runner = _Runner()
    ex = _make_executor(runner, workspace_id="ws-1", tools=[])
    await _run(ex, _build_context(text="hi"))

    assert calls == [[]]


# --------------------------------------------------------------------------- #
# Reply plumbing + OFFSEC-003 error sanitisation.
# --------------------------------------------------------------------------- #
@pytest.mark.asyncio
async def test_final_text_is_returned_as_the_reply():
    runner = _Runner(events=[_ADKEvent("thinking...", final=False), _ADKEvent("the answer is 42", final=True)])
    ex = _make_executor(runner, workspace_id="ws-1")
    q = await _run(ex, _build_context(text="q"))
    # base enqueues exactly one reply event (message mode — no Task/TaskUpdater)
    assert len(q.events) == 1


@pytest.mark.asyncio
async def test_run_agent_sanitizes_runtime_errors():
    """A raw ADK exception is sanitised (OFFSEC-003), never leaked to A2A."""
    class _BoomRunner(_Runner):
        def run_async(self, **_k):
            raise RuntimeError("adk exploded\nsecret traceback line")

    ex = _make_executor(_BoomRunner(), workspace_id="ws-1", tools=[])
    reply = await ex.run_agent("hi", "workspace:ws-1", _build_context(text="hi"))
    assert reply.startswith("[A2A_ERROR] ")
    assert "secret traceback" not in reply
    assert "\n" not in reply


# --------------------------------------------------------------------------- #
# GUARD — the thin subclass MUST NOT override the base's enforced execute().
# --------------------------------------------------------------------------- #
def test_execute_is_inherited_not_overridden():
    # The whole point of BUG 3: GoogleADKA2AExecutor implements only run_agent and
    # INHERITS execute() (stable session id + NO history injection). Overriding it
    # would silently opt out of the shared contract — this is the tripwire.
    assert GoogleADKA2AExecutor.execute is SubprocessA2AExecutor.execute


def test_run_agent_is_implemented():
    # The runtime-specific half of the contract is actually provided here.
    assert GoogleADKA2AExecutor.run_agent is not SubprocessA2AExecutor.run_agent
