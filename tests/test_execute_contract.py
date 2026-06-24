"""Async contract tests for GoogleADKA2AExecutor.execute().

Pins the two a2a-sdk v1 task-mode contract bugs the live e2e surfaced
(2026-05-30) that pure-helper tests structurally could not catch:

  bug #3: a Task must be enqueued (state=SUBMITTED) BEFORE the first
          TaskStatusUpdateEvent (updater.start_work) — else JSON-RPC -32603
          "Agent should enqueue Task before TaskStatusUpdateEvent event".
  bug #4: once in task mode, the terminal reply must go through
          updater.complete(message=...) (a COMPLETED TaskStatusUpdateEvent),
          NOT a raw event_queue.enqueue_event(message) — else -32603
          "Received Message object in task mode. Use TaskStatusUpdateEvent...".

execute() resolves its a2a/google/runtime deps via lazy imports, so we
install minimal sys.modules stubs before invoking it.
"""

import sys
import types as _t

import pytest


# ---- minimal stubs for execute()'s lazy imports -------------------------

class _FakeMessage:
    def __init__(self, text, task_id=None, context_id=None):
        self.text = text
        self.task_id = task_id
        self.context_id = context_id


def _install_execute_stubs():
    # a2a.helpers.new_text_message
    helpers = _t.ModuleType("a2a.helpers")
    helpers.new_text_message = lambda text, task_id=None, context_id=None, **k: _FakeMessage(text, task_id, context_id)
    sys.modules["a2a.helpers"] = helpers

    # a2a.server.tasks.TaskUpdater — records the calls execute() makes
    tasks = _t.ModuleType("a2a.server.tasks")

    class _FakeUpdater:
        def __init__(self, queue, task_id, context_id):
            self.queue = queue
            self.task_id = task_id
            self.context_id = context_id
            self.calls = []
            queue._updater = self  # expose to the test

        async def start_work(self):
            self.calls.append(("start_work",))

        async def add_artifact(self, parts, artifact_id=None, append=False):
            self.calls.append(("add_artifact", append))

        async def complete(self, message=None):
            self.calls.append(("complete", message))

        async def failed(self, message=None):
            self.calls.append(("failed", message))

    tasks.TaskUpdater = _FakeUpdater
    sys.modules["a2a.server.tasks"] = tasks

    # a2a.types: Part, Task, TaskStatus (extend the conftest-installed module)
    a2a_types = sys.modules.get("a2a.types") or _t.ModuleType("a2a.types")
    a2a_types.Part = type("Part", (), {"__init__": lambda self, text=None: setattr(self, "text", text)})
    a2a_types.Task = type("Task", (), {"__init__": lambda self, id=None, context_id=None, status=None: None})
    a2a_types.TaskStatus = type("TaskStatus", (), {"__init__": lambda self, state=None: None})
    sys.modules["a2a.types"] = a2a_types

    # google.genai.types: Content + Part
    genai = _t.ModuleType("google.genai")
    gtypes = _t.ModuleType("google.genai.types")
    gtypes.Content = type("Content", (), {"__init__": lambda self, role=None, parts=None: None})
    gtypes.Part = type("Part", (), {"__init__": lambda self, text=None: setattr(self, "text", text)})
    genai.types = gtypes
    sys.modules["google.genai"] = genai
    sys.modules["google.genai.types"] = gtypes

    # molecule_runtime.adapters.shared_runtime + executor_helpers
    sr = _t.ModuleType("molecule_runtime.adapters.shared_runtime")

    async def _set_current_task(*_a, **_k):
        return None

    sr.brief_task = lambda x: (x or "")[:40]
    sr.extract_message_text = lambda _ctx: ""  # unused: get_user_input wins
    sr.set_current_task = _set_current_task
    sys.modules["molecule_runtime.adapters.shared_runtime"] = sr

    eh = _t.ModuleType("molecule_runtime.executor_helpers")
    eh.task_state_value = lambda name: name
    sys.modules["molecule_runtime.executor_helpers"] = eh

    # core#3082: google-adk execute() reports actually-loaded MCP tool ids via
    # molecule_runtime.platform_agent_identity.set_loaded_mcp_tools. The CI
    # runner does not install the real runtime package, so provide a recording
    # stub that lets contract tests assert the hook is called with the inventory.
    pai = _t.ModuleType("molecule_runtime.platform_agent_identity")
    pai._loaded_calls: list[list[str]] = []
    pai.set_loaded_mcp_tools = lambda tools: pai._loaded_calls.append(list(tools or []))
    sys.modules["molecule_runtime.platform_agent_identity"] = pai


_install_execute_stubs()

from google_adk_executor import GoogleADKA2AExecutor  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_loaded_calls():
    sys.modules["molecule_runtime.platform_agent_identity"]._loaded_calls.clear()
    yield


# ---- fakes for the runner / session / event queue / context -------------

class _Part:
    def __init__(self, text):
        self.text = text


class _Content:
    def __init__(self, parts):
        self.parts = parts


class _Event:
    def __init__(self, text, final):
        self.content = _Content([_Part(text)])
        self._final = final

    def is_final_response(self):
        return self._final


class _SessionService:
    async def get_session(self, **_k):
        return object()  # session exists → no create

    async def create_session(self, **_k):
        return object()


class _Runner:
    def __init__(self, events):
        self.session_service = _SessionService()
        self._events = events

    def run_async(self, *, user_id, session_id, new_message):
        async def _gen():
            for e in self._events:
                yield e
        return _gen()


class _Tool:
    def __init__(self, name):
        self.name = name


class _McpToolset:
    """Fake McpToolset exposing the platform tool inventory."""
    def __init__(self, *names):
        self._names = names

    def get_tools(self):
        return [_Tool(n) for n in self._names]


_DEFAULT_TOOLS = [
    _McpToolset(
        "list_peers",
        "commit_memory",
        "send_message_to_user",
        "create_workspace",
    ),
]


class _Queue:
    def __init__(self):
        self.enqueued = []
        self._updater = None

    async def enqueue_event(self, ev):
        self.enqueued.append(ev)


class _Ctx:
    def __init__(self, text, current_task=None, task_id="t1", context_id="c1"):
        self._text = text
        self.current_task = current_task
        self.task_id = task_id
        self.context_id = context_id

    def get_user_input(self):
        return self._text


async def _run(events, *, text="hello", current_task=None, tools=None):
    ex = GoogleADKA2AExecutor(
        _Runner(events), app_name="molecule", user_id="u1", tools=tools if tools is not None else _DEFAULT_TOOLS
    )
    q = _Queue()
    await ex.execute(_Ctx(text, current_task=current_task), q)
    return q, q._updater


@pytest.mark.asyncio
async def test_success_completes_via_updater_not_raw_enqueue():
    q, updater = await _run([_Event("Hello from Gemini on ADK.", final=True)])
    kinds = [c[0] for c in updater.calls]
    # bug #4: terminal reply must be updater.complete(), never a raw enqueue
    assert "complete" in kinds, f"expected updater.complete(); calls={kinds}"
    msg = next(c[1] for c in updater.calls if c[0] == "complete")
    assert msg.text == "Hello from Gemini on ADK."
    # the only thing enqueued raw is the SUBMITTED Task (bug #3), never the reply text
    assert len(q.enqueued) == 1, f"only the Task may be raw-enqueued; got {q.enqueued}"
    # core#3082: the loaded MCP inventory is reported independent of this turn's calls
    pai = sys.modules["molecule_runtime.platform_agent_identity"]
    assert len(pai._loaded_calls) == 1
    assert "mcp__molecule-platform__create_workspace" in pai._loaded_calls[0]


@pytest.mark.asyncio
async def test_fresh_request_enqueues_task_before_start_work():
    q, updater = await _run([_Event("ok", final=True)])
    # bug #3: exactly one raw enqueue (the Task), and start_work happened after
    assert len(q.enqueued) == 1
    assert updater.calls[0][0] == "start_work"  # updater created AFTER the Task enqueue


@pytest.mark.asyncio
async def test_continuation_skips_task_enqueue():
    # when the SDK already has the task (current_task set), no raw Task enqueue
    q, updater = await _run([_Event("ok", final=True)], current_task=object())
    assert q.enqueued == [], "continuation turn must not re-enqueue a Task"
    assert "complete" in [c[0] for c in updater.calls]


@pytest.mark.asyncio
async def test_error_path_uses_updater_failed():
    class _Boom(_Runner):
        def run_async(self, **_k):
            raise RuntimeError("adk exploded")
    ex = GoogleADKA2AExecutor(_Boom([]), app_name="molecule", user_id="u1")
    q = _Queue()
    await ex.execute(_Ctx("hi"), q)
    kinds = [c[0] for c in q._updater.calls]
    assert "failed" in kinds, f"error path must call updater.failed(); calls={kinds}"


@pytest.mark.asyncio
async def test_3082_no_loaded_toolset_reports_empty_inventory():
    # present-only (no McpToolset loaded) must report empty inventory so the
    # platform gate degrades rather than trusting a guessed static list.
    q, updater = await _run([_Event("ok", final=True)], tools=[])
    assert "complete" in [c[0] for c in updater.calls]
    pai = sys.modules["molecule_runtime.platform_agent_identity"]
    assert len(pai._loaded_calls) == 1
    assert pai._loaded_calls[0] == []
