"""A2A executor for the google-adk runtime.

Molecule-authored (NOT ADK's native ``A2aAgentExecutor``): ADK pins
``a2a-sdk<0.4`` for its a2a layer, while the platform's A2A server is on
``a2a-sdk>=1.0`` — incompatible. So we use ADK purely as the agent engine
(``LlmAgent`` + ``Runner`` + ``McpToolset``) and bridge its ``Runner``
event stream onto the platform's a2a-1.x ``EventQueue``/``TaskUpdater``
ourselves — the same shape ``LangGraphA2AExecutor`` uses.

Platform-citizen scope (RFC internal#730, approved): the A2A event
contract + heartbeat task accounting (``set_current_task``) are
implemented here because they are load-bearing (online/busy/offline
recovery + scheduler concurrency). OWASP compliance, OTEL spans, and the
Temporal durable wrapper are OFF/dormant in production today, so they are
deferred (gated no-op) — see the RFC's "out of scope" section.
"""

from __future__ import annotations

import inspect
import logging
import uuid
from collections.abc import Iterable
from typing import Any

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue

logger = logging.getLogger(__name__)

# Platform MCP server slug used to turn raw MCP tool names (e.g. ``create_workspace``)
# into the namespaced platform IDs (``mcp__molecule-platform__create_workspace``)
# the controlplane's online/degraded gate expects.
_PLATFORM_MCP_SERVER = "molecule-platform"


# ---------------------------------------------------------------------------
# Pure helpers — unit-tested without ADK, a2a, a live platform, or a key.
# ADK Event/Content/Part objects are duck-typed (``.content.parts[*].text``
# + ``.is_final_response()``), so tests drive lightweight fakes.
# ---------------------------------------------------------------------------

def extract_event_text(event) -> list[str]:
    """Return the non-empty text parts an ADK event carries.

    Only ``text`` parts surface; function-call / function-response parts
    (tool traffic) are skipped so raw tool JSON never leaks into A2A output.
    """
    content = getattr(event, "content", None)
    parts = getattr(content, "parts", None) or []
    texts: list[str] = []
    for part in parts:
        text = getattr(part, "text", None)
        if isinstance(text, str) and text:
            texts.append(text)
    return texts


def is_final(event) -> bool:
    """True when the event is ADK's terminal response for the turn."""
    fn = getattr(event, "is_final_response", None)
    if callable(fn):
        try:
            return bool(fn())
        except Exception:
            return False
    return False


def collect_response_text(events: Iterable, *, on_chunk=None) -> str:
    """Drain events (sync iterable), optionally stream chunks, return final text.

    The text of the ``is_final_response()`` event wins. If no event is flagged
    final (defensive), fall back to the concatenation of everything streamed,
    so a non-empty model output is never dropped.
    """
    final_text = ""
    streamed: list[str] = []
    for event in events:
        texts = extract_event_text(event)
        if is_final(event):
            final_text = "".join(texts)
        else:
            for t in texts:
                streamed.append(t)
                if on_chunk is not None:
                    on_chunk(t)
    return final_text or "".join(streamed)


def sanitize_error(exc: Exception) -> str:
    """Turn a raw SDK exception into an A2A-safe, single-line, tagged string.

    Never leak a Google SDK stack trace to the calling agent. Mirrors the
    OFFSEC-003 sanitisation the langgraph/claude executors apply.
    """
    msg = str(exc).strip() or exc.__class__.__name__
    first_line = msg.splitlines()[0][:300]
    return f"[A2A_ERROR] google-adk runtime error: {first_line}"


async def extract_loaded_mcp_tools(tools) -> list[str]:
    """Return the loaded MCP tool ids from the ADK agent's tool inventory.

    ``tools`` is ``agent.tools`` — for the platform concierge this contains one
    or more ``McpToolset`` objects. Each ``McpToolset`` exposes ``get_tools()``
    (sync or async) returning the actually-loaded tool declarations. ADK returns
    their raw MCP names (e.g. ``create_workspace``); we normalise them to the
    platform IDs the controlplane gate expects, such as
    ``mcp__molecule-platform__create_workspace``. Non-MCP tools are ignored.

    Duck-typed and defensive: we never import ``google.adk`` here, and a
    toolset that fails to enumerate is skipped rather than crashing the turn.
    """
    result: list[str] = []
    seen: set[str] = set()

    for tool in tools or []:
        sub_tools: list[Any] = []
        get_tools = getattr(tool, "get_tools", None)
        if callable(get_tools):
            try:
                maybe_tools = get_tools()
                if inspect.iscoroutine(maybe_tools):
                    maybe_tools = await maybe_tools
                sub_tools = list(maybe_tools or [])
            except Exception as exc:  # noqa: BLE001
                logger.debug(
                    "extract_loaded_mcp_tools: toolset enumeration failed: %s",
                    exc,
                )
                continue
        else:
            # ``agent.tools`` may contain LangChain-style builtins or already-
            # expanded tools; only MCP toolset inventories are reported here.
            continue

        for t in sub_tools:
            name = getattr(t, "name", None)
            if not isinstance(name, str) or not name:
                continue
            # Normalise raw MCP tool names to the platform namespaced ID.
            if not name.startswith("mcp__"):
                name = f"mcp__{_PLATFORM_MCP_SERVER}__{name}"
            if name not in seen:
                seen.add(name)
                result.append(name)

    return result


def extract_incoming_text(context, primary) -> str:
    """Resolve the user's text (and any attachment manifest) for this turn.

    ``primary`` is the platform's ``extract_message_text`` — it is
    attachment-aware (emits the ``Attached files:`` manifest with local paths,
    the only channel that tells the agent a file exists) and resolves correctly
    on the pinned a2a-sdk 1.0.3. It runs FIRST so a text+attachment message
    never loses its files.

    Only when ``primary`` yields nothing do we fall back to the a2a SDK's own
    ``context.get_user_input()`` — a version-stable safety net for the failure
    mode that triggered this fix: on a2a-sdk 1.1.0 ``extract_message_text``
    silently returned ``""`` (part shape changed) and every turn errored with
    "no text content" (e2e-found 2026-05-29). The fallback recovers the text
    (attachments would be unavailable in that degraded path, but a future SDK
    bump no longer hard-breaks the turn). Pure: ``context`` duck-typed,
    ``primary`` injected.
    """
    text = primary(context)
    if text:
        return text
    try:
        return (context.get_user_input() or "").strip()
    except Exception:  # noqa: BLE001 — older/edge SDKs lack it
        return ""


# ---------------------------------------------------------------------------
# Executor
# ---------------------------------------------------------------------------

class GoogleADKA2AExecutor(AgentExecutor):
    """Bridges an ADK ``Runner`` to the platform's a2a-1.x event model.

    One ADK session per A2A ``context_id`` (created lazily). Emits a
    ``working`` status, streams intermediate text as artifacts, and returns
    the final response as a terminal A2A message. Heartbeat task accounting
    via ``set_current_task`` brackets every turn.
    """

    def __init__(self, runner, *, app_name: str, user_id: str, model: str = "unknown", heartbeat=None, tools=None):
        self._runner = runner
        self._app_name = app_name
        self._user_id = user_id
        self._model = model
        self._heartbeat = heartbeat
        self._tools = tools or []

    async def _ensure_session(self, session_id: str) -> None:
        svc = self._runner.session_service
        existing = await svc.get_session(
            app_name=self._app_name, user_id=self._user_id, session_id=session_id
        )
        if existing is None:
            await svc.create_session(
                app_name=self._app_name, user_id=self._user_id, session_id=session_id
            )

    async def execute(self, context: RequestContext, event_queue: EventQueue) -> None:
        from a2a.helpers import new_text_message
        from a2a.server.tasks import TaskUpdater
        from a2a.types import Part, Task, TaskStatus
        from google.genai import types

        from molecule_runtime.adapters.shared_runtime import (
            brief_task,
            extract_message_text,
            set_current_task,
        )
        from molecule_runtime.executor_helpers import task_state_value

        # extract_message_text (attachment-aware) first, context.get_user_input()
        # as the SDK-version-stable empty-result fallback — see extract_incoming_text doc.
        user_input = extract_incoming_text(context, extract_message_text)
        if not user_input:
            await event_queue.enqueue_event(
                new_text_message("Error: message contained no text content.")
            )
            return

        task_id = context.task_id or str(uuid.uuid4())
        context_id = context.context_id or str(uuid.uuid4())

        # A2A v1 contract (a2a-sdk >= 1.0): a Task must be enqueued before any
        # TaskStatusUpdateEvent. The SDK auto-creates the Task only for
        # continuation messages (existing task resolves via the task manager);
        # for a FRESH request context.current_task is None and the first
        # updater.start_work() is rejected with InvalidAgentResponseError
        # "Agent should enqueue Task before TaskStatusUpdateEvent event". Mirror
        # the platform reference executor: enqueue a SUBMITTED Task first.
        # (e2e-found 2026-05-30 — only reachable once the incoming-text fix let
        # execution proceed past extraction.)
        if getattr(context, "current_task", None) is None:
            # task_state_value resolves TASK_STATE_SUBMITTED across the SDK's
            # protobuf TaskState (TaskState.Value(name)) and test-stub shapes —
            # the platform's TaskState is a protobuf enum, NOT a Python enum, so
            # TaskState.submitted does not exist. Mirrors the reference executor.
            await event_queue.enqueue_event(
                Task(
                    id=task_id,
                    context_id=context_id,
                    status=TaskStatus(state=task_state_value("TASK_STATE_SUBMITTED")),
                )
            )

        updater = TaskUpdater(event_queue, task_id, context_id)

        # Heartbeat task accounting — load-bearing (online/busy/offline + scheduler).
        await set_current_task(self._heartbeat, brief_task(user_input))
        try:
            await self._ensure_session(context_id)
            await updater.start_work()

            # core#3082: enumerate the actually-loaded MCP tool inventory from the
            # agent's toolset(s). This is the available-tool inventory, independent
            # of whether the current turn invokes any particular tool.
            loaded_tools = await extract_loaded_mcp_tools(self._tools)

            new_message = types.Content(role="user", parts=[types.Part(text=user_input)])
            events = self._runner.run_async(
                user_id=self._user_id, session_id=context_id, new_message=new_message
            )
            final_text = await self._drain(events, updater, Part)
            from molecule_runtime.platform_agent_identity import set_loaded_mcp_tools
            set_loaded_mcp_tools(loaded_tools)
            # A2A v1 (a2a-sdk >= 1.0): once a Task is enqueued (above) the
            # executor is in TASK MODE, where a raw Message enqueue is rejected
            # ("Received Message object in task mode. Use TaskStatusUpdateEvent
            # or TaskArtifactUpdateEvent instead.", JSON-RPC -32603 — e2e-found
            # 2026-05-30). The terminal reply must go through updater.complete(),
            # which wraps the Message in a COMPLETED TaskStatusUpdateEvent.
            # Mirrors the platform reference executor (a2a_executor.py:674).
            await updater.complete(
                message=new_text_message(
                    final_text or "(no response)", task_id=task_id, context_id=context_id
                )
            )
        except Exception as exc:  # noqa: BLE001 — SDK errors are sanitised, never raised to A2A
            logger.exception("google-adk execute failed for context %s", context_id)
            # Task mode: terminal errors publish a FAILED TaskStatusUpdateEvent
            # via updater.failed(), not a raw Message enqueue (same -32603 rule).
            await updater.failed(
                message=new_text_message(
                    sanitize_error(exc), task_id=task_id, context_id=context_id
                )
            )
        finally:
            # Clear the in-flight task so heartbeat active_tasks decrements.
            await set_current_task(self._heartbeat, None)

    async def _drain(self, events, updater, Part) -> str:
        """Async-drain ADK's event stream, emit artifacts, return final text."""
        final_text = ""
        streamed: list[str] = []
        artifact_id = str(uuid.uuid4())
        has_streamed = False
        async for event in events:
            texts = extract_event_text(event)
            if is_final(event):
                final_text = "".join(texts)
            else:
                for t in texts:
                    streamed.append(t)
                    await updater.add_artifact(
                        parts=[Part(text=t)], artifact_id=artifact_id, append=has_streamed
                    )
                    has_streamed = True
        return final_text or "".join(streamed)

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        # ADK's in-memory Runner has no mid-run cancellation hook; the A2A
        # framework marks the task cancelled. Nothing to tear down here.
        logger.info("google-adk: cancel requested for context %s", getattr(context, "context_id", "?"))
