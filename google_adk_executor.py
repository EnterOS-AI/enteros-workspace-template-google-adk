"""A2A executor for the google-adk runtime.

Molecule-authored (NOT ADK's native ``A2aAgentExecutor``): ADK pins
``a2a-sdk<0.4`` for its a2a layer, while the platform's A2A server is on
``a2a-sdk>=1.0`` — incompatible. So we use ADK purely as the agent engine
(``LlmAgent`` + ``Runner`` + ``McpToolset``) and bridge its ``Runner``
event stream onto the platform's a2a-1.x model.

tenant-agent BUG 3 — inherit the SHARED session CONTRACT
========================================================
This executor used to hand-roll ``AgentExecutor.execute()`` and, like every
other subprocess/one-shot runtime, silently diverged from the platform's
session contract:

  * it keyed the ADK session on the per-request ``context_id`` (execute()
    passed ``session_id=context_id`` to ``Runner.run_async``), which the
    a2a-sdk mints FRESH each turn when the canvas threads none — so ADK's
    ``InMemorySessionService`` opened a NEW session every message and the
    agent re-greeted with no memory of the prior turn.

That contract is now INHERITED from the SSOT base ``SubprocessA2AExecutor``
(``molecule_runtime.subprocess_executor``): the base's ``execute()`` derives a
STABLE, WORKSPACE_ID-keyed session id (``derive_session_id``) so ADK's own
``InMemorySessionService`` RESUMES the same session across turns — that native
session IS the continuity. The base deliberately passes ONLY the current user
message to ``run_agent``; it does NOT force-inject ``metadata.history`` into the
task text (that double-fed context alongside the native session and grew the
prompt unboundedly). Older/other history is retrieved only if the agent CHOOSES
to call the platform-workspace ``get_conversation_history`` MCP tool — never
shoved into every turn.

This class is a THIN subclass that implements ONLY ``run_agent`` — the ADK
``Runner`` shell-out — and MUST NOT override ``execute()`` (the base's shared
contract test is the tripwire that enforces this).

Platform-citizen scope (RFC internal#730): heartbeat task accounting is handled
by the base (``set_current_task`` brackets every turn via the injected
heartbeat). The loaded-MCP-tool inventory (core#3082) and the OFFSEC-003 error
sanitisation remain here because they are ADK-specific.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Iterable
from typing import Any

# The shared subprocess-executor base (tenant-agent BUG 3): the session
# CONTRACT — a STABLE, WORKSPACE_ID-keyed session id, with continuity supplied by
# the runtime's OWN native session (NOT a force-injected transcript) — lives ONCE
# in the SSOT runtime SDK so every subprocess/one-shot runtime inherits it.
# GoogleADKA2AExecutor below is a thin subclass that provides ONLY the ADK Runner
# shell-out (run_agent). It is a hard import (no fallback): the base ships with
# molecule-ai-workspace-runtime, and a runtime too old to carry it must fail
# loudly rather than silently run WITHOUT the enforced contract. (Deploy
# ordering: runtime #222 releases the base before this template's image pins a
# runtime that has it.)
from molecule_runtime.subprocess_executor import SubprocessA2AExecutor

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


# ---------------------------------------------------------------------------
# Executor — a THIN SubprocessA2AExecutor subclass (tenant-agent BUG 3)
# ---------------------------------------------------------------------------

class GoogleADKA2AExecutor(SubprocessA2AExecutor):
    """Drives an ADK ``Runner`` for one A2A turn.

    The session CONTRACT is INHERITED from ``SubprocessA2AExecutor``: the base's
    ``execute()`` derives a STABLE, WORKSPACE_ID-keyed session id
    (``derive_session_id``) — NOT the per-request ``context_id`` the a2a-sdk mints
    fresh each turn — and passes ONLY the current user message through (it does
    NOT force-inject conversation history). This class provides ONLY
    ``run_agent`` — the ADK ``Runner.run_async`` shell-out keyed on that stable
    session id, so ADK's ``InMemorySessionService`` RESUMES the same session and
    that native session supplies the continuity.

    It MUST NOT override ``execute()``; the contract lives in the base and is
    guarded by the shared contract test.
    """

    runtime_label = "google-adk"

    def __init__(
        self,
        runner,
        *,
        app_name: str,
        user_id: str,
        model: str = "unknown",
        heartbeat=None,
        tools=None,
        workspace_id: str = "",
    ):
        # WORKSPACE_ID-first stable identity + heartbeat live in the base.
        super().__init__(workspace_id=workspace_id, heartbeat=heartbeat)
        self._runner = runner
        self._app_name = app_name
        self._user_id = user_id
        self._model = model
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

    async def run_agent(self, task_text: str, session_id: str, context) -> str:
        """Run one ADK turn and return its final text (base template method).

        Args:
            task_text: the CURRENT user message — passed straight to ADK as the
                new ``user`` content. NO conversation history is prepended: the
                base does not inject a transcript, and continuity comes from the
                native ADK session resumed via ``session_id`` (tenant-agent BUG 3).
            session_id: the STABLE, WORKSPACE_ID-keyed session id from the base
                (``derive_session_id``). Passed to ADK's session so its
                ``InMemorySessionService`` RESUMES the same session across turns
                instead of opening a new one per request.
            context: the raw A2A RequestContext (unused here — the base already
                extracted the current message).
        """
        from google.genai import types
        from molecule_runtime.platform_agent_identity import set_loaded_mcp_tools

        # core#3082: report the actually-loaded MCP tool INVENTORY (independent
        # of whether this turn invokes any tool). Reported up front so the
        # platform's online/degraded gate sees it even if the turn later errors.
        set_loaded_mcp_tools(await extract_loaded_mcp_tools(self._tools))

        try:
            await self._ensure_session(session_id)
            new_message = types.Content(role="user", parts=[types.Part(text=task_text)])
            events = self._runner.run_async(
                user_id=self._user_id, session_id=session_id, new_message=new_message
            )
            final_text = await self._drain(events)
            return final_text or "(no response)"
        except Exception as exc:  # noqa: BLE001 — never leak a Google SDK trace to A2A
            logger.exception("google-adk run_agent failed for session %s", session_id)
            return sanitize_error(exc)

    async def _drain(self, events) -> str:
        """Async-drain ADK's event stream and return the final text."""
        final_text = ""
        streamed: list[str] = []
        async for event in events:
            texts = extract_event_text(event)
            if is_final(event):
                final_text = "".join(texts)
            else:
                streamed.extend(texts)
        return final_text or "".join(streamed)
