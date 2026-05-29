"""Google ADK runtime adapter for Molecule AI.

Runs Gemini via Google's Agent Development Kit (ADK) as a workspace agent,
governed by Molecule's A2A org. ADK is used as the agent ENGINE only
(``LlmAgent`` + ``Runner`` + ``McpToolset``) — install ``google-adk[mcp]``,
never the ``[a2a]`` extra (it pins an incompatible a2a-sdk). Tools reach the
agent through ADK's native ``McpToolset`` pointed at the platform's
``a2a_mcp_server`` — the same MCP surface the CLI runtimes use (no LangChain).

Design verified empirically (RFC internal#730):
- ``google-adk[mcp]==2.1.0`` coexists with the platform's ``a2a-sdk>=1.0``.
- ``LlmAgent``/``Runner``/``McpToolset`` build a Gemini agent from pure config.
"""

from __future__ import annotations

import logging
import os
import sys

from a2a.server.agent_execution import AgentExecutor

from molecule_runtime.adapters.base import AdapterConfig, BaseAdapter

from _routing import ResolvedModel, resolve_model, vertex_location
from google_adk_executor import GoogleADKA2AExecutor

logger = logging.getLogger(__name__)


def _safe_agent_name(workspace_id: str) -> str:
    """ADK agent names must be valid identifiers; derive one from the id."""
    base = "ws_" + (workspace_id or "agent").replace("-", "_")
    cleaned = "".join(ch if (ch.isalnum() or ch == "_") else "_" for ch in base)
    return cleaned[:60] or "molecule_agent"


class GoogleADKAdapter(BaseAdapter):
    """Molecule runtime adapter backed by Google ADK (``adk-python``)."""

    def __init__(self) -> None:
        self._resolved: ResolvedModel | None = None

    @staticmethod
    def name() -> str:
        return "google-adk"

    @staticmethod
    def display_name() -> str:
        return "Google ADK"

    @staticmethod
    def description() -> str:
        return (
            "Google Agent Development Kit (adk-python) runtime — Gemini via "
            "AI Studio or Vertex AI, with platform tools over MCP and ADK "
            "session state per A2A context."
        )

    @staticmethod
    def get_config_schema() -> dict:
        return {
            "model": {
                "type": "string",
                "description": "google_genai:gemini-2.5-pro (AI Studio) or vertex:gemini-2.5-pro",
                "default": "google_genai:gemini-2.5-pro",
            },
        }

    async def setup(self, config: AdapterConfig) -> None:
        """Resolve + validate the model and credentials before any task runs."""
        resolved = resolve_model(config.model, os.environ)
        self._resolved = resolved
        if resolved.backend == "vertex":
            # Make the Vertex selection explicit so a stale GOOGLE_API_KEY
            # can't shadow it; ADK reads these env vars natively.
            os.environ["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
            os.environ.setdefault("GOOGLE_CLOUD_LOCATION", vertex_location(os.environ))
        logger.info("google-adk setup: model=%s backend=%s", resolved.model, resolved.backend)

    def _build_mcp_toolset(self):
        """ADK McpToolset → the platform's a2a_mcp_server over stdio.

        Same MCP server the CLI runtimes launch (`get_mcp_server_path()`),
        exposing delegate/memory/peer/messaging tools. The subprocess inherits
        the full container env so WORKSPACE_ID / PLATFORM_URL propagate and the
        tools self-authenticate from /configs/.auth_token. ``McpToolset``
        depends on the ``mcp`` package, not ``a2a-sdk`` — unaffected by the
        ADK↔platform a2a-sdk version split.
        """
        from google.adk.tools.mcp_tool.mcp_toolset import McpToolset
        from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
        from mcp import StdioServerParameters

        from molecule_runtime.executor_helpers import get_mcp_server_path

        return McpToolset(
            connection_params=StdioConnectionParams(
                server_params=StdioServerParameters(
                    command=sys.executable,
                    args=[get_mcp_server_path()],
                    env=dict(os.environ),
                ),
                timeout=30.0,
            ),
        )

    async def create_executor(self, config: AdapterConfig) -> AgentExecutor:
        from google.adk.agents import LlmAgent
        from google.adk.runners import Runner
        from google.adk.sessions import InMemorySessionService

        if self._resolved is None:
            await self.setup(config)
        assert self._resolved is not None

        # Reuse the shared pipeline for the assembled system prompt + coordinator
        # detection. Deliberately ignore its LangChain tools — tools come via
        # MCPToolset instead (RFC internal#730).
        setup_result = await self._common_setup(config)

        if self._resolved.needs_litellm:
            from google.adk.models.lite_llm import LiteLlm
            model = LiteLlm(model=self._resolved.model)
        else:
            model = self._resolved.model

        agent = LlmAgent(
            name=_safe_agent_name(config.workspace_id),
            model=model,
            instruction=setup_result.system_prompt or "",
            tools=[self._build_mcp_toolset()],
        )

        app_name = "molecule"
        user_id = config.workspace_id or "molecule-workspace"
        runner = Runner(
            app_name=app_name,
            agent=agent,
            session_service=InMemorySessionService(),
        )
        logger.info(
            "google-adk executor ready: model=%s coordinator=%s",
            self._resolved.model, setup_result.is_coordinator,
        )
        return GoogleADKA2AExecutor(
            runner,
            app_name=app_name,
            user_id=user_id,
            model=self._resolved.model,
            heartbeat=config.heartbeat,
        )


Adapter = GoogleADKAdapter
