# molecule-ai-workspace-template-google-adk

Workspace template for the **`google-adk`** runtime — Gemini agents powered by
Google's Agent Development Kit (ADK), governed by Molecule's A2A org.

## Design (RFC internal#730)
- **ADK as engine only:** `LlmAgent` + `Runner` + `McpToolset`. Install
  `google-adk[mcp]==2.1.0` — **never** the `[a2a]` extra (it pins
  `a2a-sdk<0.4`, incompatible with the platform's `a2a-sdk>=1.0`).
- **Tools over MCP:** ADK's native `McpToolset` connects (stdio) to the
  platform's `a2a_mcp_server` — the same 8-tool surface the CLI runtimes use
  (delegate / memory / peers / messaging). No LangChain.
- **Molecule-authored executor** (`google_adk_executor.py`) drives ADK's
  `Runner` and bridges its event stream to the platform's a2a-1.x
  `EventQueue`/`TaskUpdater`, with heartbeat task accounting. OWASP / OTEL /
  Temporal are deferred (off/dormant in prod) per the RFC.
- **BYOK:** set `GOOGLE_API_KEY` (AI Studio) — or Vertex via
  `GOOGLE_GENAI_USE_VERTEXAI=1` + `GOOGLE_CLOUD_PROJECT`.

## Files
| File | Role |
|---|---|
| `adapter.py` | `GoogleADKAdapter` — builds MCPToolset + LlmAgent + Runner |
| `google_adk_executor.py` | Runner → a2a-1.x bridge + heartbeat |
| `_routing.py` | `provider:model` → bare Gemini id; AI Studio / Vertex |
| `config.yaml` | default runtime config (Gemini 2.5 Pro) |
| `Dockerfile` / `entrypoint.sh` | lean image (no node/T4); uid-1000 drop |

## Status
Unit-tested (`tests/`, routing + executor helpers). Stage A/B/C verification +
cross-repo registration (molecule-core manifest/knownRuntimes/canvas;
controlplane runtime_image_pins/providers; molecule-ci validator) +
providers-projection are tracked follow-ups per RFC internal#730.
