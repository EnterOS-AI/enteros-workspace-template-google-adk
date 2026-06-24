"""Regression guard: the platform-managed default model needs LiteLlm.

The google-adk runtime's DEFAULT model is `platform:gemini-2.5-*` (config.yaml),
which routes Gemini through Molecule's OpenAI-compatible LLM proxy.
`adapter.create_executor()` builds that path with
`from google.adk.models.lite_llm import LiteLlm`. In google-adk 2.1.0, LiteLlm
lives behind the `[extensions]` extra (it pulls `litellm`); installing only
`[mcp]` makes that import raise

    ImportError: LiteLLM support requires: pip install google-adk[extensions]

at boot, leaving the workspace un-servable on its default model (the
"google-adk fails to start / no URL" symptom). Because the import is LAZY, a
`[mcp]`-only image still builds clean and passes the import-only docker smoke,
and the e2e live arm only exercises AI-Studio BYOK (which does NOT need
LiteLlm) — so nothing caught the gap until a live `platform:` repro.

Two layers of guard:
  1. test_dockerfile_installs_litellm_extra (here) — a pure, dependency-free
     static check that the Dockerfile keeps the `extensions` extra. Runs in the
     stubbed `Adapter unit tests` CI job (no ADK install needed).
  2. ci.yml "litellm import smoke" — runs `python -c "import
     google.adk.models.lite_llm"` INSIDE the freshly built image, proving the
     extra actually resolved at install time. Runs in the docker-build job.

Keep both: (1) is fast and always runs; (2) catches an upstream packaging
change that moves LiteLlm out of `extensions` without us editing the Dockerfile.
"""

import os
import re

_TEMPLATE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _dockerfile_text() -> str:
    with open(os.path.join(_TEMPLATE_DIR, "Dockerfile"), encoding="utf-8") as fh:
        return fh.read()


def test_dockerfile_installs_litellm_extra():
    """The google-adk install line must include the `extensions` extra."""
    text = _dockerfile_text()
    # The pinned install spec (the RUN line, not the prose) must carry extensions.
    pinned = re.search(r'pip install[^\n]*google-adk\[([^\]]*)\]==', text)
    assert pinned is not None, "no pinned `pip install google-adk[...]==` line found"
    extras = {e.strip() for e in pinned.group(1).split(",")}
    assert "extensions" in extras, (
        "google-adk install is missing the `extensions` extra — LiteLlm "
        "(the platform-managed default model's backend) will ImportError at "
        f"boot. Found extras: {sorted(extras)}"
    )
    # mcp must remain (McpToolset / platform tools) and a2a must stay OUT
    # (it pins an incompatible a2a-sdk<0.4).
    assert "mcp" in extras, "google-adk install dropped the `mcp` extra (McpToolset)"
    assert "a2a" not in extras, (
        "google-adk install must NOT include the `a2a` extra (pins a2a-sdk<0.4, "
        "incompatible with the platform's a2a-sdk>=1.0)"
    )


def test_adapter_platform_path_imports_litellm_symbol():
    """The adapter's platform branch imports LiteLlm from the extensions module.

    Static assertion (no ADK import) that adapter.py still references the
    `google.adk.models.lite_llm` import the Dockerfile extra provisions — so a
    future refactor that renames the import path re-triggers a Dockerfile
    review rather than silently re-breaking the boot.
    """
    with open(os.path.join(_TEMPLATE_DIR, "adapter.py"), encoding="utf-8") as fh:
        adapter_src = fh.read()
    assert "from google.adk.models.lite_llm import LiteLlm" in adapter_src, (
        "adapter.py no longer imports LiteLlm from google.adk.models.lite_llm — "
        "if the platform path changed, update the Dockerfile extras + this guard"
    )
