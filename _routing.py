"""Model-string → ADK model resolution for the google-adk runtime.

Pure logic, no ADK import — so it unit-tests without the heavy SDK and
without a live API key.

Workspace configs carry a ``provider:model`` string (e.g.
``google_genai:gemini-2.5-pro``). Google ADK's ``LlmAgent`` wants the
*bare* Gemini model id (``gemini-2.5-pro``) and reads credentials from
the environment:

  * AI Studio (default): ``GOOGLE_API_KEY``
  * Vertex AI:           ``GOOGLE_GENAI_USE_VERTEXAI=1`` +
                         ``GOOGLE_CLOUD_PROJECT`` (+ ``GOOGLE_CLOUD_LOCATION``)

This strips the provider prefix, decides AI-Studio-vs-Vertex, and fails
loudly (actionable message) when the required credential is absent —
rather than letting ADK raise an opaque SDK error mid-task.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

_GOOGLE_PREFIXES = frozenset(
    {"google", "google_genai", "googlegenai", "gemini", "vertex", "vertexai"}
)
_VERTEX_PREFIXES = frozenset({"vertex", "vertexai"})
_DEFAULT_VERTEX_LOCATION = "us-central1"
_TRUTHY = frozenset({"1", "true", "yes", "on"})


@dataclass(frozen=True)
class ResolvedModel:
    """Outcome of resolving a workspace ``provider:model`` string for ADK."""

    model: str       # what to hand LlmAgent(model=...): bare gemini id, or "provider/model" for LiteLlm
    backend: str     # "ai_studio" | "vertex" | "litellm"
    is_gemini: bool

    @property
    def needs_litellm(self) -> bool:
        return self.backend == "litellm"


def _split_prefix(model_str: str) -> tuple[str, str]:
    raw = (model_str or "").strip()
    if ":" in raw:
        prefix, model = raw.split(":", 1)
    else:
        prefix, model = "", raw
    model = model.strip()
    if model.startswith("models/"):
        model = model[len("models/"):]
    return prefix.strip().lower(), model


def _is_truthy(value: str | None) -> bool:
    return (value or "").strip().lower() in _TRUTHY


def resolve_model(model_str: str, env: Mapping[str, str]) -> ResolvedModel:
    """Resolve a workspace model string into an ADK-ready :class:`ResolvedModel`.

    Raises ``RuntimeError`` (actionable) when the required credential for the
    chosen backend is missing, or when a non-Gemini model is requested but the
    LiteLlm extension isn't available.
    """
    prefix, bare = _split_prefix(model_str)
    if not bare:
        raise RuntimeError(
            "Empty model string. Set workspace `model` to e.g. "
            "'google_genai:gemini-2.5-pro'."
        )

    is_gemini = prefix in _GOOGLE_PREFIXES or (
        prefix == "" and bare.lower().startswith("gemini")
    )

    if is_gemini:
        use_vertex = prefix in _VERTEX_PREFIXES or _is_truthy(
            env.get("GOOGLE_GENAI_USE_VERTEXAI")
        )
        if use_vertex:
            if not env.get("GOOGLE_CLOUD_PROJECT"):
                raise RuntimeError(
                    "Vertex AI selected (GOOGLE_GENAI_USE_VERTEXAI / vertex: prefix) "
                    "but GOOGLE_CLOUD_PROJECT is not set. Add it to workspace secrets "
                    "(and optionally GOOGLE_CLOUD_LOCATION; defaults to "
                    f"'{_DEFAULT_VERTEX_LOCATION}')."
                )
            return ResolvedModel(model=bare, backend="vertex", is_gemini=True)

        if not env.get("GOOGLE_API_KEY"):
            raise RuntimeError(
                "No GOOGLE_API_KEY found for the google-adk runtime. Get one at "
                "https://aistudio.google.com/apikey and store it as a workspace "
                "secret (or switch to Vertex with GOOGLE_GENAI_USE_VERTEXAI=1 + "
                "GOOGLE_CLOUD_PROJECT)."
            )
        return ResolvedModel(model=bare, backend="ai_studio", is_gemini=True)

    # Non-Gemini: route through LiteLlm if the extension is installed.
    try:
        import google.adk.models.lite_llm  # noqa: F401
    except Exception as exc:
        raise RuntimeError(
            f"Model {model_str!r} is not a Gemini model and LiteLlm support is "
            "unavailable (pip install 'google-adk[extensions]'). The google-adk "
            "runtime serves Gemini natively; use a google_genai:gemini-* model "
            "or install the LiteLlm extra."
        ) from exc
    litellm_model = f"{prefix}/{bare}" if prefix else bare
    return ResolvedModel(model=litellm_model, backend="litellm", is_gemini=False)


def vertex_location(env: Mapping[str, str]) -> str:
    """Vertex region, defaulting to us-central1."""
    return (env.get("GOOGLE_CLOUD_LOCATION") or "").strip() or _DEFAULT_VERTEX_LOCATION
