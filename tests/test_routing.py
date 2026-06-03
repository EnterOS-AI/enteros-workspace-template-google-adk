"""Unit tests for google-adk model/provider routing (pure, no SDK)."""

import pytest
from _routing import resolve_model, vertex_location

AISTUDIO = {"GOOGLE_API_KEY": "AIza-test"}
# Leak fix (task #64): Vertex serves via the metered Molecule proxy now — the
# platform injects OPENAI_BASE_URL (CP proxy) + OPENAI_API_KEY (org usage token).
PROXY = {"OPENAI_BASE_URL": "https://cp/api/v1/internal/llm/openai/v1", "OPENAI_API_KEY": "org-tok"}


@pytest.mark.parametrize("model_str", [
    "google_genai:gemini-2.5-pro",
    "google:gemini-2.5-pro",
    "gemini:gemini-2.5-pro",
    "gemini-2.5-pro",
    "google_genai:models/gemini-2.5-pro",
])
def test_strips_prefix_to_bare_gemini_id(model_str):
    r = resolve_model(model_str, AISTUDIO)
    assert r.model == "gemini-2.5-pro"
    assert r.is_gemini is True
    assert r.backend == "ai_studio"
    assert r.needs_litellm is False


def test_ai_studio_requires_api_key():
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        resolve_model("google_genai:gemini-2.5-pro", {})


def test_use_vertexai_optin_routes_via_proxy():
    # Leak fix: GOOGLE_GENAI_USE_VERTEXAI no longer selects on-box ADC; it routes
    # Gemini through the metered Molecule proxy (platform backend).
    r = resolve_model("google_genai:gemini-2.5-pro", {**PROXY, "GOOGLE_GENAI_USE_VERTEXAI": "1"})
    assert r.backend == "platform" and r.model == "openai/gemini-2.5-pro" and r.is_gemini


def test_vertex_prefix_routes_via_proxy():
    # Leak fix: vertex: prefix routes through the metered proxy (no on-box ADC).
    r = resolve_model("vertex:gemini-2.0-flash", PROXY)
    assert r.backend == "platform" and r.model == "openai/gemini-2.0-flash"


def test_vertex_without_proxy_env_is_actionable_error():
    # Leak fix: vertex: needs the platform proxy env, not GOOGLE_CLOUD_PROJECT.
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        resolve_model("vertex:gemini-2.5-pro", {})


def test_empty_model_is_actionable_error():
    with pytest.raises(RuntimeError, match="Empty model string"):
        resolve_model("", AISTUDIO)
    with pytest.raises(RuntimeError, match="Empty model string"):
        resolve_model("google_genai:", AISTUDIO)


def test_non_gemini_without_litellm_raises_actionable():
    with pytest.raises(RuntimeError, match="LiteLlm"):
        resolve_model("openai:gpt-4o", AISTUDIO)


def test_vertex_location_default_and_override():
    assert vertex_location({}) == "us-central1"
    assert vertex_location({"GOOGLE_CLOUD_LOCATION": "europe-west4"}) == "europe-west4"
