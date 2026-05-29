"""Unit tests for google-adk model/provider routing (pure, no SDK)."""

import pytest
from _routing import resolve_model, vertex_location

AISTUDIO = {"GOOGLE_API_KEY": "AIza-test"}
VERTEX = {"GOOGLE_GENAI_USE_VERTEXAI": "1", "GOOGLE_CLOUD_PROJECT": "proj-123"}


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


def test_vertex_via_env_flag():
    r = resolve_model("google_genai:gemini-2.5-pro", VERTEX)
    assert r.backend == "vertex" and r.model == "gemini-2.5-pro" and r.is_gemini


def test_vertex_via_prefix_without_env_flag():
    r = resolve_model("vertex:gemini-2.0-flash", {"GOOGLE_CLOUD_PROJECT": "p"})
    assert r.backend == "vertex" and r.model == "gemini-2.0-flash"


def test_vertex_requires_project():
    with pytest.raises(RuntimeError, match="GOOGLE_CLOUD_PROJECT"):
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
