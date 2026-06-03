"""Unit tests for the platform-managed routing backend (_routing.py)."""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import pytest
from _routing import resolve_model

PLATFORM_ENV = {
    "OPENAI_BASE_URL": "https://cp.example/api/v1/internal/llm/openai/v1",
    "OPENAI_API_KEY": "org-usage-token",
}

def test_platform_prefix_routes_via_proxy():
    r = resolve_model("platform:gemini-2.5-pro", PLATFORM_ENV)
    assert r.backend == "platform" and r.is_platform
    assert r.model == "openai/gemini-2.5-pro"  # bare id -> proxy -> Vertex
    assert r.is_gemini

def test_platform_flash_alias_and_molecule_prefix():
    assert resolve_model("platform:gemini-2.5-flash", PLATFORM_ENV).model == "openai/gemini-2.5-flash"
    assert resolve_model("molecule:gemini-2.5-pro", PLATFORM_ENV).backend == "platform"

def test_platform_without_proxy_env_fails_loud():
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        resolve_model("platform:gemini-2.5-pro", {})

def test_byok_paths_unchanged():
    assert resolve_model("google_genai:gemini-2.5-pro", {"GOOGLE_API_KEY": "k"}).backend == "ai_studio"
    assert resolve_model("vertex:gemini-2.5-pro", {"GOOGLE_CLOUD_PROJECT": "p"}).backend == "vertex"
