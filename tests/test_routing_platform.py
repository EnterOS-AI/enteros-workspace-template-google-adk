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

def test_vertex_prefix_now_routes_via_proxy_not_onbox_adc():
    # The keyless-Vertex leak fix: vertex: no longer delivers on-box ADC; it
    # goes through the metered proxy like platform:.
    r = resolve_model("vertex:gemini-2.5-pro", PLATFORM_ENV)
    assert r.backend == "platform"
    assert r.model == "openai/gemini-2.5-pro"

def test_use_vertexai_optin_routes_via_proxy():
    env = dict(PLATFORM_ENV); env["GOOGLE_GENAI_USE_VERTEXAI"] = "1"
    assert resolve_model("google_genai:gemini-2.5-pro", env).backend == "platform"

def test_platform_without_proxy_env_fails_loud():
    with pytest.raises(RuntimeError, match="OPENAI_BASE_URL"):
        resolve_model("platform:gemini-2.5-pro", {})

def test_aistudio_byok_unchanged():
    assert resolve_model("google_genai:gemini-2.5-pro", {"GOOGLE_API_KEY": "k"}).backend == "ai_studio"


# --- SSOT signal: MOLECULE_RESOLVED_PROVIDER (TOP PRECEDENCE) ---------------
# Core's provisioner resolves the provider ONCE and publishes the registry arm
# name here. When set it is authoritative: platform iff value == "platform";
# any other arm is BYOK and must NOT be re-promoted to platform from the model
# namespace. Only when ABSENT do the legacy prefix / GOOGLE_GENAI_USE_VERTEXAI
# triggers apply.

def test_resolved_provider_platform_routes_via_proxy():
    env = dict(PLATFORM_ENV); env["MOLECULE_RESOLVED_PROVIDER"] = "platform"
    # A bare gemini id (no platform: prefix) routes via the proxy purely on the
    # SSOT signal.
    r = resolve_model("gemini-2.5-pro", env)
    assert r.backend == "platform" and r.is_platform
    assert r.model == "openai/gemini-2.5-pro"


def test_resolved_provider_byok_arm_not_promoted_to_platform():
    # SSOT names a byok arm -> NOT platform, even though the model namespace
    # (platform:/vertex:) would otherwise (legacy) say platform. The signal wins
    # and the model falls through to the BYOK AI Studio path.
    env = {"MOLECULE_RESOLVED_PROVIDER": "ai-studio", "GOOGLE_API_KEY": "k"}
    r = resolve_model("vertex:gemini-2.5-pro", env)
    assert r.backend == "ai_studio" and not r.is_platform


def test_resolved_provider_absent_falls_back_to_legacy_prefix():
    # No SSOT signal -> legacy platform: prefix still routes via the proxy.
    r = resolve_model("platform:gemini-2.5-pro", PLATFORM_ENV)
    assert r.backend == "platform"
