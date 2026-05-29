"""Google ADK workspace-template adapter package."""
# The runtime loads this via ADAPTER_MODULE=adapter (top-level import, /app on
# sys.path). This re-export supports package-style import too.
try:
    from adapter import Adapter, GoogleADKAdapter  # top-level (container + tests)
except ImportError:  # pragma: no cover
    from .adapter import Adapter, GoogleADKAdapter  # package context

__all__ = ["Adapter", "GoogleADKAdapter"]
