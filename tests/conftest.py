"""Make the template root importable from tests/.

The repo keeps adapter.py / _routing.py / google_adk_executor.py at the root
(not under a package dir), so inject the root on sys.path. tests/pytest.ini
anchors pytest's rootdir here so the root __init__.py (which does a
package-relative ``from .adapter import`` for runtime discovery) is never
collected.

No molecule_runtime / a2a stubs are installed. After the tenant-agent BUG 3
migration, google_adk_executor.py inherits the shared ``SubprocessA2AExecutor``
base from molecule-ai-workspace-runtime, so the session/history contract and the
pure-helper tests must exercise the REAL base (installed in CI's
Adapter-unit-tests job via the private PyPI index). When the runtime is absent —
or too old to carry the base (pre runtime #222) — those two test modules
``pytest.importorskip`` themselves at import time, while the runtime-independent
_routing / Dockerfile tests keep running. This mirrors template-openclaw #139,
which dropped its per-file stubs for the same reason.
"""

import os
import sys

# adapter.py / _routing.py / google_adk_executor.py live in the parent dir of
# tests/ (template root). pytest's --import-mode=importlib + the tests/pytest.ini
# rootdir anchor means the parent isn't on sys.path automatically. Add it once so
# every test file can do `from adapter import ...` / `from _routing import ...`.
_PARENT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PARENT_DIR not in sys.path:
    sys.path.insert(0, _PARENT_DIR)
