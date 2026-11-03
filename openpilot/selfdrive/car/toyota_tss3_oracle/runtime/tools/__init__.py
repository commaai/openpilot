"""Repository tooling modules used by deterministic verification tests."""

from __future__ import annotations

from pathlib import Path

#: Absolute path of this source checkout. Importable without any sys.path or
#: PYTHONPATH setup once the project is installed (``uv sync --locked``).
REPO_ROOT = Path(__file__).resolve().parents[1]
