"""
Harness-only conftest for ``system/tests/support/tests``.

Cross-cutting system support tests live in ``test_*.py`` modules here. If the
suite is temporarily empty, we map ``NO_TESTS_COLLECTED`` to exit 0 (same hook
as ``selfdrive/test/support/tests``) so scoped runs stay green.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
  from _pytest.main import ExitCode

  if exitstatus == ExitCode.NO_TESTS_COLLECTED:
    session.exitstatus = ExitCode.OK
