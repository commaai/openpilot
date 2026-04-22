"""
Harness-only conftest for ``selfdrive/test/support/tests``.

Pytest exits with code 5 when nothing is collected; we map that to success so
scoped runs stay green if this directory is temporarily empty again.
"""

from __future__ import annotations

import pytest


@pytest.hookimpl(trylast=True)
def pytest_sessionfinish(session, exitstatus):
  from _pytest.main import ExitCode

  if exitstatus == ExitCode.NO_TESTS_COLLECTED:
    session.exitstatus = ExitCode.OK
