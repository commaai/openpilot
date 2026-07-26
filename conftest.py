"""Pytest runner configuration for the unittest suite."""

import os

# Heavy CI-only tests are invoked explicitly by their dedicated jobs.
collect_ignore = [
  "openpilot/selfdrive/test/process_replay/test_processes.py",
  "openpilot/selfdrive/test/process_replay/test_regen.py",
  "openpilot/tools/sim/",
]


def pytest_collection_modifyitems(items):
  if os.environ.get("SKIP_SLOW"):
    items[:] = [item for item in items if not getattr(getattr(item, "cls", None), "SLOW_TEST", False)]
