"""pycabana - Pure PySide6 rewrite of cabana CAN bus analyzer"""

__all__ = ["main"]


def main():
  """Entry point - import lazily to avoid import-time Qt issues."""
  from openpilot.tools.cabana.pycabana.main import main as _main

  return _main()
