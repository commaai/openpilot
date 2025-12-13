"""pycabana - Python wrapper for cabana CAN bus analyzer"""
import os
import subprocess
import sys

CABANA_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "cabana")


def run_cabana(args: list[str] | None = None) -> int:
  """Launch the C++ cabana executable.

  Args:
    args: Command line arguments (without program name). If None, uses sys.argv[1:]

  Returns:
    Exit code from cabana
  """
  if not os.path.exists(CABANA_PATH):
    print(f"cabana not found at {CABANA_PATH}")
    print("Build with: scons tools/cabana")
    return 1

  if args is None:
    args = sys.argv[1:]

  cmd = [CABANA_PATH] + list(args)
  return subprocess.call(cmd)


def main():
  """Entry point when run as a module."""
  return run_cabana()
