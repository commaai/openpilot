"""
pycabana - Launch cabana from Python

Usage:
  python -m openpilot.tools.cabana.pycabana [cabana args...]
  python -m openpilot.tools.cabana.pycabana --demo
  python -m openpilot.tools.cabana.pycabana "a]2a0ccea32023010|2023-07-27--13-01-19"

Build cabana first with:
  scons tools/cabana
"""
import sys
from openpilot.tools.cabana.pycabana import main

if __name__ == "__main__":
  sys.exit(main())
