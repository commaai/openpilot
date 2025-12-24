#!/usr/bin/env python3
"""pycabana - PySide6 CAN bus analyzer.

Usage:
  python cabana.py [route]
  python cabana.py --demo
"""

import sys

from openpilot.tools.cabana.pycabana.main import main

if __name__ == "__main__":
  sys.exit(main())
