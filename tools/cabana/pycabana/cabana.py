#!/usr/bin/env python3
"""pycabana - PySide6 CAN bus analyzer.

Usage:
  python cabana.py [route]
  python cabana.py --demo
"""

import sys


def main():
  from openpilot.tools.cabana.pycabana.main import main as _main

  return _main()


if __name__ == "__main__":
  sys.exit(main())
