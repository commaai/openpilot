"""pycabana - PySide6 CAN bus analyzer.

Usage:
  python cabana.py [route]
  python cabana.py --demo
"""

import argparse
import signal
import sys

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"


def main():
  parser = argparse.ArgumentParser(
    description="pycabana - PySide6 CAN bus analyzer",
    formatter_class=argparse.RawDescriptionHelpFormatter,
  )
  parser.add_argument(
    "route",
    nargs="?",
    help="Route to load (e.g., 'a2a0ccea32023010|2023-07-27--13-01-19')",
  )
  parser.add_argument(
    "--demo",
    action="store_true",
    help="Load demo route",
  )
  parser.add_argument(
    "--dbc",
    help="DBC file to use for decoding (not yet implemented)",
  )

  args = parser.parse_args()

  route = args.route
  if args.demo:
    route = DEMO_ROUTE
  elif not route:
    parser.print_help()
    print("\nNo route specified. Use --demo to load a demo route.")
    return 1

  from PySide6.QtCore import QTimer
  from PySide6.QtWidgets import QApplication

  from openpilot.tools.cabana.pycabana.mainwindow import MainWindow
  from openpilot.tools.cabana.pycabana.streams.replay import ReplayStream

  app = QApplication(sys.argv)
  app.setApplicationName("pycabana")
  app.setOrganizationName("comma.ai")

  # Handle ctrl-c
  signal.signal(signal.SIGINT, signal.SIG_DFL)

  stream = ReplayStream()
  window = MainWindow(stream)
  window.show()

  print(f"Loading route: {route}")
  if not stream.loadRoute(route):
    print(f"Failed to start loading route: {route}")
    return 1

  return app.exec()


if __name__ == "__main__":
  sys.exit(main())
