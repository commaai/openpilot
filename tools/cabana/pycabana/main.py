"""pycabana - PySide2 CAN bus analyzer.

Usage:
  python -m openpilot.tools.cabana.pycabana [route]
  python -m openpilot.tools.cabana.pycabana --demo
"""

import argparse
import sys

from PySide2.QtWidgets import QApplication

from openpilot.tools.cabana.pycabana.mainwindow import MainWindow
from openpilot.tools.cabana.pycabana.streams.replay import ReplayStream


# Demo route for testing
DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"


def main():
  # Parse arguments
  parser = argparse.ArgumentParser(
    description="pycabana - PySide2 CAN bus analyzer",
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

  # Determine route to load
  route = args.route
  if args.demo:
    route = DEMO_ROUTE
  elif not route:
    print("Usage: python -m openpilot.tools.cabana.pycabana [route]")
    print("       python -m openpilot.tools.cabana.pycabana --demo")
    print("\nNo route specified. Use --demo to load a demo route.")
    return 1

  # Create Qt application
  app = QApplication(sys.argv)
  app.setApplicationName("pycabana")
  app.setOrganizationName("comma.ai")

  # Create stream and start loading
  stream = ReplayStream()

  # Create main window
  window = MainWindow(stream)
  window.show()

  # Start loading route (async via QThread)
  print(f"Loading route: {route}")
  if not stream.loadRoute(route):
    print(f"Failed to start loading route: {route}")
    return 1

  # Run event loop
  return app.exec_()


if __name__ == "__main__":
  sys.exit(main())
