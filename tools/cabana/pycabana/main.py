"""pycabana - PySide6 CAN bus analyzer.

Usage:
  python cabana.py [route]
  python cabana.py --demo
"""

import argparse
import signal
import sys
import time

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
  parser.add_argument(
    "--strict",
    action="store_true",
    help="Exit with error if UI thread is blocked (for testing)",
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

  stream = ReplayStream()
  exit_code = 0

  def sigint_handler(*_):
    stream.stop()
    app.quit()

  signal.signal(signal.SIGINT, sigint_handler)

  window = MainWindow(stream)
  window.show()

  # Strict mode: detect UI thread blocking (only after loading completes)
  timer = None
  if args.strict:
    last_tick = [0.0]
    max_allowed_delay = 0.5  # 500ms
    strict_enabled = [False]

    def enable_strict():
      strict_enabled[0] = True
      last_tick[0] = time.monotonic()

    stream.loadFinished.connect(enable_strict)

    def check_responsiveness():
      nonlocal exit_code
      now = time.monotonic()
      if not strict_enabled[0]:
        return
      delay = now - last_tick[0]
      if delay > max_allowed_delay:
        print(f"STRICT MODE: UI thread blocked for {delay:.2f}s, exiting")
        exit_code = 2
        stream.stop()
        app.quit()
      last_tick[0] = now

    timer = QTimer()
    timer.timeout.connect(check_responsiveness)
    timer.start(100)  # Check every 100ms

  print(f"Loading route: {route}")
  if not stream.loadRoute(route):
    print(f"Failed to start loading route: {route}")
    return 1

  app.exec()
  return exit_code


if __name__ == "__main__":
  sys.exit(main())
