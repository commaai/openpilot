#!/usr/bin/env python3
import argparse
import resource
import sys

from openpilot.tools.replay.consoleui import ConsoleUI
from openpilot.tools.replay.replay import Replay, DEMO_ROUTE
from openpilot.tools.replay.seg_mgr import ReplayFlags

HELP_TEXT = """
Usage: python -m tools.replay [options] [route]

Options:
  -a, --allow        Whitelist of services to send (comma-separated)
  -b, --block        Blacklist of services to send (comma-separated)
  -c, --cache        Cache <n> segments in memory. Default is 5
  -s, --start        Start from <seconds>
  -x, --playback     Playback <speed>
      --demo         Use a demo route instead of providing your own
      --auto         Auto load the route from the best available source (no video)
  -d, --data_dir     Local directory with routes
  -p, --prefix       Set OPENPILOT_PREFIX
      --dcam         Load driver camera
      --ecam         Load wide road camera
      --no-loop      Stop at the end of the route
      --no-cache     Turn off local cache
      --qcam         Load qcamera
      --no-vipc      Do not output video
      --all          Output all messages including bookmarkButton, uiDebug, userBookmark
  -h, --help         Show this help message
"""


def main():
  # Increase file descriptor limit on macOS
  if sys.platform == 'darwin':
    try:
      resource.setrlimit(resource.RLIMIT_NOFILE, (1024, 1024))
    except Exception:
      pass

  parser = argparse.ArgumentParser(
    description='openpilot replay tool',
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog=HELP_TEXT
  )
  parser.add_argument('route', nargs='?', default='', help='Route to replay')
  parser.add_argument('-a', '--allow', type=str, default='', help='Whitelist of services (comma-separated)')
  parser.add_argument('-b', '--block', type=str, default='', help='Blacklist of services (comma-separated)')
  parser.add_argument('-c', '--cache', type=int, default=-1, help='Number of segments to cache')
  parser.add_argument('-s', '--start', type=int, default=0, help='Start from <seconds>')
  parser.add_argument('-x', '--playback', type=float, default=-1, help='Playback speed')
  parser.add_argument('-d', '--data_dir', type=str, default='', help='Local directory with routes')
  parser.add_argument('-p', '--prefix', type=str, default='', help='OPENPILOT_PREFIX')
  parser.add_argument('--demo', action='store_true', help='Use demo route')
  parser.add_argument('--auto', action='store_true', help='Auto load from best source')
  parser.add_argument('--dcam', action='store_true', help='Load driver camera')
  parser.add_argument('--ecam', action='store_true', help='Load wide road camera')
  parser.add_argument('--no-loop', action='store_true', help='Stop at end of route')
  parser.add_argument('--no-cache', action='store_true', help='Disable local cache')
  parser.add_argument('--qcam', action='store_true', help='Load qcamera')
  parser.add_argument('--no-vipc', action='store_true', help='Do not output video')
  parser.add_argument('--all', action='store_true', help='Output all messages')
  parser.add_argument('--headless', action='store_true', help='Run without UI (for testing)')

  args = parser.parse_args()

  # Determine route
  route = args.route
  if args.demo:
    route = DEMO_ROUTE
  if not route:
    print("No route provided. Use --help for usage information.")
    return 1

  # Parse flags
  flags = ReplayFlags.NONE
  if args.dcam:
    flags |= ReplayFlags.DCAM
  if args.ecam:
    flags |= ReplayFlags.ECAM
  if args.no_loop:
    flags |= ReplayFlags.NO_LOOP
  if args.no_cache:
    flags |= ReplayFlags.NO_FILE_CACHE
  if args.qcam:
    flags |= ReplayFlags.QCAMERA
  if args.no_vipc:
    flags |= ReplayFlags.NO_VIPC
  if args.all:
    flags |= ReplayFlags.ALL_SERVICES

  # Parse allow/block lists
  allow = [s.strip() for s in args.allow.split(',') if s.strip()]
  block = [s.strip() for s in args.block.split(',') if s.strip()]

  # Set prefix if provided
  if args.prefix:
    import os
    os.environ['OPENPILOT_PREFIX'] = args.prefix

  # Create replay instance
  replay = Replay(
    route=route,
    allow=allow,
    block=block,
    flags=flags,
    data_dir=args.data_dir,
    auto_source=args.auto
  )

  if args.cache > 0:
    replay.segment_cache_limit = args.cache

  if args.playback > 0:
    replay.speed = max(0.2, min(8.0, args.playback))

  if not replay.load():
    return 1

  replay.start(args.start)

  if args.headless:
    # Headless mode - run for a bit and exit
    import time
    print("Running in headless mode...")
    print(f"  stream_thread: {replay._stream_thread}")
    print(f"  event_data segments: {list(replay.get_event_data().segments.keys())}")
    try:
      for i in range(10):
        time.sleep(1)
        event_data = replay.get_event_data()
        print(f"  {replay.current_seconds:.1f}s / {replay.max_seconds:.1f}s, events: {len(event_data.events)}, segs: {list(event_data.segments.keys())}")
    except KeyboardInterrupt:
      pass
    print("Done")
    return 0

  # Create UI and start replay
  console_ui = ConsoleUI(replay)
  return console_ui.exec()


if __name__ == '__main__':
  sys.exit(main())
