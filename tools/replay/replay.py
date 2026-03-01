#!/usr/bin/env python3
import argparse
import curses
import os
import platform
import resource
import sys

from openpilot.tools.replay.replay_pyx import PyReplay, REPLAY_FLAGS, get_demo_route
from openpilot.tools.replay.consoleui import ConsoleUI, SPEED_ARRAY


def parse_args():
  parser = argparse.ArgumentParser(
    description="Mock openpilot components by publishing logged messages.",
    epilog="Find your routes at https://connect.comma.ai",
  )
  parser.add_argument("route", nargs="?", default="", help="Route string (e.g. a2a0cbe32|2021-01-01--00-00-00)")
  parser.add_argument("-a", "--allow", default="", help="Services to include (comma-separated)")
  parser.add_argument("-b", "--block", default="", help="Services to exclude (comma-separated)")
  parser.add_argument("-c", "--cache", type=int, default=-1, help="Cache <n> segments in memory (default: 5)")
  parser.add_argument("-s", "--start", type=int, default=0, help="Start from <seconds>")
  parser.add_argument("-x", "--playback", type=float, default=-1.0, metavar="SPEED",
                      help=f"Playback speed ({SPEED_ARRAY[0]} to {SPEED_ARRAY[-1]})")
  parser.add_argument("--demo", action="store_true", help="Use a demo route")
  parser.add_argument("--auto", action="store_true", dest="auto_source",
                      help="Auto load route from best available source (no video)")
  parser.add_argument("-d", "--data_dir", default="", help="Local directory with routes")
  parser.add_argument("-p", "--prefix", default="", help="Set OPENPILOT_PREFIX for IPC isolation")
  parser.add_argument("--dcam", action="store_true", help="Load driver camera")
  parser.add_argument("--ecam", action="store_true", help="Load wide road camera")
  parser.add_argument("--no-loop", action="store_true", help="Stop at the end of the route")
  parser.add_argument("--no-cache", action="store_true", help="Turn off local cache")
  parser.add_argument("--qcam", action="store_true", help="Load qcamera")
  parser.add_argument("--no-hw-decoder", action="store_true", help="Disable HW video decoding")
  parser.add_argument("--no-vipc", action="store_true", help="Do not output video")
  parser.add_argument("--all", action="store_true",
                      help="Output all messages including uiDebug, userBookmark")

  # Print help when no arguments provided
  if len(sys.argv) == 1:
    parser.print_help(sys.stderr)
    sys.exit(1)

  return parser.parse_args()


def main():
  args = parse_args()

  route = args.route
  if args.demo:
    route = get_demo_route()
  if not route:
    print("No route provided. Run with --help for usage.", file=sys.stderr)
    return 1

  # macOS has a default fd limit of 256 which is too low
  if platform.system() == "Darwin":
    try:
      soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
      resource.setrlimit(resource.RLIMIT_NOFILE, (min(1024, hard), hard))
    except (ValueError, OSError):
      pass

  flags = REPLAY_FLAGS.REPLAY_FLAG_NONE
  if args.dcam:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_DCAM
  if args.ecam:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_ECAM
  if args.no_loop:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_NO_LOOP
  if args.no_cache:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_NO_FILE_CACHE
  if args.qcam:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_QCAMERA
  if args.no_hw_decoder:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_NO_HW_DECODER
  if args.no_vipc:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_NO_VIPC
  if args.all:
    flags |= REPLAY_FLAGS.REPLAY_FLAG_ALL_SERVICES

  # Use OpenpilotPrefix context manager for proper directory setup/cleanup
  prefix_ctx = None
  if args.prefix:
    from openpilot.common.prefix import OpenpilotPrefix
    prefix_ctx = OpenpilotPrefix(args.prefix, clean_dirs_on_exit=True, shared_download_cache=True)
    prefix_ctx.__enter__()

  try:
    allow = [s.strip() for s in args.allow.split(",") if s.strip()] if args.allow else []
    block = [s.strip() for s in args.block.split(",") if s.strip()] if args.block else []

    replay = PyReplay(route, allow, block, flags=flags, data_dir=args.data_dir, auto_source=args.auto_source)

    if args.cache > 0:
      replay.set_segment_cache_limit(args.cache)
    if args.playback > 0:
      replay.set_speed(max(SPEED_ARRAY[0], min(args.playback, SPEED_ARRAY[-1])))

    if not replay.load():
      print("Failed to load route.", file=sys.stderr)
      return 1

    def curses_main(stdscr):
      ui = ConsoleUI(replay, stdscr)
      replay.start(args.start)
      return ui.exec()

    os.environ.setdefault("TERMINFO_DIRS", "/usr/share/terminfo:/lib/terminfo:/usr/lib/terminfo")
    return curses.wrapper(curses_main)
  finally:
    if prefix_ctx is not None:
      prefix_ctx.__exit__(None, None, None)


if __name__ == "__main__":
  sys.exit(main() or 0)
