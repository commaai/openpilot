#!/usr/bin/env python3
"""CLI for the openpilot → Rerun bridge."""

from __future__ import annotations

import argparse
import logging
import sys

import rerun as rr

from openpilot.tools.rerun_bridge.ingest import DEMO_ROUTE, ingest_route

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger("rerun_bridge")


def parse_args(argv: list[str]) -> argparse.Namespace:
  parser = argparse.ArgumentParser(description="Ingest an openpilot route into Rerun")
  parser.add_argument("route", nargs="?", help="Route name (dongle/route or dongle/route/start:end)")
  parser.add_argument("--demo", action="store_true", help="Use the demo route")
  parser.add_argument("--data-dir", help="Local route data directory")
  parser.add_argument("--layout", default="tuning", help="Jotpluggler layout preset name")
  parser.add_argument("--selector", choices=["qlog", "rlog"], default="qlog", help="Log type to ingest")
  parser.add_argument("--output", "-o", help="Save recording to .rrd instead of opening the viewer")
  parser.add_argument("--spawn", action="store_true", help="Open the Rerun viewer after ingest")
  parser.add_argument("--no-video", action="store_true", help="Skip camera video ingest")
  parser.add_argument("--no-can", action="store_true", help="Skip DBC CAN signal decode")
  parser.add_argument("--video-frame-skip", type=int, default=15, help="Log every Nth camera frame")
  parser.add_argument("--app-id", default="openpilot", help="Rerun application id")
  return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
  args = parse_args(argv or sys.argv[1:])
  route = DEMO_ROUTE if args.demo else (args.route or "")
  if not route:
    logger.error("route required (or pass --demo)")
    return 2

  spawn = args.spawn or (not args.output)
  rr.init(args.app_id, recording_id=route.replace("/", "_"), spawn=spawn)
  if args.output:
    rr.save(args.output)

  stats = ingest_route(
    rr,
    route,
    data_dir=args.data_dir,
    layout_name=args.layout,
    selector=args.selector,
    include_video=not args.no_video,
    video_frame_skip=max(1, args.video_frame_skip),
    include_can=not args.no_can,
  )

  logger.info(
    "ingest complete: %d messages, %d series, %d logs, %d gps points, cameras=%s",
    stats.messages,
    stats.series_paths,
    stats.log_entries,
    stats.gps_points,
    stats.camera_frames,
  )

  if args.output:
    logger.info("saved %s", args.output)
  return 0


if __name__ == "__main__":
  raise SystemExit(main())