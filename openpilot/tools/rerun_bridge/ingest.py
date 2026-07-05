"""Ingest an openpilot route into a Rerun recording."""

from __future__ import annotations

import logging
from dataclasses import dataclass

from openpilot.tools.cabana.dbc.generate_dbc_json import generate_dbc_dict
from openpilot.tools.lib.logreader import LogReader, ReadMode
from openpilot.tools.lib.route import Route
from openpilot.tools.rerun_bridge.blueprint import make_blueprint
from openpilot.tools.rerun_bridge.can_decode import decode_can_messages
from openpilot.tools.rerun_bridge.custom_series import apply_layout_custom_series
from openpilot.tools.rerun_bridge.extract import EventExtractor, finalize_series
from openpilot.tools.rerun_bridge.logs import extract_logs_and_timeline
from openpilot.tools.rerun_bridge.rerun_log import log_gps, log_series, log_text_logs, log_timeline
from openpilot.tools.rerun_bridge.video import log_camera_streams

logger = logging.getLogger(__name__)

DEMO_ROUTE = "5beb9b58bd12b691/0000010a--a51155e496"


@dataclass
class IngestStats:
  messages: int = 0
  series_paths: int = 0
  log_entries: int = 0
  gps_points: int = 0
  camera_frames: dict[str, int] | None = None


def _detect_dbc(route: Route, lr: LogReader) -> str:
  try:
    cp = lr.first("carParams")
    if cp is None:
      return ""
    fingerprint = str(cp.carFingerprint)
    return generate_dbc_dict().get(fingerprint, "")
  except Exception:
    return ""


def ingest_route(
  rr,
  route_name: str,
  *,
  data_dir: str | None = None,
  layout_name: str | None = "tuning",
  selector: str = "qlog",
  include_video: bool = True,
  video_frame_skip: int = 5,
  include_deprecated: bool = True,
  include_can: bool = True,
) -> IngestStats:
  route = Route(route_name, data_dir=data_dir)
  mode = ReadMode.QLOG if selector == "qlog" else ReadMode.RLOG
  if data_dir:
    paths = route.qlog_paths() if selector == "qlog" else route.log_paths()
    paths = [path for path in paths if path]
    if not paths:
      raise RuntimeError(f"no {selector} logs found in {data_dir}")
    lr = LogReader(paths, default_mode=mode)
  else:
    lr = LogReader(route_name, default_mode=mode)

  logger.info("loading %s (%s)", route_name, selector)
  messages = list(lr)
  stats = IngestStats(messages=len(messages))

  extractor = EventExtractor(include_deprecated=include_deprecated)
  store = extractor.process_events(messages)
  dbc_name = _detect_dbc(route, lr)
  if include_can and dbc_name:
    logger.info("decoding CAN with DBC %s", dbc_name)
    decode_can_messages(store, dbc_name)

  series = finalize_series(store, include_deprecated=include_deprecated)
  if layout_name:
    try:
      from openpilot.tools.rerun_bridge.blueprint import load_layout
      layout = load_layout(layout_name)
      series = apply_layout_custom_series(layout, series)
    except FileNotFoundError:
      logger.warning("layout %s not found; skipping custom series", layout_name)

  stats.series_paths = log_series(rr, series)
  logs, timeline = extract_logs_and_timeline(messages)
  stats.log_entries = log_text_logs(rr, logs)
  log_timeline(rr, timeline)
  stats.gps_points = log_gps(rr, series)

  if include_video:
    stats.camera_frames = log_camera_streams(rr, route, series, frame_skip=video_frame_skip)

  blueprint = make_blueprint(layout_name)
  rr.send_blueprint(blueprint, make_active=True, make_default=True)
  return stats