import cereal
import json
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
import argparse
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentName
from openpilot.tools.lib.helpers import save_log
from urllib.parse import urlparse, parse_qs
import sys
import os
import tempfile
import multiprocessing
from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.foxglove.json_schema import get_event_schemas
from openpilot.tools.foxglove.foxglove_schemas import RAW_IMAGE, COMPRESSED_IMAGE, FRAME_TRANSFORM, LOCATION_FIX, LOG
from openpilot.tools.foxglove.utils import register_schema, register_channel, register, message, toQuaternion
from openpilot.tools.lib.framereader import FrameReader
from openpilot.tools.foxglove.transforms import transform_camera, TRANSFORMERS
import numpy as np

juggle_dir = os.path.dirname(os.path.realpath(__file__))

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"

schemas = get_event_schemas()

def juggle_route(route_or_segment_name, segment_count, qlog, ci=False, fcam=False, dcam=False, ecam=False):
  segment_start = 0
  fcam_paths = []
  dcamera_paths = []
  ecamera_paths = []
  if 'cabana' in route_or_segment_name:
    query = parse_qs(urlparse(route_or_segment_name).query)
    route_or_segment_name = query["route"][0]

  if route_or_segment_name.startswith(("http://", "https://", "cd:/")) or os.path.isfile(route_or_segment_name):
    logs = [route_or_segment_name]
  elif ci:
    route_or_segment_name = SegmentName(route_or_segment_name, allow_route_name=True)
    route = route_or_segment_name.route_name.canonical_name
    segment_start = max(route_or_segment_name.segment_num, 0)
    logs = [get_url(route, i) for i in range(100)]  # Assume there not more than 100 segments
  else:
    route_or_segment_name = SegmentName(route_or_segment_name, allow_route_name=True)
    segment_start = max(route_or_segment_name.segment_num, 0)

    if route_or_segment_name.segment_num != -1 and segment_count is None:
      segment_count = 1

    r = Route(route_or_segment_name.route_name.canonical_name, route_or_segment_name.data_dir)
    fcam_paths = r.camera_paths()
    dcam_paths = r.dcamera_paths()
    ecam_paths = r.ecamera_paths()
    logs = r.qlog_paths() if qlog else r.log_paths()

  segment_end = segment_start + segment_count if segment_count else None
  logs = logs[segment_start:segment_end]
  fcam_paths = fcam_paths[segment_start:segment_end]
  dcam_paths = dcam_paths[segment_start:segment_end]
  ecam_paths = ecam_paths[segment_start:segment_end]

  if None in logs:
    resp = input(f"{logs.count(None)}/{len(logs)} of the rlogs in this segment are missing, would you like to fall back to the qlogs? (y/n) ")
    if resp == 'y':
      logs = r.qlog_paths()[segment_start:segment_end]
    else:
      print("Please try a different route or segment")
      return

  all_data = []
  with multiprocessing.Pool(24) as pool:
    for d in pool.map(load_segment, logs):
      all_data += d

  cams = {}
  if fcam:
    cams["roadCameraState"] = { "data": [], "idx": 0 }
    for p in fcam_paths:
      cams["roadCameraState"]["data"].append(FrameReader(p))
  if dcam:
    cams["driverCameraState"] = { "data": [], "idx": 0 }
    for p in dcam_paths:
      cams["driverCameraState"]["data"].append(FrameReader(p))
  if ecam:
    cams["wideRoadCameraState"] = { "data": [], "idx": 0 }
    for p in ecam_paths:
      cams["wideRoadCameraState"]["data"].append(FrameReader(p))

  with tempfile.NamedTemporaryFile(suffix='.rlog', dir=juggle_dir) as tmp:
    save_log(tmp.name, all_data, compress=False)
    del all_data
    convert_log(f"{str(route_or_segment_name.canonical_name).split('|')[1]}.mcap", tmp.name, cams)

def load_segment(segment_name):
  if segment_name is None:
    return []

  try:
    return list(LogReader(segment_name))
  except (AssertionError, ValueError) as e:
    print(f"Error parsing {segment_name}: {e}")
    return []

def convert_log(name, log_file, cams):
  channel_exclusions = ['logMonoTime', 'valid']

  with open(name, "wb") as f:
    writer = Writer(f)
    writer.start()

    cam_schema_id = 0
    cam_channels = {}
    if len(cams.keys()) > 0:
      cam_schema_id = register_schema(writer, "foxglove.RawImage", RAW_IMAGE)
    if "roadCameraState" in cams:
      cam_channels["roadCameraState"] = register_channel(writer, "/fcam", cam_schema_id)
    if "driverCameraState" in cams:
      cam_channels["driverCameraState"] = register_channel(writer, "/dcam", cam_schema_id)
    if "wideRoadCameraState" in cams:
      cam_channels["wideRoadCameraState"] = register_channel(writer, "/ecam", cam_schema_id)

    type_to_schema = {}
    for k in list(schemas.keys()):
      schema_id = register_schema(writer, k, schemas[k])
      type_to_schema[k] = schema_id

    typeToChannel = {}
    for k in list(schemas.keys()):
      if k in channel_exclusions:
        continue
      typeToChannel[k] = register_channel(writer, k, type_to_schema[k])

    compressedImageSchema, thumbnailChannel = register(writer, "/thumbnail", "foxglove.CompressedImage", COMPRESSED_IMAGE)
    frameTransformSchema, frameTransformChannel = register(writer, "/frameTransform", "foxglove.FrameTransform", FRAME_TRANSFORM)
    locationFixSchema, liveLocationChannel = register(writer, "/liveLocation", "foxglove.LocationFix", LOCATION_FIX)
    logsSchema, logsChannel = register(writer, "/log", "foxglove.Log", LOG)

    channel_map = {
      "thumbnail": thumbnailChannel,
      "modelV2": frameTransformChannel,
      "liveLocationKalman": liveLocationChannel,
      "errorLogMessage": logsChannel,
      "logMessage": logsChannel,
    }

    logf = open(log_file, 'rb')
    events = cereal.log.Event.read_multiple(logf)

    event_dicts = [(event.to_dict(), str(event.which)) for event in events]

    offset = 0
    for idx in np.arange(0, len(event_dicts)):
      e, w = event_dicts[idx]
      if w == "initData":
        offset = int(e["initData"]["wallTimeNanos"]) - int(e["logMonoTime"])
      elif w in cams:
        segment = cams[w]["index"] // 1200
        idx = cams[w]["index"] % 1200
        frame = transform_camera(e, offset, w, cams[w]["data"][segment], idx)
        message(writer, cam_channels[w], e, offset, frame)
        cams[w]["index"] += 1
      elif w in TRANSFORMERS:
        data = TRANSFORMERS[w](e, offset)
        message(writer, channel_map[w], e, offset, data)

      message(writer, typeToChannel[w], e, offset, e)


    writer.finish()

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description="A helper to convert openpilot routes to mcap files for foxglove studio",
                                   formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("--demo", action="store_true", help="Use the demo route instead of providing one")
  parser.add_argument("--qlog", action="store_true", help="Use qlogs")
  parser.add_argument("--ci", action="store_true", help="Download data from openpilot CI bucket")
  parser.add_argument("--fcam", action="store_true", help="Include fcamera data")
  parser.add_argument("--dcam", action="store_true", help="Include dcamera data")
  parser.add_argument("--ecam", action="store_true", help="Include ecamera data")
  parser.add_argument("route_or_segment_name", nargs='?', help="The route or segment name to plot (cabana share URL accepted)")
  parser.add_argument("segment_count", type=int, nargs='?', help="The number of segments to plot")

  if len(sys.argv) == 1:
    parser.print_help()
    sys.exit()
  args = parser.parse_args()

  route_or_segment_name = DEMO_ROUTE if args.demo else args.route_or_segment_name.strip()
  juggle_route(route_or_segment_name, args.segment_count, args.qlog, args.ci, args.fcam, args.dcam, args.ecam)
