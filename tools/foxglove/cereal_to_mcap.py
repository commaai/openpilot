import cereal
import json
from mcap.writer import Writer
from mcap.well_known import SchemaEncoding, MessageEncoding
from base64 import b64encode
import argparse
from openpilot.tools.lib.logreader import LogReader
from openpilot.tools.lib.route import Route, SegmentName
from openpilot.tools.lib.helpers import save_log
from urllib.parse import urlparse, parse_qs
import sys
import os
import tempfile
import multiprocessing
import math
from openpilot.selfdrive.test.openpilotci import get_url
from openpilot.tools.foxglove.json_schema import get_event_schemas, rawImage, compressedImage, frameTransform, locationFix, logs
from openpilot.tools.lib.framereader import FrameReader

juggle_dir = os.path.dirname(os.path.realpath(__file__))

DEMO_ROUTE = "a2a0ccea32023010|2023-07-27--13-01-19"

def nan2None(obj):
    if isinstance(obj, dict):
        return {k:nan2None(v) for k,v in obj.items()}
    elif isinstance(obj, list):
        return [nan2None(v) for v in obj]
    elif isinstance(obj, float) and math.isnan(obj):
        return None
    return obj

class Base64Encoder(json.JSONEncoder):
    def encode(self, obj, *args, **kwargs):
        return super().encode(nan2None(obj), *args, **kwargs)
    # pylint: disable=method-hidden
    def default(self, o):
        if isinstance(o, bytes):
            return b64encode(o).decode()
        if math.isnan(o):
            return 0
        return json.JSONEncoder.default(self, o)

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

  fcamera = None
  dcamera = None
  ecamera = None
  if fcam:
    fcamera = []
    for p in fcam_paths:
      fcamera.append(FrameReader(p))
  if dcam:
    dcamera = []
    for p in dcam_paths:
      dcamera.append(FrameReader(p))
  if ecam:
    ecamera = []
    for p in ecam_paths:
      ecamera.append(FrameReader(p))

  with tempfile.NamedTemporaryFile(suffix='.rlog', dir=juggle_dir) as tmp:
    save_log(tmp.name, all_data, compress=False)
    del all_data
    convert_log(f"{str(route_or_segment_name.canonical_name).split('|')[1]}.mcap", tmp.name, fcamera, dcamera, ecamera)

def load_segment(segment_name):
  if segment_name is None:
    return []

  try:
    return list(LogReader(segment_name))
  except (AssertionError, ValueError) as e:
    print(f"Error parsing {segment_name}: {e}")
    return []

def convert_log(name, log_file, fcam=None, dcam=None, ecam=None):
  channel_exclusions = ['logMonoTime', 'valid']
  fcam_index = 0
  dcam_index = 0
  ecam_index = 0
  with open(name, "wb") as f:
    writer = Writer(f)
    writer.start()

    cam_schema_id = 0
    fcam_channel = None
    dcam_channel = None
    ecam_channel = None
    if fcam is not None or dcam is not None or ecam is not None:
      cam_schema_id = writer.register_schema(
        name="foxglove.RawImage",
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(rawImage), "utf-8"),
      )
    if fcam is not None:
      fcam_channel = writer.register_channel(
        topic="/fcam",
        message_encoding=MessageEncoding.JSON,
        schema_id=cam_schema_id,
      )
    if dcam is not None:
      dcam_channel = writer.register_channel(
        topic="/dcam",
        message_encoding=MessageEncoding.JSON,
        schema_id=cam_schema_id,
      )
    if ecam is not None:
      ecam_channel = writer.register_channel(
        topic="/ecam",
        message_encoding=MessageEncoding.JSON,
        schema_id=cam_schema_id,
      )

    type_to_schema = {}
    for k in list(schemas.keys()):
      schema_id = writer.register_schema(
        name=k,
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(schemas[k]), "utf-8"),
      )
      type_to_schema[k] = schema_id

    typeToChannel = {}
    for k in list(schemas.keys()):
      if k in channel_exclusions:
        continue
      typeToChannel[k] = writer.register_channel(
        topic=k,
        message_encoding=MessageEncoding.JSON,
        schema_id=type_to_schema[k],
      )

    compressedImageSchema = writer.register_schema(
        name="foxglove.CompressedImage",
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(compressedImage), "utf-8"),
    )
    thumbnailChannel = writer.register_channel(
      topic="/thumbnail",
      message_encoding=MessageEncoding.JSON,
      schema_id=compressedImageSchema,
    )
    frameTransformSchema = writer.register_schema(
        name="foxglove.FrameTransform",
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(frameTransform), "utf-8"),
    )
    frameTransformChannel = writer.register_channel(
      topic="/frameTransform",
      message_encoding=MessageEncoding.JSON,
      schema_id=frameTransformSchema,
    )
    locationFixSchema = writer.register_schema(
        name="foxglove.LocationFix",
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(locationFix), "utf-8"),
    )
    locationFixChannel = writer.register_channel(
      topic="/liveLocation",
      message_encoding=MessageEncoding.JSON,
      schema_id=locationFixSchema,
    )
    logsSchema = writer.register_schema(
        name="foxglove.Log",
        encoding=SchemaEncoding.JSONSchema,
        data=bytes(json.dumps(logs), "utf-8"),
    )
    logsChannel = writer.register_channel(
      topic="/log",
      message_encoding=MessageEncoding.JSON,
      schema_id=logsSchema,
    )
    logf = open(log_file, 'rb')
    events = cereal.log.Event.read_multiple(logf)

    offset = 0
    for event in events:
      e = event.to_dict()
      if str(event.which) == "initData":
        offset = int(e["initData"]["wallTimeNanos"]) - int(e["logMonoTime"])
      elif str(event.which) == "roadCameraState" and fcam is not None:
        segment = fcam_index // 1200
        idx = fcam_index % 1200
        frame = {
          "timestamp": {
            "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
            "sec": (int(e["logMonoTime"]) + offset) // 1000000000
          },
          "frame_id": str(e["roadCameraState"]["frameId"]),
          "data": bytes(fcam[segment].get(idx, pix_fmt="rgb24")[0]),
          "width": fcam[segment].w,
          "height": fcam[segment].h,
          "encoding": "rgb8",
          "step": fcam[segment].w*3,
        }
        writer.add_message(
            fcam_channel,
            log_time=int(e["logMonoTime"]) + offset,
            data=json.dumps(frame, cls=Base64Encoder).encode("utf-8"),
            publish_time=int(e["logMonoTime"]) + offset,
        )
        fcam_index += 1
      elif str(event.which) == "driverCameraState" and dcam is not None:
        segment = dcam_index // 1200
        idx = dcam_index % 1200
        frame = {
          "timestamp": {
            "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
            "sec": (int(e["logMonoTime"]) + offset) // 1000000000
          },
          "frame_id": str(e["roadCameraState"]["frameId"]),
          "data": bytes(dcam[segment].get(idx, pix_fmt="rgb24")[0]),
          "width": dcam[segment].w,
          "height": dcam[segment].h,
          "encoding": "rgb8",
          "step": dcam[segment].w*3,
        }
        writer.add_message(
            dcam_channel,
            log_time=int(e["logMonoTime"]) + offset,
            data=json.dumps(frame, cls=Base64Encoder).encode("utf-8"),
            publish_time=int(e["logMonoTime"]) + offset,
        )
        dcam_index += 1
      elif str(event.which) == "wideRoadCameraState" and ecam is not None:
        segment = ecam_index // 1200
        idx = ecam_index % 1200
        frame = {
          "timestamp": {
            "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
            "sec": (int(e["logMonoTime"]) + offset) // 1000000000
          },
          "frame_id": str(e["roadCameraState"]["frameId"]),
          "data": bytes(ecam[segment].get(idx, pix_fmt="rgb24")[0]),
          "width": ecam[segment].w,
          "height": ecam[segment].h,
          "encoding": "rgb8",
          "step": ecam[segment].w*3,
        }
        writer.add_message(
            ecam_channel,
            log_time=int(e["logMonoTime"]) + offset,
            data=json.dumps(frame, cls=Base64Encoder).encode("utf-8"),
            publish_time=int(e["logMonoTime"]) + offset,
        )
        ecam_index += 1
      elif str(event.which) == "modelV2":
        position = e["modelV2"]["temporalPose"]["transStd"]
        orientation = e["modelV2"]["temporalPose"]["rotStd"]
        data = {
            "timestamp": {
                "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
                "sec": (int(e["logMonoTime"]) + offset) // 1000000000
            },
            "parent_frame_id": str(e["modelV2"]["frameId"] - 1),
            "child_frame_id": str(e["modelV2"]["frameId"]),
            "translation": {"x":position[0], "y": position[1], "z": position[2]},
            "rotation": toQuaternion(orientation[0], orientation[1], orientation[2])
        }
        writer.add_message(
          frameTransformChannel,
          log_time=int(e["logMonoTime"]) + offset,
          data=json.dumps(data, cls=Base64Encoder).encode("utf-8"),
          publish_time=int(e["logMonoTime"]) + offset,
        )
      elif str(event.which) == "liveLocationKalman":
        data = {
            "timestamp": {
                "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
                "sec": (int(e["logMonoTime"]) + offset) // 1000000000
            },
            "frame_id": e["logMonoTime"],
            "latitude": e["liveLocationKalman"]["positionGeodetic"]["value"][0],
            "longitude": e["liveLocationKalman"]["positionGeodetic"]["value"][1],
            "altitude": e["liveLocationKalman"]["positionGeodetic"]["value"][2],
            "position_covariance_type": 1,
            "position_covariance": [0, 0, 0, 0, 0, 0, 0, 0, 0],
        }
        writer.add_message(
          locationFixChannel,
          log_time=int(e["logMonoTime"]) + offset,
          data=json.dumps(data, cls=Base64Encoder).encode("utf-8"),
          publish_time=int(e["logMonoTime"]) + offset,
        )
      elif str(event.which) == "thumbnail":
        data = {
            "timestamp": {
                "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
                "sec": (int(e["logMonoTime"]) + offset) // 1000000000
            },
            "frame_id": str(e["thumbnail"]["frameId"]),
            "format": "jpeg",
            "data": e["thumbnail"]["thumbnail"],
        }
        writer.add_message(
          thumbnailChannel,
          log_time=int(e["logMonoTime"]) + offset,
          data=json.dumps(data, cls=Base64Encoder).encode("utf-8"),
          publish_time=int(e["logMonoTime"]) + offset,
        )
      elif str(event.which) == "errorLogMessage":
        message = json.loads(e["errorLogMessage"])
        name = "Unknown"
        file = "Unknown"
        line = 0
        level = 0
        if "level" in message:
            if message["level"] == "ERROR":
                level = 4
            elif message["level"] == "WARNING":
                level = 3
            elif message["level"] == "INFO":
                level = 2
        if "ctx" in message and "daemon" in message["ctx"]:
            name = message["ctx"]["daemon"]
        if "filename" in message:
            file = message["filename"]
        if "lineno" in message:
            line = message["lineno"]

        data = {
            "timestamp": {
                "nsec": (int(e["logMonoTime"]) + offset) % 1000000000,
                "sec": (int(e["logMonoTime"]) + offset) // 1000000000
            },
            "level": level,
            "message": e["errorLogMessage"],
            "name": name,
            "file": file,
            "line": line,
        }
        writer.add_message(
          logsChannel,
          log_time=int(e["logMonoTime"]) + offset,
          data=json.dumps(data, cls=Base64Encoder).encode("utf-8"),
          publish_time=int(e["logMonoTime"]) + offset,
        )

      writer.add_message(
          typeToChannel[str(event.which)],
          log_time=int(e["logMonoTime"]) + offset,
          data=json.dumps(e, cls=Base64Encoder).encode("utf-8"),
          publish_time=int(e["logMonoTime"]) + offset,
      )


    writer.finish()

def toQuaternion(roll, pitch, yaw):
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)
    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)

    q = {"w": 0, "x": 0, "y": 0, "z": 0}
    q["w"] = cr * cp * cy + sr * sp * sy
    q["x"] = sr * cp * cy - cr * sp * sy
    q["y"] = cr * sp * cy + sr * cp * sy
    q["z"] = cr * cp * sy - sr * sp * cy

    return q

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



