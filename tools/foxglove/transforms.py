from openpilot.tools.foxglove.utils import toQuaternion

def timestamp(event, offset):
  return {
    "nsec": (int(event["logMonoTime"]) + offset) % 1000000000,
    "sec": (int(event["logMonoTime"]) + offset) // 1000000000
  }


def transform_camera(event, offset, camera, segment, frame):
  frame = {
    "timestamp": timestamp(event, offset),
    "frame_id": str(event[camera]["frameId"]),
    "data": bytes(segment.get(frame, pix_fmt="rgb24")[0]),
    "width": segment.w,
    "height": segment.h,
    "encoding": "rgb8",
    "step": segment.w*3,
  }

def modelV2(event, offset):
  position = event["modelV2"]["temporalPose"]["transStd"]
  orientation = event["modelV2"]["temporalPose"]["rotStd"]
  return {
    "timestamp": timestamp(event, offset),
    "parent_frame_id": str(event["modelV2"]["frameId"] - 1),
    "child_frame_id": str(event["modelV2"]["frameId"]),
    "translation": {"x":position[0], "y": position[1], "z": position[2]},
    "rotation": toQuaternion(orientation[0], orientation[1], orientation[2])
  }

def liveLocationKalman(event, offset):
  return {
    "timestamp": timestamp(event, offset),
    "frame_id": event["logMonoTime"],
    "latitude": event["liveLocationKalman"]["positionGeodetic"]["value"][0],
    "longitude": event["liveLocationKalman"]["positionGeodetic"]["value"][1],
    "altitude": event["liveLocationKalman"]["positionGeodetic"]["value"][2],
    "position_covariance_type": 1,
    "position_covariance": [0, 0, 0, 0, 0, 0, 0, 0, 0],
  }

def thumbnail(event, offset):
  return {
    "timestamp": timestamp(event, offset),
    "frame_id": str(event["thumbnail"]["frameId"]),
    "format": "jpeg",
    "data": event["thumbnail"]["thumbnail"],
  }

def errorLogMessage(event, offset):
  log_message = json.loads(event["errorLogMessage"])
  name = "Unknown"
  file = "Unknown"
  line = 0
  level = 4
  if "ctx" in log_message and "daemon" in log_message["ctx"]:
    name = log_message["ctx"]["daemon"]
    if "filename" in log_message:
      file = log_message["filename"]
      if "lineno" in log_message:
        line = log_message["lineno"]

        data = {
          "timestamp": {
            "nsec": (int(event["logMonoTime"]) + offset) % 1000000000,
            "sec": (int(event["logMonoTime"]) + offset) // 1000000000
          },
          "level": level,
          "message": event["errorLogMessage"],
          "name": name,
          "file": file,
          "line": line,
        }

def logMessage(event, offset):
  log_message = json.loads(event["errorLogMessage"])
  name = "Unknown"
  file = "Unknown"
  line = 0
  level = 2

  if "level" in message:
    if message["level"] == "ERROR":
      level = 4
    elif message["level"] == "WARNING":
      level = 3
    elif message["level"] == "INFO":
      level = 2
    elif message["level"] == "DEBUG":
      level = 1

  if "ctx" in log_message and "daemon" in log_message["ctx"]:
    name = log_message["ctx"]["daemon"]
    if "filename" in log_message:
      file = log_message["filename"]
      if "lineno" in log_message:
        line = log_message["lineno"]

        data = {
          "timestamp": {
            "nsec": (int(event["logMonoTime"]) + offset) % 1000000000,
            "sec": (int(event["logMonoTime"]) + offset) // 1000000000
          },
          "level": level,
          "message": event["errorLogMessage"],
          "name": name,
          "file": file,
          "line": line,
        }

TRANSFORMERS = {
  "modelV2": modelV2,
  "liveLocationKalman": liveLocationKalman,
  "thumbnail": thumbnail,
  "errorLogMessage": errorLogMessage,
  "logMessage": logMessage
}
