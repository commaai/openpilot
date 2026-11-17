#!/usr/bin/env python3
"""DashC smart event classification.

Classifies the raw output of the already-running driver-monitoring model
(dmonitoringmodeld -> driverStateV2) into severity/category tags and writes them
into the active segment. The dashcam uploader reads the tag and prefixes the
Drive filename so unsafe events are easy to find and review.

This deliberately uses driverStateV2 directly: in dashcam mode dmonitoringd's
policy is not fed carState/modelV2, so its derived driverMonitoringState is not
available. No car model (modeld), car control, or extra camera is involved.
"""
import json
import os

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware.hw import Paths
from openpilot.common.realtime import config_realtime_process
from openpilot.system.loggerd.uploader import listdir_by_creation

EVENT_TAG = "dashcam_event.json"


def classify_driver_state(msg):
  """Return (severity 0-3, categories) for one driverStateV2 sample.

  Thresholds mirror the stock openpilot driver-monitoring policy so the tags
  mean the same thing as the car's own distraction/eye/pose/phone/sleep states.
  """
  # Thresholds kept local so this function stays a pure, standalone classifier.
  def tier(value, thresholds):
    for level, threshold in enumerate(thresholds):
      if value < threshold:
        return level
    return len(thresholds)

  face_threshold = 0.7
  eye_threshold = 0.65
  blink_thresholds = (0.865, 0.92, 0.96)
  phone_thresholds = (0.5, 0.7, 0.85)
  sleep_thresholds = (0.75, 0.85, 0.93)
  pitch_offset, yaw_offset = 0.011, 0.075
  pitch_natural, yaw_natural = 0.449, 0.402

  driver = msg.rightDriverData if msg.wheelOnRightProb > 0.5 else msg.leftDriverData
  if driver.faceProb <= face_threshold:
    return 0, ()

  # Eyes closed: blend only eyes the model considers open enough to judge.
  left_blink = driver.leftBlinkProb if driver.leftEyeProb > eye_threshold else 0.0
  right_blink = driver.rightBlinkProb if driver.rightEyeProb > eye_threshold else 0.0
  blink = (left_blink + right_blink) * 0.5

  # Off-road pose: normalized deviation from the natural head orientation.
  orientation = driver.faceOrientation
  pitch = abs(orientation[0] - pitch_offset) / pitch_natural if len(orientation) > 0 else 0.0
  yaw = abs(orientation[1] - yaw_offset) / yaw_natural if len(orientation) > 1 else 0.0
  pose = max(pitch, yaw)

  scores = {
    "phone": tier(driver.phoneProb, phone_thresholds),
    "sleep": tier(driver.sleepProb, sleep_thresholds),
    "eye": tier(blink, blink_thresholds),
    "pose": tier(pose, (1.0, 1.3, 1.6)),
  }
  categories = tuple(key for key, score in scores.items() if score > 0)
  return max(scores.values()), categories


def format_tag(severity: int, categories: tuple[str, ...]) -> str:
  """Build the Drive filename prefix, e.g. '[HIGH-PHONE,DROWSY]--'."""
  names = {"phone": "PHONE", "sleep": "DROWSY", "eye": "EYES_CLOSED", "pose": "LOOK_AWAY"}
  levels = ("NONE", "LOW", "MED", "HIGH")
  if severity <= 0 or not categories:
    return ""
  level = levels[severity] if 0 <= severity < len(levels) else "NONE"
  return f"[{level}-{','.join(names.get(c, c.upper()) for c in categories)}]--"


def write_tag(path: str, prefix: str, severity: int, categories):
  """Atomically write the tag for a segment so the uploader never reads a partial file."""
  tmp = path + ".tmp"
  with open(tmp, "w") as f:
    json.dump({"prefix": prefix, "severity": severity, "categories": categories}, f)
  os.replace(tmp, path)


def events_thread():
  config_realtime_process([0, 1, 2, 3], 5)

  sm = messaging.SubMaster(["driverStateV2"], poll="driverStateV2")
  root = Paths.log_root()
  segment = None
  severity, categories = 0, ()
  written = None

  while True:
    sm.update()
    msg = sm["driverStateV2"]
    if not sm.updated["driverStateV2"]:
      continue

    # Re-check the active segment once a second (20 Hz model output).
    if segment is None or msg.frameId % 20 == 1:
      dirs = listdir_by_creation(root)
      if dirs:
        candidate = os.path.join(root, dirs[-1])
        if candidate != segment:
          segment, severity, categories, written = candidate, 0, (), None

    if segment is None:
      continue

    sev, cats = classify_driver_state(msg)
    if sev > 0:
      severity = max(severity, sev)
      categories = tuple(sorted(set(categories) | set(cats)))
      state = (severity, categories)
      if state != written:
        write_tag(os.path.join(segment, EVENT_TAG), format_tag(*state), severity, categories)
        written = state


def main():
  events_thread()


if __name__ == "__main__":
  main()
