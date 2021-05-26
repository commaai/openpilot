#!/usr/bin/env python3
import os
import subprocess
import time
import sys

import numpy as np
from PIL import Image
from typing import List

import cereal.messaging as messaging
from common.params import Params
from common.realtime import DT_MDL
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size, leon_d_frame_size, tici_f_frame_size
from selfdrive.hardware import TICI
from selfdrive.controls.lib.alertmanager import set_offroad_alert
from selfdrive.manager.process_config import managed_processes

LM_THRESH = 120  # defined in selfdrive/camerad/imgproc/utils.h


def jpeg_write(fn, dat):
  img = Image.fromarray(dat)
  img.save(fn, "JPEG")


def extract_image(dat, frame_sizes):
  img = np.frombuffer(dat, dtype=np.uint8)
  w, h = frame_sizes[len(img) // 3]
  b = img[::3].reshape(h, w)
  g = img[1::3].reshape(h, w)
  r = img[2::3].reshape(h, w)
  return np.dstack([r, g, b])


def rois_in_focus(lapres: List[float]) -> float:
  sz = len(lapres)
  return sum([1. / sz for sharpness in
              lapres if sharpness >= LM_THRESH])


def get_snapshots(frame="roadCameraState", front_frame="driverCameraState", focus_perc_threshold=0., no_roi=False):
  frame_sizes = [eon_f_frame_size, eon_d_frame_size, leon_d_frame_size, tici_f_frame_size]
  frame_sizes = {w * h: (w, h) for (w, h) in frame_sizes}

  sockets = []
  if frame is not None:
    sockets.append(frame)
  if front_frame is not None:
    sockets.append(front_frame)

  sm = messaging.SubMaster(sockets)

  if no_roi:
    time.sleep(3.0)
    while min(sm.logMonoTime.values()) == 0:
      sm.update()
  else:
    start_t = time.monotonic()
    while time.monotonic() - start_t < 10:
      sm.update()
      # wait 4 sec for startup and AF
      if sm.rcv_frame[frame] * DT_MDL > 4. and rois_in_focus(sm[frame].sharpnessScore) >= focus_perc_threshold:
        break

  rear = extract_image(sm[frame].image, frame_sizes) if frame is not None else None
  front = extract_image(sm[front_frame].image, frame_sizes) if front_frame is not None else None
  return rear, front


def snapshot(no_roi):
  params = Params()
  front_camera_allowed = params.get_bool("RecordFront")

  if (not params.get_bool("IsOffroad")) or params.get_bool("IsTakingSnapshot"):
    print("Already taking snapshot")
    return None, None

  params.put_bool("IsTakingSnapshot", True)
  set_offroad_alert("Offroad_IsTakingSnapshot", True)
  time.sleep(2.0)  # Give thermald time to read the param, or if just started give camerad time to start

  # Check if camerad is already started
  try:
    subprocess.check_call(["pgrep", "camerad"])
    print("Camerad already running")
    params.put_bool("IsTakingSnapshot", False)
    params.delete("Offroad_IsTakingSnapshot")
    return None, None
  except subprocess.CalledProcessError:
    pass

  os.environ["SEND_ROAD"] = "1"
  os.environ["SEND_WIDE_ROAD"] = "1"

  if front_camera_allowed:
    os.environ["SEND_DRIVER"] = "1"

  managed_processes['camerad'].start()
  frame = "wideRoadCameraState" if TICI else "roadCameraState"
  front_frame = "driverCameraState" if front_camera_allowed else None
  focus_perc_threshold = 0. if TICI else 10 / 12.

  rear, front = get_snapshots(frame, front_frame, focus_perc_threshold, no_roi)
  managed_processes['camerad'].stop()

  params.put_bool("IsTakingSnapshot", False)
  set_offroad_alert("Offroad_IsTakingSnapshot", False)

  if not front_camera_allowed:
    front = None

  return rear, front


if __name__ == "__main__":
  no_roi = sys.argv[1].lower() == 'no_roi'
  if no_roi:
    print('Using previous behavior!')
  pic, fpic = snapshot(no_roi)
  if pic is not None:
    print(pic.shape)
    suffix = 0
    no_roi_text = 'prev' if no_roi else 'new'
    while os.path.exists((f_name := f'/tmp/back_{no_roi_text}.{suffix}.jpg')):
      suffix += 1

    jpeg_write(f_name, pic)
    if fpic is not None:
      jpeg_write("/tmp/front.jpg", fpic)
  else:
    print("Error taking snapshot")
