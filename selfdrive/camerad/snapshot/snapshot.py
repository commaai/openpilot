#!/usr/bin/env python3
import os
import subprocess
import time

import numpy as np
from PIL import Image
from typing import List

import cereal.messaging as messaging
from common.params import Params
from common.realtime import DT_MDL
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size, tici_f_frame_size
from selfdrive.hardware import HARDWARE, TICI
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


def get_snapshots(frame="roadCameraState", front_frame="driverCameraState", focus_perc_threshold=0.):
  frame_sizes = [eon_f_frame_size, eon_d_frame_size, tici_f_frame_size]
  frame_sizes = {w * h: (w, h) for (w, h) in frame_sizes}

  sockets = []
  if frame is not None:
    sockets.append(frame)
  if front_frame is not None:
    sockets.append(front_frame)

  # wait 4 sec from camerad startup for focus and exposure
  sm = messaging.SubMaster(sockets)
  while sm[sockets[0]].frameId < int(4. / DT_MDL):
    sm.update()

  start_t = time.monotonic()
  while time.monotonic() - start_t < 10:
    sm.update()
    if min(sm.rcv_frame.values()) > 1 and rois_in_focus(sm[frame].sharpnessScore) >= focus_perc_threshold:
      break

  rear = extract_image(sm[frame].image, frame_sizes) if frame is not None else None
  front = extract_image(sm[front_frame].image, frame_sizes) if front_frame is not None else None
  return rear, front


def snapshot():
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

  try:
    HARDWARE.set_power_save(False)
    managed_processes['camerad'].start()
    frame = "wideRoadCameraState" if TICI else "roadCameraState"
    front_frame = "driverCameraState" if front_camera_allowed else None
    focus_perc_threshold = 0. if TICI else 10 / 12.

    rear, front = get_snapshots(frame, front_frame, focus_perc_threshold)
  finally:
    managed_processes['camerad'].stop()
    HARDWARE.set_power_save(True)

    params.put_bool("IsTakingSnapshot", False)
    set_offroad_alert("Offroad_IsTakingSnapshot", False)

  if not front_camera_allowed:
    front = None

  return rear, front


if __name__ == "__main__":
  pic, fpic = snapshot()
  if pic is not None:
    print(pic.shape)
    jpeg_write("/tmp/back.jpg", pic)
    if fpic is not None:
      jpeg_write("/tmp/front.jpg", fpic)
  else:
    print("Error taking snapshot")
