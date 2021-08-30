#!/usr/bin/env python3
import subprocess
import time

import numpy as np
from PIL import Image
from typing import List

import cereal.messaging as messaging
from cereal.visionipc.visionipc_pyx import VisionIpcClient, VisionStreamType  # pylint: disable=no-name-in-module, import-error
from common.params import Params
from common.realtime import DT_MDL
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size, tici_f_frame_size
from selfdrive.hardware import TICI
from selfdrive.controls.lib.alertmanager import set_offroad_alert
from selfdrive.manager.process_config import managed_processes

LM_THRESH = 120  # defined in selfdrive/camerad/imgproc/utils.h

VISION_STREAMS = {
  "roadCameraState": VisionStreamType.VISION_STREAM_RGB_BACK,
  "driverCameraState": VisionStreamType.VISION_STREAM_RGB_FRONT,
  "wideRoadCameraState": VisionStreamType.VISION_STREAM_RGB_WIDE,
}

def jpeg_write(fn, dat):
  img = Image.fromarray(dat)
  img.save(fn, "JPEG")


def extract_image(img, frame_sizes):
  w, h = frame_sizes[len(img) // 3]
  b = img[::3].reshape(h, w)
  g = img[1::3].reshape(h, w)
  r = img[2::3].reshape(h, w)
  return np.dstack([r, g, b])


def rois_in_focus(lapres: List[float]) -> float:
  return sum([1. / len(lapres) for sharpness in lapres if sharpness >= LM_THRESH])


def get_snapshots(frame="roadCameraState", front_frame="driverCameraState", focus_perc_threshold=0.):
  frame_sizes = [eon_f_frame_size, eon_d_frame_size, tici_f_frame_size]
  frame_sizes = {w * h: (w, h) for (w, h) in frame_sizes}

  sockets = [s for s in (frame, front_frame) if s is not None]
  sm = messaging.SubMaster(sockets)
  vipc_clients = {s: VisionIpcClient("camerad", VISION_STREAMS[s], True) for s in sockets}

  # wait 4 sec from camerad startup for focus and exposure
  while sm[sockets[0]].frameId < int(4. / DT_MDL):
    sm.update()

  for client in vipc_clients.values():
    client.connect(False)

  # wait for focus
  start_t = time.monotonic()
  while time.monotonic() - start_t < 10:
    sm.update(100)
    if min(sm.rcv_frame.values()) > 1 and rois_in_focus(sm[frame].sharpnessScore) >= focus_perc_threshold:
      break

  # grab images
  rear, front = None, None
  if frame is not None:
    rear = extract_image(vipc_clients[frame].recv(), frame_sizes)

  #rear = extract_image(sm[frame].image, frame_sizes) if frame is not None else None
  #front = extract_image(sm[front_frame].image, frame_sizes) if front_frame is not None else None
  return rear, front


def snapshot():
  params = Params()

  if (not params.get_bool("IsOffroad")) or params.get_bool("IsTakingSnapshot"):
    print("Already taking snapshot")
    return None, None

  front_camera_allowed = params.get_bool("RecordFront")
  set_offroad_alert("Offroad_IsTakingSnapshot", True)
  params.put_bool("IsTakingSnapshot", True)
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

  try:
    managed_processes['camerad'].start()
    frame = "wideRoadCameraState" if TICI else "roadCameraState"
    front_frame = "driverCameraState" if front_camera_allowed else None
    focus_perc_threshold = 0. if TICI else 10 / 12.

    rear, front = get_snapshots(frame, front_frame, focus_perc_threshold)
  finally:
    managed_processes['camerad'].stop()
    params.put_bool("IsTakingSnapshot", False)
    set_offroad_alert("Offroad_IsTakingSnapshot", False)
  managed_processes['camerad'].start()
  frame = "wideRoadCameraState" if TICI else "roadCameraState"
  front_frame = "driverCameraState" if front_camera_allowed else None
  focus_perc_threshold = 0. if TICI else 10 / 12.

  rear, front = get_snapshots(frame, front_frame, focus_perc_threshold)
  managed_processes['camerad'].stop()

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
