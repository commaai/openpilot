#!/usr/bin/env python3
import os
import signal
import subprocess
import time

import numpy as np
from PIL import Image

import cereal.messaging as messaging
from common.basedir import BASEDIR
from common.params import Params
from common.transformations.camera import eon_f_frame_size, eon_d_frame_size, leon_d_frame_size, tici_f_frame_size
from selfdrive.hardware import TICI
from selfdrive.controls.lib.alertmanager import set_offroad_alert


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


def get_snapshots(frame="frame", front_frame="frontFrame"):
  frame_sizes = [eon_f_frame_size, eon_d_frame_size, leon_d_frame_size, tici_f_frame_size]
  frame_sizes = {w * h: (w, h) for (w, h) in frame_sizes}

  sm = messaging.SubMaster([frame, front_frame])
  while min(sm.logMonoTime.values()) == 0:
    sm.update()

  rear = extract_image(sm[frame].image, frame_sizes)
  front = extract_image(sm[front_frame].image, frame_sizes)
  return rear, front


def snapshot():
  params = Params()
  front_camera_allowed = int(params.get("RecordFront"))

  if params.get("IsOffroad") != b"1" or params.get("IsTakingSnapshot") == b"1":
    print("Already taking snapshot")
    return None, None

  params.put("IsTakingSnapshot", "1")
  set_offroad_alert("Offroad_IsTakingSnapshot", True)
  time.sleep(2.0)  # Give thermald time to read the param, or if just started give camerad time to start

  # Check if camerad is already started
  try:
    subprocess.check_call(["pgrep", "camerad"])

    print("Camerad already running")
    params.put("IsTakingSnapshot", "0")
    params.delete("Offroad_IsTakingSnapshot")
    return None, None
  except subprocess.CalledProcessError:
    pass

  env = os.environ.copy()
  env["SEND_REAR"] = "1"
  env["SEND_WIDE"] = "1"
  env["SEND_FRONT"] = "1"
  proc = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"),
                          cwd=os.path.join(BASEDIR, "selfdrive/camerad"), env=env)
  time.sleep(3.0)

  frame = "wideFrame" if TICI else "frame"
  rear, front = get_snapshots(frame)

  proc.send_signal(signal.SIGINT)
  proc.communicate()

  params.put("IsTakingSnapshot", "0")
  set_offroad_alert("Offroad_IsTakingSnapshot", False)

  if not front_camera_allowed:
    front = None

  return rear, front


if __name__ == "__main__":
  pic, fpic = snapshot()
  if pic is not None:
    print(pic.shape)
    jpeg_write("/tmp/back.jpg", pic)
    jpeg_write("/tmp/front.jpg", fpic)
  else:
    print("Error taking snapshot")
