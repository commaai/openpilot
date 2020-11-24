#!/usr/bin/env python3
import os
import signal
import subprocess
import time
from PIL import Image
from common.basedir import BASEDIR
from common.params import Params
from selfdrive.camerad.snapshot.visionipc import VisionIPC
from selfdrive.controls.lib.alertmanager import set_offroad_alert


def jpeg_write(fn, dat):
  img = Image.fromarray(dat)
  img.save(fn, "JPEG")


def snapshot():
  params = Params()
  front_camera_allowed = int(params.get("RecordFront"))

  if params.get("IsOffroad") != b"1" or params.get("IsTakingSnapshot") == b"1":
    return None

  params.put("IsTakingSnapshot", "1")
  set_offroad_alert("Offroad_IsTakingSnapshot", True)
  time.sleep(2.0)  # Give thermald time to read the param, or if just started give camerad time to start

  # Check if camerad is already started
  ps = subprocess.Popen("ps | grep camerad", shell=True, stdout=subprocess.PIPE)
  ret = list(filter(lambda x: 'grep ' not in x, ps.communicate()[0].decode('utf-8').strip().split("\n")))
  if len(ret) > 0:
    params.put("IsTakingSnapshot", "0")
    params.delete("Offroad_IsTakingSnapshot")
    return None

  proc = subprocess.Popen(os.path.join(BASEDIR, "selfdrive/camerad/camerad"), cwd=os.path.join(BASEDIR, "selfdrive/camerad"))
  time.sleep(3.0)

  ret = None
  start_time = time.time()
  while time.time() - start_time < 5.0:
    try:
      ipc = VisionIPC()
      pic = ipc.get()
      del ipc

      if front_camera_allowed:
        ipc_front = VisionIPC(front=True)
        fpic = ipc_front.get()
        del ipc_front
      else:
        fpic = None

      ret = pic, fpic
      break
    except Exception:
      time.sleep(1)

  proc.send_signal(signal.SIGINT)
  proc.communicate()

  params.put("IsTakingSnapshot", "0")
  set_offroad_alert("Offroad_IsTakingSnapshot", False)
  return ret


if __name__ == "__main__":
  pic, fpic = snapshot()
  print(pic.shape)
  jpeg_write("/tmp/back.jpg", pic)
  jpeg_write("/tmp/front.jpg", fpic)
