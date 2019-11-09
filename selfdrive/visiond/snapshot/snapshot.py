#!/usr/bin/env python3
import os
import subprocess
from PIL import Image
from common.params import Params
import time
import signal

from selfdrive.visiond.snapshot.visionipc import VisionIPC

def jpeg_write(fn, dat):
  img = Image.fromarray(dat)
  img.save(fn, "JPEG")

def snapshot():
  # note: super sketch race condition if you start the car at the same time at this
  # TODO: lock car starting?
  ps = subprocess.Popen("ps | grep visiond", shell=True, stdout=subprocess.PIPE)
  ret = list(filter(lambda x: 'grep ' not in x, ps.communicate()[0].decode('utf-8').strip().split("\n")))
  if len(ret) > 0:
    return None

  front_camera_allowed = int(Params().get("RecordFront"))

  proc = subprocess.Popen(os.path.join(os.getenv("HOME"), "one/selfdrive/visiond/start.py"), cwd=os.path.join(os.getenv("HOME"), "one/selfdrive/visiond"))
  time.sleep(6.0)

  ret = None
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
  finally:
    proc.send_signal(signal.SIGINT)
    proc.communicate()

  return ret

if __name__ == "__main__":
  pic, fpic = snapshot()
  jpeg_write("/tmp/back.jpg", pic)
  jpeg_write("/tmp/front.jpg", fpic)

