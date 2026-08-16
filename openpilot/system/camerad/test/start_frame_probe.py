#!/usr/bin/env python3

import os
import shutil
import time
from pathlib import Path

LOG_ROOT = "/tmp/camera_start_probe"
os.environ["LOG_ROOT"] = LOG_ROOT
os.environ["LOGGERD_TEST"] = "1"

import openpilot.cereal.messaging as messaging
from openpilot.system.manager.process_config import managed_processes
from openpilot.tools.lib.logreader import LogReader


CAMERAS = ("narrowRoadCameraState", "wideRoadCameraState", "cabinCameraState")


def first_frame(msg):
  return getattr(msg, msg.which()).frameId


for run in range(10):
  shutil.rmtree(LOG_ROOT, ignore_errors=True)
  sockets = {camera: messaging.sub_sock(camera, conflate=False, timeout=10000) for camera in CAMERAS}
  direct = {}

  try:
    managed_processes["loggerd"].start()
    managed_processes["camerad"].start()
    direct = {camera: first_frame(messaging.recv_one_retry(sock)) for camera, sock in sockets.items()}
    time.sleep(0.5)
  finally:
    managed_processes["camerad"].stop()
    managed_processes["loggerd"].stop()

  rlog = max(Path(LOG_ROOT).glob("*--0/rlog.zst"), key=lambda path: path.stat().st_mtime)
  logged = {}
  for msg in LogReader(str(rlog)):
    service = msg.which()
    if service in CAMERAS and service not in logged:
      logged[service] = first_frame(msg)

  print(f"run={run + 1} direct={direct} logged={logged}", flush=True)

