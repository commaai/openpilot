#!/usr/bin/env python3
import os
import time
import subprocess

import cereal.messaging as messaging
from msgq.visionipc import VisionIpcClient, VisionStreamType, VisionBuf
from openpilot.common.basedir import BASEDIR
from openpilot.common.params import Params

import cv2
import numpy as np

run_cnt = 0
frame_cnt = 0
def is_tearing(img):
  global run_cnt
  global frame_cnt
  frame_cnt += 1
  image = np.array(img.data[:img.uv_offset], dtype=np.uint8).reshape((-1, img.stride))[:img.height, :img.width]

  sobel_x = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)  # Horizontal edges
  sobel_y = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)  # Vertical edges

  edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
  edge_magnitude = cv2.normalize(edge_magnitude, None, 0, 255, cv2.NORM_MINMAX)

  _, binary_mask = cv2.threshold(edge_magnitude.astype(np.uint8), 30, 255, cv2.THRESH_BINARY)
  row_sums = np.sum(binary_mask, axis=1)

  tearing_regions = row_sums > np.mean(row_sums) + 1.5 * np.std(row_sums)

  tearing = False
  output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
  for i, is_tearing in enumerate(tearing_regions):
    if is_tearing and 630 < i < 675:
      cv2.line(output_image, (0, i), (output_image.shape[1], i), (0, 0, 255), 1)
      tearing = True

  #cv2.imwrite(f"/data/tmp/frame_{run_cnt:03}_{frame_cnt:03}_{tearing}.png", image)
  #cv2.imwrite(f"/data/tmp/frame_{run_cnt:03}_{frame_cnt:03}_debug.png", output_image)
  return tearing

if __name__ == "__main__":
  if False:
    v = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    assert v.connect(False)
    while True:
      img = v.recv()
      assert img is not None
      print("is_tearing", is_tearing(img))
    exit()

  tearing_run_cnt = 0
  manager_path = BASEDIR + "/system/manager/manager.py"
  params = Params()
  for _ in range(30):
    run_cnt += 1
    print(f"====== {run_cnt} ======")

    v = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
    try:
      proc = subprocess.Popen(["python", manager_path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
      while not v.connect(False):
        time.sleep(0.1)
      time.sleep(2)

      tearing_cnt = 0
      for _ in range(30):
        while (img := v.recv()) is None:
          time.sleep(0.1)
        tearing_cnt += is_tearing(img)
      tore = tearing_cnt >= 1
      tearing_run_cnt += tore
      print(" - tore ", tore, tearing_cnt)
      print(" - route", params.get("CurrentRoute", encoding="utf8"))
      print(f" - {tearing_run_cnt:03} / {run_cnt:03} ({tearing_run_cnt/run_cnt:.2%})")


    finally:
      proc.terminate()
      if proc.wait(60) is None:
        proc.kill()
    #break
