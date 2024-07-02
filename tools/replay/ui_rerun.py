#!/usr/bin/env python3
import argparse
import os
import sys
import cereal.messaging as messaging
import numpy as np
import rerun as rr
from msgq.visionipc import VisionIpcClient, VisionStreamType
from openpilot.tools.replay.lib.ui_helpers import to_topdown_pt, get_blank_lid_overlay, UP


img = np.zeros((480, 640, 3), dtype='uint8')
ANGLE_SCALE = 5.0

def maybe_update_radar_rerun(lt, lid_overlay):
  ar_pts = []
  if lt is not None:
    ar_pts = {}
    for track in lt:
      ar_pts[track.trackId] = [track.dRel, track.yRel, track.vRel, track.aRel, track.oncoming, track.stationary]
  for ids, pt in ar_pts.items():
    # negative here since radar is left positive
    px, py = to_topdown_pt(pt[0], -pt[1])
    if px != -1:
      if pt[-1]:
        color = 240
      elif pt[-2]:
        color = 230
      else:
        color = 255
      if int(ids) == 1:
        lid_overlay[px - 2:px + 2, py - 10:py + 10] = 100
      else:
        lid_overlay[px - 2:px + 2, py - 2:py + 2] = color
  return lid_overlay


def getMsgs(addr):
  prevliveTracksTime = -1

  sm = messaging.SubMaster(['carState', 'longitudinalPlan', 'carControl', 'radarState', 'liveCalibration', 'controlsState',
                          'liveTracks', 'modelV2', 'liveParameters', 'roadCameraState'], addr=addr)
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)

  while True:
    # ***** frame *****
    if not vipc_client.is_connected():
      vipc_client.connect(True)

    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue

    sm.update(0)
    liveTracksTime = sm.logMonoTime['liveTracks']
    if liveTracksTime != prevliveTracksTime:
      rr.set_time_nanos("TIMELINE", liveTracksTime)
      prevliveTracksTime = liveTracksTime
      lid_overlay = get_blank_lid_overlay(UP).copy()
      maybe_update_radar_rerun(sm["liveTracks"], lid_overlay)
      rr.log("tracks", rr.Image(lid_overlay))


def get_arg_parser():
  parser = argparse.ArgumentParser(
    description="Show replay data in a UI.",
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

  parser.add_argument("ip_address", nargs="?", default="127.0.0.1",
                      help="The ip address on which to receive zmq messages.")

  parser.add_argument("--frame-address", default=None,
                      help="The frame address (fully qualified ZMQ endpoint for frames) on which to receive zmq messages.")
  return parser


if __name__ == "__main__":
  args = get_arg_parser().parse_args(sys.argv[1:])
  rr.init("rerun_test")
  rr.spawn()
  if args.ip_address != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.context = messaging.Context()
  getMsgs(args.ip_address)
