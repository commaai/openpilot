#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import rerun as rr
import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.tools.replay.lib.rp_helpers import (UP, rerunColorPalette,
                                         get_blank_lid_overlay,
                                         update_radar_points, plot_lead,
                                         plot_model)
from msgq.visionipc import VisionIpcClient, VisionStreamType

os.environ['BASEDIR'] = BASEDIR

UP.lidar_zoom = 6

def visualize(addr):
  sm = messaging.SubMaster(['radarState', 'liveTracks', 'modelV2'], addr=addr)
  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while True:
    if not vipc_client.is_connected():
      vipc_client.connect(True)
    new_data = vipc_client.recv()
    if new_data is None or not new_data.data.any():
      continue

    sm.update(0)
    lid_overlay = get_blank_lid_overlay(UP)
    if sm.recv_frame['modelV2']:
      plot_model(sm['modelV2'], lid_overlay)
    if sm.recv_frame['radarState']:
      plot_lead(sm['radarState'], lid_overlay)
    liveTracksTime = sm.logMonoTime['liveTracks']
    if sm.updated['liveTracks']:
      update_radar_points(sm['liveTracks'], lid_overlay)
    rr.set_time_nanos("TIMELINE", liveTracksTime)
    rr.log("tracks", rr.SegmentationImage(np.flip(np.rot90(lid_overlay, k=-1), axis=1)))


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
  if args.ip_address != "127.0.0.1":
    os.environ["ZMQ"] = "1"
    messaging.reset_context()
  rr.init("RadarPoints", spawn= True)
  rr.log("tracks", rr.AnnotationContext(rerunColorPalette), static=True)
  visualize(args.ip_address)
