#!/usr/bin/env python3
import argparse
import os
import sys
import numpy as np
import rerun as rr
import cereal.messaging as messaging
from openpilot.common.basedir import BASEDIR
from openpilot.common.transformations.camera import DEVICE_CAMERAS
from openpilot.tools.replay.lib.ui_helpers import (UP, Calibration,
                                         get_blank_lid_overlay,
                                         maybe_update_radar_points, plot_lead,
                                         plot_model)
from msgq.visionipc import VisionIpcClient, VisionStreamType

os.environ['BASEDIR'] = BASEDIR

ANGLE_SCALE = 5.0
UP.lidar_zoom = 12
colorPallete = [(96, "red", (255, 0, 0)), (100, "100", (255, 36, 0)), (124, "yellow", (255, 255, 0)),
                (230, "230", (255, 36, 170)), (240, "240", (255, 146, 0)), (255, "255", (255, 255, 255))]
def visualize(addr):
  sm = messaging.SubMaster([ 'radarState', 'liveCalibration', 'liveTracks', 'modelV2', 'roadCameraState'], addr=addr)
  img = np.zeros((480, 640, 3), dtype='uint8')
  num_px = 0
  calibration = None
  lid_overlay_blank = get_blank_lid_overlay(UP)

  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while True:
    lid_overlay = lid_overlay_blank.copy()

    # ***** frame *****
    if not vipc_client.is_connected():
      vipc_client.connect(True)

    yuv_img_raw = vipc_client.recv()
    if yuv_img_raw is None or not yuv_img_raw.data.any():
      continue

    sm.update(0)

    camera = DEVICE_CAMERAS[("tici", str(sm['roadCameraState'].sensor))]
    num_px = vipc_client.width * vipc_client.height

    calib_scale = camera.fcam.width / 640.

    intrinsic_matrix = camera.fcam.intrinsics

    if sm.recv_frame['modelV2']:
      plot_model(sm['modelV2'], img, calibration, lid_overlay)

    if sm.recv_frame['radarState']:
      plot_lead(sm['radarState'], lid_overlay)

    liveTracksTime = sm.logMonoTime['liveTracks']
    maybe_update_radar_points(sm['liveTracks'], lid_overlay)
    rr.set_time_nanos("TIMELINE", liveTracksTime)
    rr.log("tracks", rr.AnnotationContext(colorPallete), static=True)
    rr.log("tracks", rr.SegmentationImage(np.flip(np.rot90(lid_overlay, k=-1), axis=1)))

    if sm.updated['liveCalibration'] and num_px:
      rpyCalib = np.asarray(sm['liveCalibration'].rpyCalib)
      calibration = Calibration(num_px, rpyCalib, intrinsic_matrix, calib_scale)


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
    messaging.context = messaging.Context()
  rr.init("rerun_radarPoints")
  rr.spawn()
  visualize(args.ip_address)
