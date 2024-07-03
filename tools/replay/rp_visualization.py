#!/usr/bin/env python3
import argparse
import os
import sys

import pygame
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

def ui_thread(addr):

  sm = messaging.SubMaster(['carState', 'longitudinalPlan', 'carControl', 'radarState', 'liveCalibration', 'controlsState',
                            'liveTracks', 'modelV2', 'liveParameters', 'roadCameraState'], addr=addr)

  img = np.zeros((480, 640, 3), dtype='uint8')
  num_px = 0
  calibration = None
  lid_overlay_blank = get_blank_lid_overlay(UP)

  vipc_client = VisionIpcClient("camerad", VisionStreamType.VISION_STREAM_ROAD, True)
  while True:
    # top_down_surface = np.zeros((UP.lidar_x, UP.lidar_y), dtype=np.uint8)
    top_down_surface = pygame.surface.Surface((UP.lidar_x, UP.lidar_y), 0, 8)

    lid_overlay = lid_overlay_blank.copy()
    top_down = top_down_surface, lid_overlay

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

    modelV2Time = sm.logMonoTime['modelV2']
    if sm.recv_frame['modelV2']:
      plot_model(sm['modelV2'], img, calibration, top_down)
      if not np.all(top_down[1] == 0):
        rr.set_time_nanos("TIMELINE", modelV2Time)
        rr.log("tracks", rr.Image(np.flip(np.rot90(top_down[1], k=-1), axis=1)))

    radarStateTime = sm.logMonoTime['radarState']
    if sm.recv_frame['radarState']:
      plot_lead(sm['radarState'], top_down)
      if not np.all(top_down[1] == 0):
        rr.set_time_nanos("TIMELINE", radarStateTime)
        rr.log("tracks", rr.Image(np.flip(np.rot90(top_down[1], k=-1), axis=1)))


    # draw all radar points
    liveTracksTime = sm.logMonoTime['liveTracks']
    maybe_update_radar_points(sm['liveTracks'], top_down[1])
    if not np.all(top_down[1] == 0):
      rr.set_time_nanos("TIMELINE", liveTracksTime)
      rr.log("tracks", rr.Image(np.flip(np.rot90(top_down[1], k=-1), axis=1)))


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
  rr.init("rerun_test")
  rr.spawn()
  ui_thread(args.ip_address)
