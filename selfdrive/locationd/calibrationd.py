#!/usr/bin/env python3

import os
import copy
import json
import numpy as np
import cereal.messaging as messaging
from selfdrive.locationd.calibration_helpers import Calibration
from selfdrive.swaglog import cloudlog
from common.params import Params, put_nonblocking
from common.transformations.model import model_height
from common.transformations.camera import view_frame_from_device_frame, get_view_frame_from_road_frame, \
                                          eon_intrinsics, get_calib_from_vp, H, W

MPH_TO_MS = 0.44704
MIN_SPEED_FILTER = 15 * MPH_TO_MS
MAX_YAW_RATE_FILTER = np.radians(2)  # per second
INPUTS_NEEDED = 300   # allow to update VP every so many frames
INPUTS_WANTED = 600   # We want a little bit more than we need for stability
WRITE_CYCLES = 400  # write every 400 cycles
VP_INIT = np.array([W/2., H/2.])

# These validity corners were chosen by looking at 1000
# and taking most extreme cases with some margin.
VP_VALIDITY_CORNERS = np.array([[W//2 - 150, 280], [W//2 + 150, 540]])
DEBUG = os.getenv("DEBUG") is not None


def is_calibration_valid(vp):
  return vp[0] > VP_VALIDITY_CORNERS[0,0] and vp[0] < VP_VALIDITY_CORNERS[1,0] and \
         vp[1] > VP_VALIDITY_CORNERS[0,1] and vp[1] < VP_VALIDITY_CORNERS[1,1]


class Calibrator():
  def __init__(self, param_put=False):
    self.param_put = param_put
    self.vp = copy.copy(VP_INIT)
    self.vps = []
    self.cal_status = Calibration.UNCALIBRATED
    self.write_counter = 0
    self.just_calibrated = False

    # Read calibration
    calibration_params = Params().get("CalibrationParams")
    if calibration_params:
      try:
        calibration_params = json.loads(calibration_params)
        self.vp = np.array(calibration_params["vanishing_point"])
        self.vps = np.tile(self.vp, (calibration_params['valid_points'], 1)).tolist()
        self.update_status()
      except Exception:
        cloudlog.exception("CalibrationParams file found but error encountered")

  def update_status(self):
    start_status = self.cal_status
    if len(self.vps) < INPUTS_NEEDED:
      self.cal_status = Calibration.UNCALIBRATED
    else:
      self.cal_status = Calibration.CALIBRATED if is_calibration_valid(self.vp) else Calibration.INVALID
    end_status = self.cal_status

    self.just_calibrated = False
    if start_status == Calibration.UNCALIBRATED and end_status == Calibration.CALIBRATED:
      self.just_calibrated = True

  def handle_cam_odom(self, log):
    trans, rot = log.trans, log.rot
    if np.linalg.norm(trans) > MIN_SPEED_FILTER and abs(rot[2]) < MAX_YAW_RATE_FILTER:
      new_vp = eon_intrinsics.dot(view_frame_from_device_frame.dot(trans))
      new_vp = new_vp[:2]/new_vp[2]
      self.vps.append(new_vp)
      self.vps = self.vps[-INPUTS_WANTED:]
      self.vp = np.mean(self.vps, axis=0)
      self.update_status()
      self.write_counter += 1
      if self.param_put and (self.write_counter % WRITE_CYCLES == 0 or self.just_calibrated):
        cal_params = {"vanishing_point": list(self.vp),
                      "valid_points": len(self.vps)}
        put_nonblocking("CalibrationParams", json.dumps(cal_params).encode('utf8'))
      return new_vp
    else:
      return None

  def send_data(self, pm):
    calib = get_calib_from_vp(self.vp)
    extrinsic_matrix = get_view_frame_from_road_frame(0, calib[1], calib[2], model_height)

    cal_send = messaging.new_message()
    cal_send.init('liveCalibration')
    cal_send.liveCalibration.calStatus = self.cal_status
    cal_send.liveCalibration.calPerc = min(len(self.vps) * 100 // INPUTS_NEEDED, 100)
    cal_send.liveCalibration.extrinsicMatrix = [float(x) for x in extrinsic_matrix.flatten()]
    cal_send.liveCalibration.rpyCalib = [float(x) for x in calib]

    pm.send('liveCalibration', cal_send)


def calibrationd_thread(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['cameraOdometry'])

  if pm is None:
    pm = messaging.PubMaster(['liveCalibration'])

  calibrator = Calibrator(param_put=True)

  # buffer with all the messages that still need to be input into the kalman
  while 1:
    sm.update()

    new_vp = calibrator.handle_cam_odom(sm['cameraOdometry'])
    if DEBUG and new_vp is not None:
      print('got new vp', new_vp)

    calibrator.send_data(pm)


def main(sm=None, pm=None):
  calibrationd_thread(sm, pm)


if __name__ == "__main__":
  main()
