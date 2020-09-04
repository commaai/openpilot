#!/usr/bin/env python3
'''
This process finds calibration values. More info on what these calibration values
are can be found here https://github.com/commaai/openpilot/tree/master/common/transformations
While the roll calibration is a real value that can be estimated, here we assume it zero,
and the image input into the neural network is not corrected for roll.
'''

import os
import copy
import json
import numpy as np
import cereal.messaging as messaging
from selfdrive.config import Conversions as CV
from selfdrive.locationd.calibration_helpers import Calibration
from selfdrive.swaglog import cloudlog
from common.params import Params, put_nonblocking
from common.transformations.model import model_height
from common.transformations.camera import get_view_frame_from_road_frame
from common.transformations.orientation import rot_from_euler, euler_from_rot

MIN_SPEED_FILTER = 15 * CV.MPH_TO_MS
MAX_VEL_ANGLE_STD = np.radians(0.25)
MAX_YAW_RATE_FILTER = np.radians(2)  # per second

# This is all 20Hz, blocks needed for efficiency
BLOCK_SIZE = 100
INPUTS_NEEDED = 5   # Minimum blocks needed for valid calibration
INPUTS_WANTED = 50   # We want a little bit more than we need for stability
RPY_INIT = np.array([0,0,0])

# These values are needed to accomodate biggest modelframe
PITCH_LIMITS = np.array([-0.09074112085129739, 0.14907572052989657])
YAW_LIMITS = np.array([-0.06912048084718224, 0.06912048084718235])
DEBUG = os.getenv("DEBUG") is not None


def is_calibration_valid(rpy):
  return (PITCH_LIMITS[0] < rpy[1] < PITCH_LIMITS[1]) and (YAW_LIMITS[0] < rpy[2] < YAW_LIMITS[1])


def sanity_clip(rpy):
  if np.isnan(rpy).any():
    rpy = RPY_INIT
  return np.array([rpy[0],
                   np.clip(rpy[1], PITCH_LIMITS[0] - .005, PITCH_LIMITS[1] + .005),
                   np.clip(rpy[2], YAW_LIMITS[0] - .005, YAW_LIMITS[1] + .005)])



class Calibrator():
  def __init__(self, param_put=False):
    self.param_put = param_put
    self.rpy = copy.copy(RPY_INIT)
    self.rpys = np.zeros((INPUTS_WANTED, 3))
    self.idx = 0
    self.block_idx = 0
    self.valid_blocks = 0
    self.cal_status = Calibration.UNCALIBRATED
    self.just_calibrated = False
    self.v_ego = 0

    # Read calibration
    if param_put:
      calibration_params = Params().get("CalibrationParams")
    else:
      calibration_params = None
    if calibration_params:
      try:
        calibration_params = json.loads(calibration_params)
        self.rpy = calibration_params["calib_radians"]
        if not np.isfinite(self.rpy).all():
          self.rpy = copy.copy(RPY_INIT)
        self.rpys = np.tile(self.rpy, (INPUTS_WANTED, 1))
        self.valid_blocks = calibration_params['valid_blocks']
        if not np.isfinite(self.valid_blocks) or self.valid_blocks < 0:
          self.valid_blocks = 0
        self.update_status()
      except Exception:
        cloudlog.exception("CalibrationParams file found but error encountered")

  def update_status(self):
    start_status = self.cal_status
    if self.valid_blocks < INPUTS_NEEDED:
      self.cal_status = Calibration.UNCALIBRATED
    else:
      self.cal_status = Calibration.CALIBRATED if is_calibration_valid(self.rpy) else Calibration.INVALID
    end_status = self.cal_status

    self.just_calibrated = False
    if start_status == Calibration.UNCALIBRATED and end_status != Calibration.UNCALIBRATED:
      self.just_calibrated = True

  def handle_v_ego(self, v_ego):
    self.v_ego = v_ego

  def handle_cam_odom(self, trans, rot, trans_std, rot_std):
    straight_and_fast = ((self.v_ego > MIN_SPEED_FILTER) and (trans[0] > MIN_SPEED_FILTER) and (abs(rot[2]) < MAX_YAW_RATE_FILTER))
    certain_if_calib = ((np.arctan2(trans_std[1], trans[0]) < MAX_VEL_ANGLE_STD) or
                        (self.valid_blocks < INPUTS_NEEDED))
    if straight_and_fast and certain_if_calib:
      observed_rpy = np.array([0,
                               -np.arctan2(trans[2], trans[0]),
                               np.arctan2(trans[1], trans[0])])
      new_rpy = euler_from_rot(rot_from_euler(self.rpy).dot(rot_from_euler(observed_rpy)))
      new_rpy = sanity_clip(new_rpy)

      self.rpys[self.block_idx] = (self.idx*self.rpys[self.block_idx] + (BLOCK_SIZE - self.idx) * new_rpy) / float(BLOCK_SIZE)
      self.idx = (self.idx + 1) % BLOCK_SIZE
      if self.idx == 0:
        self.block_idx += 1
        self.valid_blocks = max(self.block_idx, self.valid_blocks)
        self.block_idx = self.block_idx % INPUTS_WANTED
      if self.valid_blocks > 0:
        self.rpy = np.mean(self.rpys[:self.valid_blocks], axis=0)
      self.update_status()

      if self.param_put and ((self.idx == 0 and self.block_idx == 0) or self.just_calibrated):
        cal_params = {"calib_radians": list(self.rpy),
                      "valid_blocks": self.valid_blocks}
        put_nonblocking("CalibrationParams", json.dumps(cal_params).encode('utf8'))
      return new_rpy
    else:
      return None

  def send_data(self, pm):
    if self.valid_blocks > 0:
      max_rpy_calib = np.array(np.max(self.rpys[:self.valid_blocks], axis=0))
      min_rpy_calib = np.array(np.min(self.rpys[:self.valid_blocks], axis=0))
      calib_spread = np.abs(max_rpy_calib - min_rpy_calib)
    else:
      calib_spread = np.zeros(3)
    extrinsic_matrix = get_view_frame_from_road_frame(0, self.rpy[1], self.rpy[2], model_height)

    cal_send = messaging.new_message('liveCalibration')
    cal_send.liveCalibration.validBlocks = self.valid_blocks
    cal_send.liveCalibration.calStatus = self.cal_status
    cal_send.liveCalibration.calPerc = min(100 * (self.valid_blocks * BLOCK_SIZE + self.idx) // (INPUTS_NEEDED * BLOCK_SIZE), 100)
    cal_send.liveCalibration.extrinsicMatrix = [float(x) for x in extrinsic_matrix.flatten()]
    cal_send.liveCalibration.rpyCalib = [float(x) for x in self.rpy]
    cal_send.liveCalibration.rpyCalibSpread = [float(x) for x in calib_spread]

    pm.send('liveCalibration', cal_send)


def calibrationd_thread(sm=None, pm=None):
  if sm is None:
    sm = messaging.SubMaster(['cameraOdometry', 'carState'])

  if pm is None:
    pm = messaging.PubMaster(['liveCalibration'])

  calibrator = Calibrator(param_put=True)

  send_counter = 0
  while 1:
    sm.update()

    # if no inputs still publish calibration
    if not sm.updated['carState'] and not sm.updated['cameraOdometry']:
      calibrator.send_data(pm)
      continue

    if sm.updated['carState']:
      calibrator.handle_v_ego(sm['carState'].vEgo)
      if send_counter % 25 == 0:
        calibrator.send_data(pm)
      send_counter += 1

    if sm.updated['cameraOdometry']:
      new_rpy = calibrator.handle_cam_odom(sm['cameraOdometry'].trans,
                                          sm['cameraOdometry'].rot,
                                          sm['cameraOdometry'].transStd,
                                          sm['cameraOdometry'].rotStd)

      if DEBUG and new_rpy is not None:
        print('got new rpy', new_rpy)


def main(sm=None, pm=None):
  calibrationd_thread(sm, pm)


if __name__ == "__main__":
  main()
