#!/usr/bin/env python
import os
import zmq
import cv2
import copy
import json
import numpy as np
import selfdrive.messaging as messaging
from selfdrive.locationd.calibration_values import Calibration, Filter
from selfdrive.swaglog import cloudlog
from selfdrive.services import service_list
from common.params import Params
from common.ffi_wrapper import ffi_wrap
import common.transformations.orientation as orient
from common.transformations.model import model_height, get_camera_frame_from_model_frame
from common.transformations.camera import view_frame_from_device_frame, get_view_frame_from_road_frame, \
                                          eon_intrinsics, get_calib_from_vp, normalize, denormalize, H, W


FRAMES_NEEDED = 120  # allow to update VP every so many frames
VP_CYCLES_NEEDED = 2
CALIBRATION_CYCLES_NEEDED = FRAMES_NEEDED * VP_CYCLES_NEEDED
DT = 0.1      # nominal time step of 10Hz (orbd_live runs slower on pc)
VP_RATE_LIM = 2. * DT    # 2 px/s
VP_INIT = np.array([W/2., H/2.])
EXTERNAL_PATH = os.path.dirname(os.path.abspath(__file__))
# big model is 864x288
VP_VALIDITY_CORNERS = np.array([[-150., -200.], [150., 200.]])  + VP_INIT
GRID_WEIGHT_INIT = 2e6
MAX_LINES = 500    # max lines to avoid over computation
HOOD_HEIGHT = H*3/4 # the part of image usually free from the car's hood

DEBUG = os.getenv("DEBUG") is not None

# Wrap c code for slow grid incrementation
c_header = "\nvoid increment_grid(double *grid, double *lines, long long n);"
c_code = "#define H %d\n" % H
c_code += "#define W %d\n" % W
c_code += "\n" + open(os.path.join(EXTERNAL_PATH, "get_vp.c")).read()
ffi, lib = ffi_wrap('get_vp', c_code, c_header)


def increment_grid_c(grid, lines, n):
  lib.increment_grid(ffi.cast("double *", grid.ctypes.data),
                ffi.cast("double *", lines.ctypes.data),
                ffi.cast("long long", n))

def get_lines(p):
  A = (p[:,0,1] - p[:,1,1])
  B = (p[:,1,0] - p[:,0,0])
  C = (p[:,0,0]*p[:,1,1] - p[:,1,0]*p[:,0,1])
  return np.column_stack((A, B, -C))

def correct_pts(pts, rot_speeds, dt):
  pts = np.hstack((pts, np.ones((pts.shape[0],1))))
  cam_rot = dt * view_frame_from_device_frame.dot(rot_speeds)
  rot = orient.rot_matrix(*cam_rot.T).T
  pts_corrected = rot.dot(pts.T).T
  pts_corrected[:,0] /= pts_corrected[:,2]
  pts_corrected[:,1] /= pts_corrected[:,2]
  return pts_corrected[:,:2]

def gaussian_kernel(sizex, sizey, stdx, stdy, dx, dy):
  y, x = np.mgrid[-sizey:sizey+1, -sizex:sizex+1]
  g = np.exp(-((x - dx)**2 / (2. * stdx**2) + (y - dy)**2 / (2. * stdy**2)))
  return g / g.sum()

def gaussian_kernel_1d(kernel):
  #creates separable gaussian filter
  u,s,v = np.linalg.svd(kernel)
  x = u[:,0]*np.sqrt(s[0])
  y = np.sqrt(s[0])*v[0,:]
  return x, y

def blur_image(img, kernel_x, kernel_y):
  return cv2.sepFilter2D(img.astype(np.uint16), -1, kernel_x, kernel_y)

def is_calibration_valid(vp):
  return vp[0] > VP_VALIDITY_CORNERS[0,0] and vp[0] < VP_VALIDITY_CORNERS[1,0] and \
         vp[1] > VP_VALIDITY_CORNERS[0,1] and vp[1] < VP_VALIDITY_CORNERS[1,1]

class Calibrator(object):
  def __init__(self, param_put=False):
    self.param_put = param_put
    self.speed = 0
    self.vp_cycles = 0
    self.angle_offset = 0.
    self.yaw_rate = 0.
    self.l100_last_updated = 0
    self.prev_orbs = None
    self.kernel = gaussian_kernel(11, 11, 2.35, 2.35, 0, 0)
    self.kernel_x, self.kernel_y = gaussian_kernel_1d(self.kernel)

    self.vp = copy.copy(VP_INIT)
    self.cal_status = Calibration.UNCALIBRATED
    self.frame_counter = 0
    self.params = Params()
    calibration_params = self.params.get("CalibrationParams")
    if calibration_params:
      try:
        calibration_params = json.loads(calibration_params)
        self.vp = np.array(calibration_params["vanishing_point"])
        self.cal_status = Calibration.CALIBRATED if is_calibration_valid(self.vp) else Calibration.INVALID
        self.vp_cycles = VP_CYCLES_NEEDED
        self.frame_counter = CALIBRATION_CYCLES_NEEDED
      except Exception:
        cloudlog.exception("CalibrationParams file found but error encountered")

    self.vp_unfilt = self.vp
    self.orb_last_updated = 0.
    self.reset_grid()

  def reset_grid(self):
    if self.cal_status == Calibration.UNCALIBRATED:
      # empty grid
      self.grid = np.zeros((H+1, W+1), dtype=np.float)
    else:
      # gaussian grid, centered at vp
      self.grid = gaussian_kernel(W/2., H/2., 16, 16, self.vp[0] - W/2., self.vp[1] - H/2.) * GRID_WEIGHT_INIT

  def rescale_grid(self):
    self.grid *= 0.9

  def update(self, uvs, yaw_rate, speed):
    if len(uvs) < 10 or \
       abs(yaw_rate) > Filter.MAX_YAW_RATE or \
       speed < Filter.MIN_SPEED:
      return
    rot_speeds = np.array([0.,0.,-yaw_rate])
    uvs[:,1,:] = denormalize(correct_pts(normalize(uvs[:,1,:]), rot_speeds, self.dt))
    # exclude tracks where:
    # - pixel movement was less than 10 pixels
    # - tracks are in the "hood region"
    good_tracks = np.all([np.linalg.norm(uvs[:,1,:] - uvs[:,0,:], axis=1) > 10,
                  uvs[:,0,1] < HOOD_HEIGHT,
                  uvs[:,1,1] < HOOD_HEIGHT], axis = 0)
    uvs = uvs[good_tracks]
    if uvs.shape[0] > MAX_LINES:
      uvs = uvs[np.random.choice(uvs.shape[0], MAX_LINES, replace=False), :]
    lines = get_lines(uvs)

    increment_grid_c(self.grid, lines, len(lines))
    self.frame_counter += 1
    if (self.frame_counter % FRAMES_NEEDED) == 0:
      grid = blur_image(self.grid, self.kernel_x, self.kernel_y)
      argmax_vp = np.unravel_index(np.argmax(grid), grid.shape)[::-1]
      self.rescale_grid()
      self.vp_unfilt = np.array(argmax_vp)
      self.vp_cycles += 1
      if (self.vp_cycles - VP_CYCLES_NEEDED) % 10 == 0:    # update file every 10
        # skip rate_lim before writing the file to avoid writing a lagged vp
        if self.cal_status != Calibration.CALIBRATED:
          self.vp = self.vp_unfilt
        if self.param_put:
          cal_params = {"vanishing_point": list(self.vp), "angle_offset2": self.angle_offset}
          self.params.put("CalibrationParams", json.dumps(cal_params))
      return self.vp_unfilt

  def parse_orb_features(self, log):
    matches = np.array(log.orbFeatures.matches)
    n = len(log.orbFeatures.matches)
    t = float(log.orbFeatures.timestampLastEof)*1e-9
    if t == 0 or n == 0:
      return t, np.zeros((0,2,2))
    orbs = denormalize(np.column_stack((log.orbFeatures.xs,
                                        log.orbFeatures.ys)))
    if self.prev_orbs is not None:
      valid_matches = (matches > -1) & (matches < len(self.prev_orbs))
      tracks = np.hstack((orbs[valid_matches], self.prev_orbs[matches[valid_matches]])).reshape((-1,2,2))
    else:
      tracks = np.zeros((0,2,2))
    self.prev_orbs = orbs
    return t, tracks

  def update_vp(self):
    # rate limit to not move VP too fast
    self.vp = np.clip(self.vp_unfilt, self.vp - VP_RATE_LIM, self.vp + VP_RATE_LIM)
    if self.vp_cycles < VP_CYCLES_NEEDED:
      self.cal_status = Calibration.UNCALIBRATED
    else:
      self.cal_status = Calibration.CALIBRATED if is_calibration_valid(self.vp) else Calibration.INVALID

  def handle_orb_features(self, log):
    self.update_vp()
    self.time_orb, frame_raw = self.parse_orb_features(log)
    self.dt = min(self.time_orb - self.orb_last_updated, 1.)
    self.orb_last_updated = self.time_orb
    if self.time_orb - self.l100_last_updated < 1:
      return self.update(frame_raw, self.yaw_rate, self.speed)

  def handle_live100(self, log):
    self.l100_last_updated = self.time_orb
    self.speed = log.live100.vEgo
    self.angle_offset = log.live100.angleOffset
    self.yaw_rate = log.live100.curvature * self.speed

  def handle_debug(self):
    grid_blurred = blur_image(self.grid, self.kernel_x, self.kernel_y)
    grid_grey = np.clip(grid_blurred/(0.1 + np.max(grid_blurred))*255, 0, 255)
    grid_color = np.repeat(grid_grey[:,:,np.newaxis], 3, axis=2)
    grid_color[:,:,0] = 0
    cv2.circle(grid_color, tuple(self.vp.astype(np.int64)), 2, (255, 255, 0), -1)
    cv2.imshow("debug", grid_color.astype(np.uint8))

    cv2.waitKey(1)

  def send_data(self, livecalibration):
    calib = get_calib_from_vp(self.vp)
    extrinsic_matrix = get_view_frame_from_road_frame(0, calib[1], calib[2], model_height)
    ke = eon_intrinsics.dot(extrinsic_matrix)
    warp_matrix = get_camera_frame_from_model_frame(ke, model_height)

    cal_send = messaging.new_message()
    cal_send.init('liveCalibration')
    cal_send.liveCalibration.calStatus = self.cal_status
    cal_send.liveCalibration.calPerc = min(self.frame_counter * 100 / CALIBRATION_CYCLES_NEEDED, 100)
    cal_send.liveCalibration.warpMatrix2 = map(float, warp_matrix.flatten())
    cal_send.liveCalibration.extrinsicMatrix = map(float, extrinsic_matrix.flatten())

    livecalibration.send(cal_send.to_bytes())


def calibrationd_thread(gctx=None, addr="127.0.0.1"):
  context = zmq.Context()

  live100 = messaging.sub_sock(context, service_list['live100'].port, addr=addr, conflate=True)
  orbfeatures = messaging.sub_sock(context, service_list['orbFeatures'].port, addr=addr, conflate=True)
  livecalibration = messaging.pub_sock(context, service_list['liveCalibration'].port)

  calibrator = Calibrator(param_put = True)

  # buffer with all the messages that still need to be input into the kalman
  while 1:
    of = messaging.recv_one(orbfeatures)
    l100 = messaging.recv_one_or_none(live100)

    new_vp = calibrator.handle_orb_features(of)
    if DEBUG and new_vp is not None:
      print 'got new vp', new_vp
    if l100 is not None:
      calibrator.handle_live100(l100)
    if DEBUG:
      calibrator.handle_debug()

    calibrator.send_data(livecalibration)


def main(gctx=None, addr="127.0.0.1"):
  calibrationd_thread(gctx, addr)

if __name__ == "__main__":
  main()

