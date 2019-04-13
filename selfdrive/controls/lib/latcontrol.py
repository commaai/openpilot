import numpy as np
from common.realtime import sec_since_boot
from selfdrive.controls.lib.pid import PIController
from common.numpy_fast import interp
from selfdrive.kegman_conf import kegman_conf
from cereal import car

_DT = 0.01    # 100Hz

def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)

class LatControl(object):
  def __init__(self, CP):

    kegman = kegman_conf(CP)
    self.gernbySteer = True
    self.mpc_frame = 0
    self.total_desired_projection = max(0.0, CP.steerMPCReactTime + CP.steerMPCDampTime)
    self.desired_smoothing = max(1.0, CP.steerMPCDampTime / _DT)
    self.dampened_actual_angle = 0.0
    self.dampened_desired_angle = 0.0
    self.dampened_desired_rate = 0.0
    self.previous_integral = 0.0
    self.last_cloudlog_t = 0.0
    self.angle_steers_des = 0.
    self.angle_ff_ratio = 0.0
    self.standard_ff_ratio = 0.0
    self.angle_ff_gain = 1.0
    self.rate_ff_gain = 0.01
    self.average_angle_steers = 0.
    self.angle_ff_bp = [[0.5, 5.0],[0.0, 1.0]]

    KpV = [interp(25.0, CP.steerKpBP, CP.steerKpV)]
    KiV = [interp(25.0, CP.steerKiBP, CP.steerKiV)]
    self.pid = PIController(([0.], KpV),
                            ([0.], KiV),
                            k_f=CP.steerKf, pos_limit=1.0)

  def live_tune(self, CP):
    self.mpc_frame += 1
    if self.mpc_frame % 300 == 0:
      # live tuning through /data/openpilot/tune.py overrides interface.py settings
      kegman = kegman_conf()
      if kegman.conf['tuneGernby'] == "1":
        self.steerKpV = np.array([float(kegman.conf['Kp'])])
        self.steerKiV = np.array([float(kegman.conf['Ki'])])
        self.total_desired_projection = max(0.0, float(kegman.conf['dampMPC']) + float(kegman.conf['reactMPC']))
        self.desired_smoothing = max(1.0, float(kegman.conf['dampMPC']) / _DT)
        self.rate_ff_gain = float(kegman.conf['rateFF'])
        self.gernbySteer = (self.total_desired_projection > 0 or self.desired_smoothing > 1)

        # Eliminate break-points, since they aren't needed (and would cause problems for resonance)
        KpV = [interp(25.0, CP.steerKpBP, self.steerKpV)]
        KiV = [interp(25.0, CP.steerKiBP, self.steerKiV)]
        self.pid._k_i = ([0.], KiV)
        self.pid._k_p = ([0.], KpV)
        self.standard_ff_ratio = 0.0
        print(self.rate_ff_gain, self.angle_ff_gain)
      else:
        self.gernbySteer = False
        self.standard_ff_ratio = 1.0
        self.angle_ff_ratio = 0.0
      self.mpc_frame = 0

  def reset(self):
    self.pid.reset()

  def adjust_angle_gain(self):
    if (self.pid.f > 0) == (self.pid.i > 0) and abs(self.pid.i) >= abs(self.previous_integral):
      self.angle_ff_gain *= 1.0001
    elif self.angle_ff_gain > 1.0:
      self.angle_ff_gain *= 0.9999
    self.previous_integral = self.pid.i

  def update(self, active, v_ego, angle_steers, steer_override, CP, VM, path_plan):

    self.live_tune(CP)

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.pid.reset()
      self.previous_integral = 0.0
      self.dampened_desired_angle = float(angle_steers)
      self.dampened_desired_rate = 0.0
    else:
      if self.gernbySteer == False:
        self.dampened_desired_angle = float(path_plan.angleSteers)
        self.dampened_desired_rate = float(path_plan.rateSteers)
      else:
        cur_time = sec_since_boot()
        projected_desired_angle = interp(cur_time + self.total_desired_projection, path_plan.mpcTimes, path_plan.mpcAngles)
        self.dampened_desired_angle += ((projected_desired_angle - self.dampened_desired_angle) / self.desired_smoothing)
        projected_desired_rate = interp(cur_time + self.total_desired_projection, path_plan.mpcTimes, path_plan.mpcRates)
        self.dampened_desired_rate += ((projected_desired_rate - self.dampened_desired_rate) / self.desired_smoothing)

      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        steers_max = get_steer_max(CP, v_ego)
        self.pid.pos_limit = steers_max
        self.pid.neg_limit = -steers_max
        deadzone = 0.0

        if self.gernbySteer:
          angle_feedforward = self.dampened_desired_angle - path_plan.angleOffset
          self.angle_ff_ratio = interp(abs(angle_feedforward), self.angle_ff_bp[0], self.angle_ff_bp[1])
          angle_feedforward *= self.angle_ff_ratio * self.angle_ff_gain
          rate_feedforward = (1.0 - self.angle_ff_ratio) * self.rate_ff_gain * self.dampened_desired_rate
          steer_feedforward = v_ego**2 * (rate_feedforward + angle_feedforward)
        else:
          steer_feedforward = v_ego**2 * (self.dampened_desired_angle - path_plan.angleOffset)
          print(steer_feedforward)

        output_steer = self.pid.update(self.dampened_desired_angle, angle_steers, check_saturation=(v_ego > 10),
                                      override=steer_override, feedforward=steer_feedforward, speed=v_ego, deadzone=deadzone)

        if self.gernbySteer and not steer_override and v_ego > 10.0:
          if abs(angle_steers) > (self.angle_ff_bp[0][1] / 2.0):
            self.adjust_angle_gain()
          else:
            self.previous_integral = self.pid.i

    self.sat_flag = self.pid.saturated
    self.average_angle_steers += 0.1 * (angle_steers - self.average_angle_steers)

    if CP.steerControlType == car.CarParams.SteerControlType.torque:
      return float(output_steer), float(path_plan.angleSteers)
    else:
      return float(self.dampened_desired_angle), float(path_plan.angleSteers)
