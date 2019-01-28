import zmq
import math
import numpy as np
import time
import json
from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from common.numpy_fast import interp
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from cereal import car

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay):
  states[0].x = v_ego * delay
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * delay
  return states


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)

def apply_deadzone(angle, deadzone):
  if angle > deadzone:
    angle -= deadzone
  elif angle < -deadzone:
    angle += deadzone
  else:
    angle = 0.
  return angle

class LatControl(object):
  def __init__(self, CP):
    self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                            (CP.steerKiBP, CP.steerKiV),
                            k_f=CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.setup_mpc(CP.steerRateCost)
    self.smooth_factor = 2.0 * CP.steerActuatorDelay / _DT      # Multiplier for inductive component (feed forward)
    self.projection_factor = 5.0 * _DT                       #  Mutiplier for reactive component (PI)
    self.accel_limit = 2.0                                 # Desired acceleration limit to prevent "whip steer" (resistive component)
    self.ff_angle_factor = 0.5         # Kf multiplier for angle-based feed forward
    self.ff_rate_factor = 5.0         # Kf multiplier for rate-based feed forward
    self.ratioDelayExp = 2.0           # Exponential coefficient for variable steering ratio (delay)
    self.ratioDelayScale = 0.0          # Multiplier for variable steering ratio (delay)
    self.ratioScale = 6.0            # Multiplier for variable steering ratio
    self.ratioExp = 2.0                # Exponential coefficient for variable steering assist (torque)
    self.ratioAdjust = 0.85            # Fudge factor to preserve existing tuning parameters
    self.prev_angle_rate = 0
    self.feed_forward = 0.0
    self.steerActuatorDelay = CP.steerActuatorDelay
    self.angle_rate_desired = 0.0
    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_prev = 0.0
    self.angle_steers_des_time = 0.0
    self.avg_angle_steers = 0.0
    self.last_y = 0.0
    self.new_y = 0.0
    self.angle_sample_count = 0.0
    self.projected_angle_steers = 0.0
    self.lateral_error = 0.0

    # variables for dashboarding
    self.context = zmq.Context()
    self.steerpub = self.context.socket(zmq.PUB)
    self.steerpub.bind("tcp://*:8594")
    self.influxString = 'steerData3,testName=none,active=%s,ff_type=%s ff_type_a=%s,ff_type_r=%s,steer_status=%s,steer_torque_motor=%s,' \
                    'steering_control_active=%s,steer_parameter1=%s,steer_parameter2=%s,steer_parameter3=%s,steer_parameter4=%s,steer_parameter5=%s,' \
                    'steer_parameter6=%s,steer_stock_torque=%s,steer_stock_torque_request=%s,x=%s,y=%s,lateral_error=%s,y0=%s,y1=%s,y2=%s,y3=%s,y4=%s,psi=%s,delta=%s,t=%s,' \
                    'curvature_factor=%s,slip_factor=%s,resonant_period=%s,accel_limit=%s,restricted_steer_rate=%s,ff_angle_factor=%s,ff_rate_factor=%s,' \
                    'pCost=%s,lCost=%s,rCost=%s,hCost=%s,srCost=%s,torque_motor=%s,driver_torque=%s,angle_rate_count=%s,angle_rate_desired=%s,' \
                    'avg_angle_rate=%s,future_angle_steers=%s,angle_rate=%s,steer_zero_crossing=%s,center_angle=%s,angle_steers=%s,angle_steers_des=%s,' \
                    'angle_offset=%s,self.angle_steers_des_mpc=%s,steerRatio=%s,steerKf=%s,steerKpV[0]=%s,steerKiV[0]=%s,steerRateCost=%s,l_prob=%s,' \
                    'r_prob=%s,c_prob=%s,p_prob=%s,l_poly[0]=%s,l_poly[1]=%s,l_poly[2]=%s,l_poly[3]=%s,r_poly[0]=%s,r_poly[1]=%s,r_poly[2]=%s,r_poly[3]=%s,' \
                    'p_poly[0]=%s,p_poly[1]=%s,p_poly[2]=%s,p_poly[3]=%s,c_poly[0]=%s,c_poly[1]=%s,c_poly[2]=%s,c_poly[3]=%s,d_poly[0]=%s,d_poly[1]=%s,' \
                    'd_poly[2]=%s,lane_width=%s,lane_width_estimate=%s,lane_width_certainty=%s,v_ego=%s,p=%s,i=%s,f=%s %s\n~'
    self.steerdata = self.influxString
    self.frames = 0
    self.curvature_factor = 0.0
    self.slip_factor = 0.0
    self.isActive = 0

  def setup_mpc(self, steer_rate_cost):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, steer_rate_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.mpc_updated = False
    self.mpc_nans = False
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].delta = 0.0

  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, angle_rate, steer_override, d_poly, angle_offset, CP, VM, PL):
    cur_time = sec_since_boot()
    self.mpc_updated = False
    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      self.angle_steers_des_prev = self.angle_steers_des_mpc

      # Use the model's solve time instead of cur_time
      self.angle_steers_des_time = float(self.last_mpc_ts / 1000000000.0)
      self.curvature_factor = VM.curvature_factor(v_ego)

      # This is currently disabled, but it is used to compensate for variable steering rate
      ratioDelayFactor = 1. + self.ratioDelayScale * abs(angle_steers / 100.) ** self.ratioDelayExp

      # Determine a proper delay time that includes the model's processing time, which is variable
      plan_age = _DT_MPC + cur_time - float(self.last_mpc_ts / 1000000000.0)
      total_delay = ratioDelayFactor * CP.steerActuatorDelay + plan_age

      # Use steering rate from the last 2 samples to estimate acceleration for a more realistic future steering rate
      accelerated_angle_rate = 2.0 * angle_rate - self.prev_angle_rate

      # Project the future steering angle for the actuator delay only (not model delay)
      self.projected_angle_steers = ratioDelayFactor * CP.steerActuatorDelay * accelerated_angle_rate + angle_steers

      self.l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      self.r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      self.p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay and the age of the plan
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, self.projected_angle_steers, self.curvature_factor, CP.steerRatio, total_delay)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          self.l_poly, self.r_poly, self.p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, self.curvature_factor, v_ego_mpc, PL.PP.lane_width)

      # reset to current steer angle if not active or overriding
      if active:
        self.isActive = 1
        delta_desired = self.mpc_solution[0].delta[1]
      else:
        self.isActive = 0
        delta_desired = math.radians(angle_steers - angle_offset) / CP.steerRatio

      self.cur_state[0].delta = delta_desired

      self.angle_steers_des_mpc = float(math.degrees(delta_desired * CP.steerRatio) + angle_offset)

      # Use last 2 desired angles to determine the model's desired steer rate
      self.angle_rate_desired = (self.angle_steers_des_mpc - self.angle_steers_des_prev) / _DT_MPC
      self.mpc_updated = True

      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      t = sec_since_boot()
      if self.mpc_nans:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / CP.steerRatio

        if t > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = t
          cloudlog.warning("Lateral mpc - nan: True")

    elif self.frames > 20:
      self.steerpub.send(self.steerdata)
      self.frames = 0
      self.steerdata = self.influxString

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.feed_forward = 0.0
      self.pid.reset()
      ff_type = "r"
      projected_angle_steers_des = 0.0
      projected_angle_steers = 0.0
      restricted_steer_rate = 0.0
    else:
      # Interpolate desired angle between MPC updates
      self.angle_steers_des = self.angle_steers_des_prev + self.angle_rate_desired * (cur_time - self.angle_steers_des_time)
      self.avg_angle_steers = (4.0 * self.avg_angle_steers + angle_steers) / 5.0

      # Determine the target steer rate for desired angle, but prevent the acceleration limit from being exceeded
      # Restricting the steer rate creates the resistive component needed for resonance
      restricted_steer_rate = np.clip(self.angle_steers_des - float(angle_steers) , float(angle_rate) - self.accel_limit, float(angle_rate) + self.accel_limit)

      # Determine projected desired angle that is within the acceleration limit (prevent the steering wheel from jerking)
      projected_angle_steers_des = self.angle_steers_des + self.projection_factor * restricted_steer_rate

      # Determine project angle steers using current steer rate
      projected_angle_steers = float(angle_steers) + self.projection_factor * float(angle_rate)

      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max

      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        # Decide which feed forward mode should be used (angle or rate).  Use more dominant mode, and only if conditions are met
        # Spread feed forward out over a period of time to make it more inductive (for resonance)
        if abs(self.ff_rate_factor * float(restricted_steer_rate)) > abs(self.ff_angle_factor * float(self.angle_steers_des) - float(angle_offset)) - 0.5 \
            and (abs(float(restricted_steer_rate)) > abs(angle_rate) or (float(restricted_steer_rate) < 0) != (angle_rate < 0)) \
            and (float(restricted_steer_rate) < 0) == (float(self.angle_steers_des) - float(angle_offset) - 0.5 < 0):
          ff_type = "r"
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_rate_factor * v_ego**2 * float(restricted_steer_rate)) / self.smooth_factor
        elif abs(self.angle_steers_des - float(angle_offset)) > 0.5:
          ff_type = "a"
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + self.ff_angle_factor * v_ego**2 * float(apply_deadzone(float(self.angle_steers_des) - float(angle_offset), 0.5))) / self.smooth_factor
        else:
          ff_type = "r"
          self.feed_forward = (((self.smooth_factor - 1.) * self.feed_forward) + 0.0) / self.smooth_factor
      else:
        self.feed_forward = self.angle_steers_des   # feedforward desired angle
      deadzone = 0.0

      # Use projected desired and actual angles instead of "current" values, in order to make PI more reactive (for resonance)
      output_steer = self.pid.update(projected_angle_steers_des, projected_angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=self.feed_forward, speed=v_ego, deadzone=deadzone)


      # All but the last 3 lines after here are for real-time dashboarding
      self.pCost = 0.0
      self.lCost = 0.0
      self.rCost = 0.0
      self.hCost = 0.0
      self.srCost = 0.0
      self.last_ff_a = 0.0
      self.last_ff_r = 0.0
      self.center_angle = 0.0
      self.steer_zero_crossing = 0.0
      self.steer_rate_cost = 0.0
      self.avg_angle_rate = 0.0
      self.angle_rate_count = 0.0
      steer_motor = 0.0
      self.frames += 1
      steer_parameter1 = 0.0
      steer_parameter2 = 0.0
      steer_parameter3 = 0.0
      steer_parameter4 = 0.0
      steer_parameter5 = 0.0
      steer_parameter6 = 0.0
      steering_control_active = 0.0
      steer_torque_motor = 0.0
      driver_torque = 0.0
      steer_status = 0.0
      steer_stock_torque = 0.0
      steer_stock_torque_request = 0.0

      self.steerdata += ("%d,%s,%d,%d,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%f,%d|" % (self.isActive, \
      ff_type, 1 if ff_type == "a" else 0, 1 if ff_type == "r" else 0, steer_status, steer_torque_motor, steering_control_active, steer_parameter1, steer_parameter2, steer_parameter3, steer_parameter4, steer_parameter5, steer_parameter6, steer_stock_torque, steer_stock_torque_request, self.cur_state[0].x, self.cur_state[0].y, self.lateral_error, self.mpc_solution[0].y[0], self.mpc_solution[0].y[1], self.mpc_solution[0].y[2], self.mpc_solution[0].y[3], self.mpc_solution[0].y[4], self.cur_state[0].psi, self.cur_state[0].delta, self.cur_state[0].t, self.curvature_factor, self.slip_factor ,self.smooth_factor, self.accel_limit, float(restricted_steer_rate) ,self.ff_angle_factor, self.ff_rate_factor, self.pCost, self.lCost, self.rCost, self.hCost, self.srCost, steer_motor, float(driver_torque), \
      self.angle_rate_count, self.angle_rate_desired, self.avg_angle_rate, projected_angle_steers, float(angle_rate), self.steer_zero_crossing, self.center_angle, angle_steers, self.angle_steers_des, angle_offset, \
      self.angle_steers_des_mpc, CP.steerRatio, CP.steerKf, CP.steerKpV[0], CP.steerKiV[0], CP.steerRateCost, PL.PP.l_prob, \
      PL.PP.r_prob, PL.PP.c_prob, PL.PP.p_prob, self.l_poly[0], self.l_poly[1], self.l_poly[2], self.l_poly[3], self.r_poly[0], self.r_poly[1], self.r_poly[2], self.r_poly[3], \
      self.p_poly[0], self.p_poly[1], self.p_poly[2], self.p_poly[3], PL.PP.c_poly[0], PL.PP.c_poly[1], PL.PP.c_poly[2], PL.PP.c_poly[3], PL.PP.d_poly[0], PL.PP.d_poly[1], \
      PL.PP.d_poly[2], PL.PP.lane_width, PL.PP.lane_width_estimate, PL.PP.lane_width_certainty, v_ego, self.pid.p, self.pid.i, self.pid.f, int(time.time() * 100) * 10000000))

    self.sat_flag = self.pid.saturated
    self.prev_angle_rate = angle_rate
    return output_steer, float(self.angle_steers_des)
