import math
import numpy as np
from selfdrive.controls.lib.pid import PIController
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from common.numpy_fast import interp
from common.realtime import sec_since_boot
from selfdrive.swaglog import cloudlog
from cereal import car
from datetime import datetime

_DT = 0.01    # 100Hz
_DT_MPC = 0.05  # 20Hz


def calc_states_after_delay(states, v_ego, steer_angle, curvature_factor, steer_ratio, delay):
  states[0].x = v_ego * delay
  states[0].psi = v_ego * curvature_factor * math.radians(steer_angle) / steer_ratio * delay
  return states


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


class LatControl(object):
  def __init__(self, CP):
    self.pid = PIController((CP.steerKpBP, CP.steerKpV),
                            (CP.steerKiBP, CP.steerKiV),
                            k_f=CP.steerKf, pos_limit=1.0)
    self.last_cloudlog_t = 0.0
    self.setup_mpc(CP.steerRateCost)

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

    self.last_mpc_ts = 0.0
    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_prev = 0.0
    self.angle_steers_des_time = 0.0

    # Stuck actuator tracking
    self.angle_jolt = 1.0                      # When stuck, set the next mpc angle to at least this much, right or left
    self.stuck_ms = 100                        # Declare actuator stuck after x milliseconds. latcontrol seems to run every 60ms though
    self.stuck_check1 = 1500.0                 # Save the previously recorded angle_steers (start with impossible value)
    self.stuck_check2 = 2000.0                 # Check a second, adjeacent angle to detect rapid oscillations
    self.angle_steers_same = 0                 # Save a total count of near-identical angle_steers values
    self.stuck_start_time = 9000000000.000000  # Track time since angle_steers has shown the same value (start with far future date)
    #self.actuator_stuck = False                # Track if actuator is likely stuck
    self.orig_mpc = 0.0
    self.jolt_mpc = 0.0
    self.jolt_loops = 0
    self.jolt_corr_loops = 0


  def reset(self):
    self.pid.reset()

  def update(self, active, v_ego, angle_steers, steer_override, d_poly, angle_offset, CP, VM, PL):
    cur_time = sec_since_boot()
    self.mpc_updated = False
    # TODO: this creates issues in replay when rewinding time: mpc won't run
    if self.last_mpc_ts < PL.last_md_ts:
      self.last_mpc_ts = PL.last_md_ts
      self.angle_steers_des_prev = self.angle_steers_des_mpc

      curvature_factor = VM.curvature_factor(v_ego)

      l_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.l_poly))
      r_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.r_poly))
      p_poly = libmpc_py.ffi.new("double[4]", list(PL.PP.p_poly))

      # account for actuation delay
      self.cur_state = calc_states_after_delay(self.cur_state, v_ego, angle_steers, curvature_factor, CP.steerRatio, CP.steerActuatorDelay)

      v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
      self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                          l_poly, r_poly, p_poly,
                          PL.PP.l_prob, PL.PP.r_prob, PL.PP.p_prob, curvature_factor, v_ego_mpc, PL.PP.lane_width)

      # reset to current steer angle if not active or overriding
      if active:
        delta_desired = self.mpc_solution[0].delta[1]
      else:
        delta_desired = math.radians(angle_steers - angle_offset) / CP.steerRatio

      self.cur_state[0].delta = delta_desired

      self.angle_steers_des_mpc = float(math.degrees(delta_desired * CP.steerRatio) + angle_offset)
      self.angle_steers_des_time = cur_time
      self.mpc_updated = True





# tmux logging
print(datetime.utcfromtimestamp(cur_time - 28800).strftime('%Y-%m-%d %H:%M:%S'), " "),
print("angle_steers: ", angle_steers, " "),
print("MPC: ", round(self.angle_steers_des_mpc, 2), " "),
print("O-MPC: ", round(self.orig_mpc, 2), " "),
print("jolt_mpc: ", round(self.jolt_mpc, 2), " "),
print("jolt_loops: ", self.jolt_loops, " ")
# new line
print("time stuck: ", round(self.stuck_start_time, 3), " "),
print("angle_jolt: ", round(self.angle_jolt, 2), " "),
print("corr_loops: ", self.jolt_corr_loops, " ")

# Check for a desired angle between x1 and x2
if not self.jolt_loops and not self.jolt_corr_loops and 0.1 < abs((angle_steers + 1000) - (self.angle_steers_des_mpc + 1000)) < 1.5:
  # check-for-stagnant-steering code goes here
  # Don't want to interfere with cornering for now. May want to try lowering this value depending on speed
  #--------------------------------------------------
  # Attempt to determine if steering has stopped moving so we can try and force a move at low angles.
  # How long has angle_steers been one of two values no more than 0.1 apart?
  # Cabana graphs for Prius often lands right between two 1/10th numbers.

  if abs(angle_steers) < 5:  # Might move this as an AND to the above (unless we want to something for corners too)
    if (abs((angle_steers + 1000) - (self.stuck_check1 + 1000)) < 0.04 or abs((angle_steers + 1000) - (self.stuck_check2 + 1000)) < 0.04) and abs((self.stuck_check1 + 1000) - (self.stuck_check2 + 1000)) <= 0.14:
      self.angle_steers_same += 1
      if self.stuck_check2 == 2000.0:
        self.stuck_check2 = angle_steers
    # Is angle_steers about 0.1 away from either tracked value AND more than 0.1 away from the other?
    elif (abs((angle_steers + 1000) - (self.stuck_check1 + 1000)) <= 0.14 or abs((angle_steers + 1000) - (self.stuck_check2 + 1000)) <= 0.14) and (abs((angle_steers + 1000) - (self.stuck_check1 + 1000)) > 0.14 or abs((angle_steers + 1000) - (self.stuck_check2 + 1000)) > 0.14):
      if abs((angle_steers + 1000) - (self.stuck_check1 + 1000)) > 0.14:
        self.stuck_check2 = 2000.0
        self.stuck_check1 = angle_steers
      self.angle_steers_same = 0
      self.jolt_loops = 1
      self.stuck_check2 = angle_steers
      self.stuck_start_time = sec_since_boot() # Comes from time.time() [via panda] or seconds/nanoseconds since 1970. 0.5095601 = 509560100 nanoseconds
    elif abs((angle_steers + 1000) - (self.stuck_check1 + 1000)) < 0.04:
      self.angle_steers_same += 1
    else:
      self.angle_steers_same = 0
      self.jolt_loops = 0
      self.stuck_check1 = angle_steers
      self.stuck_check2 = 2000.0
      self.stuck_start_time = sec_since_boot()

  if self.angle_steers_same and sec_since_boot() - self.stuck_start_time >= self.stuck_ms * 0.001: # Turn ms into sec
    self.jolt_loops = 1
    self.orig_mpc = self.angle_steers_des_mpc
    #print("O-MPC: ", round(self.orig_mpc, 2), " "),
    
  #if self.angle_steers_des_mpc + 1000 > angle_steers + 1000:
  #  self.steer_direction = "left"
    # left calc here

  #else:
  #  self.steer_direction = "right"
    # right calc here

# self.jolt_loops will now be incremented by stuck detection code vs actuator_stuck

if self.jolt_loops:
  if (self.angle_steers_des_mpc and angle_steers) or (not self.angle_steers_des_mpc and not angle_steers):
    # Both are positive or both are negative so use subtraction
    # Is the difference less than self.angle_jolt in degrees?
    # This is the positive calculation
    if 0.01 < abs(self.angle_steers_des_mpc - angle_steers) < self.angle_jolt:        # Don't care if it's zero
      # Angle falls within angle_jolt window
      # Is the desired turn, to the left or to the right of the CURRENT position?
      if self.angle_steers_des_mpc < angle_steers:
        # It's to the right (smaller number than angle_steers)
        # Find the difference from 0.5 and make the new angle 0.5d _smaller_ than angle_steers
        # If we SAVE this difference to a variable, we could use it to immediately turn back to the correct position
        self.jolt_mpc = self.angle_steers_des_mpc - float(self.angle_jolt - (self.angle_jolt - abs((self.angle_steers_des_mpc - angle_steers))))
      else:
        # It's to the left
        # Find the difference from 0.5 and make the new angle 0.5d _bigger_ than angle_steers
        self.jolt_mpc = self.angle_steers_des_mpc + float(self.angle_jolt - (self.angle_jolt - abs((self.angle_steers_des_mpc - angle_steers))))
  else:
    # Only one is negative so use addition
    # Is the difference less than self.angle_jolt in degrees?
    # This is the addition calculation
    if 0.01 < abs(self.angle_steers_des_mpc + angle_steers) < self.angle_jolt:
      # Angle falls within angle_jolt window
      # Is the desired turn, to the left or to the right?
      if self.angle_steers_des_mpc < angle_steers:
        # It's to the right (smaller number than angle_steers)
        # Find the difference from 0.5 and make the new angle 0.5d _smaller_ than angle_steers
        self.jolt_mpc = self.angle_steers_des_mpc - float(self.angle_jolt - (self.angle_jolt - abs((self.angle_steers_des_mpc + angle_steers))))
      else:
        # It's to the left
        # Find the difference from 0.5 and make the new angle 0.5d _bigger_ than angle_steers
        self.jolt_mpc = self.angle_steers_des_mpc + float(self.angle_jolt - (self.angle_jolt - abs((self.angle_steers_des_mpc + angle_steers))))
  #print("jolt_mpc: ", round(self.jolt_mpc, 2), " "),

  # See if we've arrived near the intended jolt angle
  if abs((angle_steers + 1000) - (self.jolt_mpc + 1000)) <= abs(0.05 * self.jolt_mpc):  # within 5% of jolt_mpc. But, what if it's negative HUH? added abs to right side fixes it
    # jolt move done so correction move code goes here
    self.angle_steers_des_mpc = self.orig_mpc  # need to figure out how to apply multiplier; move slightly farther in the correction direction
    self.jolt_loops = 0
    self.jolt_corr_loops = 1
  else:
    self.angle_steers_des_mpc = self.jolt_mpc
    self.jolt_loops += 1
    self.jolt_corr_loops = 0  # necessary?

elif self.jolt_corr_loops:
  if abs((angle_steers + 1000) - (self.orig_mpc + 1000)) <= abs(0.05 * self.orig_mpc): # SHOULD be equal to or "farther left/right" of orig_mpc
  self.angle_steers_des_mpc = self.orig_mpc  # need to figure out how to apply multiplier; move slightly farther in the correction direction
  self.jolt_corr_loops += 1
  self.jolt_loops = 0  # necessary?










      #  Check for infeasable MPC solution
      self.mpc_nans = np.any(np.isnan(list(self.mpc_solution[0].delta)))
      t = sec_since_boot()
      if self.mpc_nans:
        self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.LANE, MPC_COST_LAT.HEADING, CP.steerRateCost)
        self.cur_state[0].delta = math.radians(angle_steers) / CP.steerRatio

        if t > self.last_cloudlog_t + 5.0:
          self.last_cloudlog_t = t
          cloudlog.warning("Lateral mpc - nan: True")

    if v_ego < 0.3 or not active:
      output_steer = 0.0
      self.pid.reset()
    else:
      # TODO: ideally we should interp, but for tuning reasons we keep the mpc solution
      # constant for 0.05s.
      #dt = min(cur_time - self.angle_steers_des_time, _DT_MPC + _DT) + _DT  # no greater than dt mpc + dt, to prevent too high extraps
      #self.angle_steers_des = self.angle_steers_des_prev + (dt / _DT_MPC) * (self.angle_steers_des_mpc - self.angle_steers_des_prev)
      self.angle_steers_des = self.angle_steers_des_mpc
      steers_max = get_steer_max(CP, v_ego)
      self.pid.pos_limit = steers_max
      self.pid.neg_limit = -steers_max
      steer_feedforward = self.angle_steers_des   # feedforward desired angle
      if CP.steerControlType == car.CarParams.SteerControlType.torque:
        steer_feedforward *= v_ego**2  # proportional to realigning tire momentum (~ lateral accel)
      deadzone = 0.0
      output_steer = self.pid.update(self.angle_steers_des, angle_steers, check_saturation=(v_ego > 10), override=steer_override,
                                     feedforward=steer_feedforward, speed=v_ego, deadzone=deadzone)

    self.sat_flag = self.pid.saturated
    return output_steer, float(self.angle_steers_des)
