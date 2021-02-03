import os
import math
import numpy as np
from common.realtime import sec_since_boot, DT_MDL
from common.numpy_fast import interp
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc import libmpc_py
from selfdrive.controls.lib.drive_helpers import MPC_COST_LAT, MPC_N, CAR_ROTATION_RADIUS
from selfdrive.controls.lib.lane_planner import LanePlanner, TRAJECTORY_SIZE
from selfdrive.config import Conversions as CV
from common.params import Params
import cereal.messaging as messaging
from cereal import log

LaneChangeState = log.PathPlan.LaneChangeState
LaneChangeDirection = log.PathPlan.LaneChangeDirection

LOG_MPC = os.environ.get('LOG_MPC', False)

LANE_CHANGE_SPEED_MIN = 45 * CV.MPH_TO_MS
LANE_CHANGE_TIME_MAX = 10.

DESIRES = {
  LaneChangeDirection.none: {
    LaneChangeState.off: log.PathPlan.Desire.none,
    LaneChangeState.preLaneChange: log.PathPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.PathPlan.Desire.none,
    LaneChangeState.laneChangeFinishing: log.PathPlan.Desire.none,
  },
  LaneChangeDirection.left: {
    LaneChangeState.off: log.PathPlan.Desire.none,
    LaneChangeState.preLaneChange: log.PathPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.PathPlan.Desire.laneChangeLeft,
    LaneChangeState.laneChangeFinishing: log.PathPlan.Desire.laneChangeLeft,
  },
  LaneChangeDirection.right: {
    LaneChangeState.off: log.PathPlan.Desire.none,
    LaneChangeState.preLaneChange: log.PathPlan.Desire.none,
    LaneChangeState.laneChangeStarting: log.PathPlan.Desire.laneChangeRight,
    LaneChangeState.laneChangeFinishing: log.PathPlan.Desire.laneChangeRight,
  },
}


class PathPlanner():
  def __init__(self, CP):
    self.LP = LanePlanner()

    self.last_cloudlog_t = 0
    self.steer_rate_cost = CP.steerRateCost

    self.setup_mpc()
    self.solution_invalid_cnt = 0
    self.lane_change_enabled = Params().get('LaneChangeEnabled') == b'1'
    self.lane_change_state = LaneChangeState.off
    self.lane_change_direction = LaneChangeDirection.none
    self.lane_change_timer = 0.0
    self.lane_change_ll_prob = 1.0
    self.prev_one_blinker = False
    self.desire = log.PathPlan.Desire.none

    self.path_xyz = np.zeros((TRAJECTORY_SIZE,3))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.y_pts = np.zeros(TRAJECTORY_SIZE)

  def setup_mpc(self):
    self.libmpc = libmpc_py.libmpc
    self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, self.steer_rate_cost)

    self.mpc_solution = libmpc_py.ffi.new("log_t *")
    self.cur_state = libmpc_py.ffi.new("state_t *")
    self.cur_state[0].x = 0.0
    self.cur_state[0].y = 0.0
    self.cur_state[0].psi = 0.0
    self.cur_state[0].tire_angle = 0.0

    self.angle_steers_des = 0.0
    self.angle_steers_des_mpc = 0.0
    self.angle_steers_des_time = 0.0

  def update(self, sm, CP, VM):
    v_ego = sm['carState'].vEgo
    active = sm['controlsState'].active
    steering_wheel_angle_offset_deg = sm['liveParameters'].angleOffset
    steering_wheel_angle_deg = sm['carState'].steeringAngle
    measured_tire_angle = -math.radians(steering_wheel_angle_deg - steering_wheel_angle_offset_deg) / VM.sR


    # Update vehicle model
    x = max(sm['liveParameters'].stiffnessFactor, 0.1)
    sr = max(sm['liveParameters'].steerRatio, 0.1)
    VM.update_params(x, sr)
    curvature_factor = VM.curvature_factor(v_ego)


    md = sm['modelV2']
    self.LP.parse_model(sm['modelV2'])
    if len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE:
      self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      self.t_idxs = np.array(md.position.t)
      self.plan_yaw = list(md.orientation.z)

    # Lane change logic
    one_blinker = sm['carState'].leftBlinker != sm['carState'].rightBlinker
    below_lane_change_speed = v_ego < LANE_CHANGE_SPEED_MIN

    if sm['carState'].leftBlinker:
      self.lane_change_direction = LaneChangeDirection.left
    elif sm['carState'].rightBlinker:
      self.lane_change_direction = LaneChangeDirection.right

    if (not active) or (self.lane_change_timer > LANE_CHANGE_TIME_MAX) or (not self.lane_change_enabled):
      self.lane_change_state = LaneChangeState.off
      self.lane_change_direction = LaneChangeDirection.none
    else:
      torque_applied = sm['carState'].steeringPressed and \
                       ((sm['carState'].steeringTorque > 0 and self.lane_change_direction == LaneChangeDirection.left) or
                        (sm['carState'].steeringTorque < 0 and self.lane_change_direction == LaneChangeDirection.right))

      blindspot_detected = ((sm['carState'].leftBlindspot and self.lane_change_direction == LaneChangeDirection.left) or
                            (sm['carState'].rightBlindspot and self.lane_change_direction == LaneChangeDirection.right))

      lane_change_prob = self.LP.l_lane_change_prob + self.LP.r_lane_change_prob

      # State transitions
      # off
      if self.lane_change_state == LaneChangeState.off and one_blinker and not self.prev_one_blinker and not below_lane_change_speed:
        self.lane_change_state = LaneChangeState.preLaneChange
        self.lane_change_ll_prob = 1.0

      # pre
      elif self.lane_change_state == LaneChangeState.preLaneChange:
        if not one_blinker or below_lane_change_speed:
          self.lane_change_state = LaneChangeState.off
        elif torque_applied and not blindspot_detected:
          self.lane_change_state = LaneChangeState.laneChangeStarting

      # starting
      elif self.lane_change_state == LaneChangeState.laneChangeStarting:
        # fade out over .5s
        self.lane_change_ll_prob = max(self.lane_change_ll_prob - 2*DT_MDL, 0.0)
        # 98% certainty
        if lane_change_prob < 0.02 and self.lane_change_ll_prob < 0.01:
          self.lane_change_state = LaneChangeState.laneChangeFinishing

      # finishing
      elif self.lane_change_state == LaneChangeState.laneChangeFinishing:
        # fade in laneline over 1s
        self.lane_change_ll_prob = min(self.lane_change_ll_prob + DT_MDL, 1.0)
        if one_blinker and self.lane_change_ll_prob > 0.99:
          self.lane_change_state = LaneChangeState.preLaneChange
        elif self.lane_change_ll_prob > 0.99:
          self.lane_change_state = LaneChangeState.off

    if self.lane_change_state in [LaneChangeState.off, LaneChangeState.preLaneChange]:
      self.lane_change_timer = 0.0
    else:
      self.lane_change_timer += DT_MDL

    self.prev_one_blinker = one_blinker

    self.desire = DESIRES[self.lane_change_direction][self.lane_change_state]

    # Turn off lanes during lane change
    if self.desire == log.PathPlan.Desire.laneChangeRight or self.desire == log.PathPlan.Desire.laneChangeLeft:
      self.LP.lll_prob *= self.lane_change_ll_prob
      self.LP.rll_prob *= self.lane_change_ll_prob
    d_path_xyz = self.LP.get_d_path(v_ego, self.t_idxs, self.path_xyz)
    y_pts = np.interp(v_ego * self.t_idxs[:MPC_N+1], np.linalg.norm(d_path_xyz, axis=1), d_path_xyz[:,1])
    heading_pts = np.interp(v_ego * self.t_idxs[:MPC_N+1], np.linalg.norm(self.path_xyz, axis=1), self.plan_yaw)
    self.y_pts = y_pts

    v_ego_mpc = max(v_ego, 5.0)  # avoid mpc roughness due to low speed
    assert len(y_pts) == MPC_N + 1
    assert len(heading_pts) == MPC_N + 1
    self.libmpc.run_mpc(self.cur_state, self.mpc_solution,
                        float(v_ego_mpc),
                        curvature_factor,
                        CAR_ROTATION_RADIUS,
                        list(y_pts),
                        list(heading_pts))
    # init state for next
    self.cur_state.x = 0.0
    self.cur_state.y = 0.0
    self.cur_state.psi = 0.0
    self.cur_state.tire_angle = interp(DT_MDL, self.t_idxs[:MPC_N+1], self.mpc_solution.tire_angle)

    # TODO this needs more thought, use .2s extra for now to estimate other delays
    delay = CP.steerActuatorDelay + .2
    next_tire_angle = interp(DT_MDL + delay, self.t_idxs[:MPC_N+1], self.mpc_solution.tire_angle)
    next_tire_angle_rate = self.mpc_solution.tire_angle_rate[0]

    # TODO This gets around the fact that MPC can plan to turn and turn back in the
    # time between now and delay, need better abstraction between planner and controls
    plan_ahead_idx = sum(self.t_idxs < delay)
    if next_tire_angle_rate > 0:
      next_tire_angle = max(list(self.mpc_solution.tire_angle)[:plan_ahead_idx] + [next_tire_angle])
    else:
      next_tire_angle = min(list(self.mpc_solution.tire_angle)[:plan_ahead_idx] + [next_tire_angle])

    # reset to current steer angle if not active or overriding
    if active:
      tire_angle_desired = next_tire_angle
      desired_tire_angle_rate = next_tire_angle_rate
    else:
      tire_angle_desired = measured_tire_angle
      desired_tire_angle_rate = 0.0

    # negative sign, controls uses different convention
    self.desired_steering_wheel_angle_deg = -float(math.degrees(tire_angle_desired * VM.sR)) + steering_wheel_angle_offset_deg
    self.desired_steering_wheel_angle_rate_deg = -float(math.degrees(desired_tire_angle_rate * VM.sR))

    #  Check for infeasable MPC solution
    mpc_nans = any(math.isnan(x) for x in self.mpc_solution.tire_angle)
    t = sec_since_boot()
    if mpc_nans:
      self.libmpc.init(MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, CP.steerRateCost)
      self.cur_state.tire_angle = measured_tire_angle

      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.mpc_solution[0].cost > 20000. or mpc_nans:   # TODO: find a better way to detect when MPC did not converge
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0

  def publish(self, sm, pm):
    plan_solution_valid = self.solution_invalid_cnt < 2
    plan_send = messaging.new_message('pathPlan')
    plan_send.valid = sm.all_alive_and_valid(service_list=['carState', 'controlsState', 'liveParameters', 'modelV2'])
    plan_send.pathPlan.laneWidth = float(self.LP.lane_width)
    plan_send.pathPlan.dPathPoints = [float(x) for x in self.y_pts]
    plan_send.pathPlan.lProb = float(self.LP.lll_prob)
    plan_send.pathPlan.rProb = float(self.LP.rll_prob)
    plan_send.pathPlan.dProb = float(self.LP.d_prob)

    plan_send.pathPlan.angleSteers = float(self.desired_steering_wheel_angle_deg)
    plan_send.pathPlan.rateSteers = float(self.desired_steering_wheel_angle_rate_deg)
    plan_send.pathPlan.angleOffset = float(sm['liveParameters'].angleOffsetAverage)
    plan_send.pathPlan.mpcSolutionValid = bool(plan_solution_valid)
    plan_send.pathPlan.paramsValid = bool(sm['liveParameters'].valid)

    plan_send.pathPlan.desire = self.desire
    plan_send.pathPlan.laneChangeState = self.lane_change_state
    plan_send.pathPlan.laneChangeDirection = self.lane_change_direction

    pm.send('pathPlan', plan_send)

    if LOG_MPC:
      dat = messaging.new_message('liveMpc')
      dat.liveMpc.x = list(self.mpc_solution[0].x)
      dat.liveMpc.y = list(self.mpc_solution[0].y)
      dat.liveMpc.psi = list(self.mpc_solution[0].psi)
      dat.liveMpc.tire_angle = list(self.mpc_solution[0].tire_angle)
      dat.liveMpc.cost = self.mpc_solution[0].cost
      pm.send('liveMpc', dat)
