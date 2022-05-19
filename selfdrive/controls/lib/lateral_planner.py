import numpy as np
from common.realtime import sec_since_boot, DT_MDL
from common.numpy_fast import interp
from selfdrive.swaglog import cloudlog
from selfdrive.controls.lib.lateral_mpc_lib.lat_mpc import LateralMpc, X_DIM
from selfdrive.controls.lib.drive_helpers import CONTROL_N, MPC_COST_LAT, LAT_MPC_N
from selfdrive.controls.lib.lane_planner import LanePlanner, TRAJECTORY_SIZE
from selfdrive.controls.lib.desire_helper import DesireHelper
import cereal.messaging as messaging
from cereal import log


class LateralPlanner:
  def __init__(self, CP, use_lanelines=True, wide_camera=False, use_rot_rad=False):
    self.use_lanelines = use_lanelines
    self.LP = LanePlanner(wide_camera)
    self.DH = DesireHelper()

    self.last_cloudlog_t = 0
    self.steer_rate_cost = CP.steerRateCost
    self.factor1 = (CP.wheelbase) if use_rot_rad else 0.0
    self.factor2 = ((2 * CP.centerToFront * CP.mass) / (CP.wheelbase * CP.tireStiffnessRear)) if use_rot_rad else 0.0
    self.solution_invalid_cnt = 0

    self.path_xyz = np.zeros((TRAJECTORY_SIZE, 3))
    self.path_xyz_stds = np.ones((TRAJECTORY_SIZE, 3))
    self.plan_yaw = np.zeros((TRAJECTORY_SIZE,))
    self.t_idxs = np.arange(TRAJECTORY_SIZE)
    self.y_pts = np.zeros(TRAJECTORY_SIZE)

    self.lat_mpc = LateralMpc()
    self.reset_mpc(np.zeros(X_DIM))

  def reset_mpc(self, x0=np.zeros(X_DIM)):
    self.x0 = x0
    self.lat_mpc.reset(x0=self.x0)

  def update(self, sm):
    v_ego = sm['carState'].vEgo
    measured_curvature = sm['controlsState'].curvature

    # Parse model predictions
    md = sm['modelV2']
    self.LP.parse_model(md)
    if len(md.position.x) == TRAJECTORY_SIZE and len(md.orientation.x) == TRAJECTORY_SIZE:
      self.path_xyz = np.column_stack([md.position.x, md.position.y, md.position.z])
      self.speed_forward = np.linalg.norm(np.column_stack([md.velocity.x, md.velocity.y, md.velocity.z]), axis=1)
      self.t_idxs = np.array(md.position.t)
      self.plan_yaw = list(md.orientation.z)
      self.plan_yaw_rate = list(md.orientationRate.z)
      self.plan_curvature = self.plan_yaw_rate / self.speed_forward
      self.plan_curvature_rate = np.gradient(self.plan_curvature, self.t_idxs)
    if len(md.position.xStd) == TRAJECTORY_SIZE:
      self.path_xyz_stds = np.column_stack([md.position.xStd, md.position.yStd, md.position.zStd])

    # Lane change logic
    lane_change_prob = self.LP.l_lane_change_prob + self.LP.r_lane_change_prob
    self.DH.update(sm['carState'], sm['controlsState'].active, lane_change_prob)

    # Turn off lanes during lane change
    if self.DH.desire == log.LateralPlan.Desire.laneChangeRight or self.DH.desire == log.LateralPlan.Desire.laneChangeLeft:
      self.LP.lll_prob *= self.DH.lane_change_ll_prob
      self.LP.rll_prob *= self.DH.lane_change_ll_prob

    # Calculate final driving path and set MPC costs
    if self.use_lanelines:
      d_path_xyz = self.LP.get_d_path(v_ego, self.t_idxs, self.path_xyz)
      self.lat_mpc.set_weights(MPC_COST_LAT.PATH, MPC_COST_LAT.HEADING, MPC_COST_LAT.CURV, MPC_COST_LAT.CURV_RATE)
    else:
      d_path_xyz = self.path_xyz
      heading_cost = interp(v_ego, [5.0, 10.0], [MPC_COST_LAT.HEADING, MPC_COST_LAT.HEADING_LL])
      self.lat_mpc.set_weights(MPC_COST_LAT.PATH, heading_cost, MPC_COST_LAT.CURV, MPC_COST_LAT.CURV_RATE)

    y_pts = d_path_xyz[:LAT_MPC_N + 1, 1]
    heading_pts = self.plan_yaw[:LAT_MPC_N + 1]
    curv_pts = self.plan_curvature[:LAT_MPC_N + 1]
    curv_rate_pts = self.plan_curvature_rate[:LAT_MPC_N + 1]
    self.y_pts = y_pts
    self.curv_rate_pts = curv_rate_pts

    assert len(y_pts) == LAT_MPC_N + 1
    assert len(heading_pts) == LAT_MPC_N + 1
    # p = np.array([v_ego, self.factor1 - (self.factor2 * v_ego**2)])
    lateral_factor = np.max(np.column_stack((np.zeros(LAT_MPC_N + 1), self.factor1 - (self.factor2 * self.speed_forward[:LAT_MPC_N + 1]**2))), axis=1)
    p = np.column_stack((self.speed_forward[:LAT_MPC_N + 1], lateral_factor))
    self.lat_mpc.run(self.x0,
                     p,
                     y_pts,
                     heading_pts,
                     curv_pts, 
                     curv_rate_pts)
    # init state for next
    self.x0[3] = interp(DT_MDL, self.t_idxs[:LAT_MPC_N + 1], self.lat_mpc.x_sol[:, 3])

    #  Check for infeasible MPC solution
    mpc_nans = np.isnan(self.lat_mpc.x_sol[:, 3]).any()
    t = sec_since_boot()
    if mpc_nans or self.lat_mpc.solution_status != 0:
      self.reset_mpc()
      self.x0[3] = measured_curvature
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning("Lateral mpc - nan: True")

    if self.lat_mpc.cost > 20000. or mpc_nans:
      self.solution_invalid_cnt += 1
    else:
      self.solution_invalid_cnt = 0

  def publish(self, sm, pm):
    plan_solution_valid = self.solution_invalid_cnt < 2
    plan_send = messaging.new_message('lateralPlan')
    plan_send.valid = sm.all_checks(service_list=['carState', 'controlsState', 'modelV2'])

    lateralPlan = plan_send.lateralPlan
    lateralPlan.modelMonoTime = sm.logMonoTime['modelV2']
    lateralPlan.laneWidth = float(self.LP.lane_width)
    lateralPlan.dPathPoints = self.y_pts.tolist()
    lateralPlan.psis = self.lat_mpc.x_sol[0:CONTROL_N, 2].tolist()
    lateralPlan.curvatures = self.lat_mpc.x_sol[0:CONTROL_N, 3].tolist()
    lateralPlan.curvatureRates = [float(x) for x in self.lat_mpc.u_sol[0:CONTROL_N - 1]] + [float(self.curv_rate_pts[CONTROL_N-1])]
    lateralPlan.lProb = float(self.LP.lll_prob)
    lateralPlan.rProb = float(self.LP.rll_prob)
    lateralPlan.dProb = float(self.LP.d_prob)

    lateralPlan.mpcSolutionValid = bool(plan_solution_valid)
    lateralPlan.solverExecutionTime = self.lat_mpc.solve_time

    lateralPlan.desire = self.DH.desire
    lateralPlan.useLaneLines = self.use_lanelines
    lateralPlan.laneChangeState = self.DH.lane_change_state
    lateralPlan.laneChangeDirection = self.DH.lane_change_direction

    pm.send('lateralPlan', plan_send)
