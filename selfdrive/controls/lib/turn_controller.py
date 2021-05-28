import numpy as np
import math
from cereal import log
from common.numpy_fast import interp
from common.params import Params
from common.realtime import sec_since_boot
from selfdrive.config import Conversions as CV
from selfdrive.controls.lib.lane_planner import TRAJECTORY_SIZE


_LON_MPC_STEP = 0.2  # Time stemp of longitudinal control (5 Hz)
_MIN_V = 5.6  # Do not operate under 20km/h

_ENTERING_PRED_CURVATURE_TH = 0.003  # Predicitve curvature threshold to trigger entering turn state.
_ENTERING_PRED_LAT_ACC_TH = 1.0  # Predicted Lat Acc threshold to trigger entering turn state.
_ABORT_ENTERING_CURVATURE_TH = 0.0015  # Curvature threshold to abort entering state if road straightens.

_TURNING_CURVATURE_TH = 0.0022  # Curvature threshold to trigger turning turn state.
_LEAVING_CURVATURE_TH = 0.002  # Curvature threshold to trigger leaving turn state.
_FINISH_CURVATURE_TH = 0.0015  # Curvature threshold to trigger the end of turn cycle.

_LEAVING_ACC = 0.0  # Allowed acceleration when leaving the turn.

_EVAL_STEP = 5.  # evaluate curvature every 5mts
_EVAL_START = 20.  # start evaluating 0 mts ahead
_EVAL_LENGHT = 150.  # evaluate curvature for 150mts
_EVAL_RANGE = np.arange(_EVAL_START, _EVAL_LENGHT, _EVAL_STEP)

_MAX_JERK_ACC_INCREASE = 0.5  # Maximum jerk allowed when increasing acceleration.

# Lookup table for maximum lateral acceleration according
# to R079r4e regulation for M1 category vehicles.
_A_LAT_REG_MAX_V = [2., 2., 2., 2.]  # Currently all the same for all speed ranges
_A_LAT_REG_MAX_BP = [2.8, 16.7, 27.8, 36.1]  # 10, 60, 100, 130 km/h

# Lookup table for the minimum deceleration during the ENTERING state
# depending on the actual maximum absolute lateral acceleration predicted on the turn ahead.
_ENTERING_SMOOTH_DECEL_V = [-0.3, -1.]  # min decel value allowed on ENTERING state
_ENTERING_SMOOTH_DECEL_BP = [1., 3]  # absolute value of lat acc ahead

# Lookup table for the acceleration for the TURNING state
# depending on the current lateral acceleration of the vehicle.
_TURNING_ACC_V = [0.5, -0.2, -0.4]  # acc value
_TURNING_ACC_BP = [1., 2., 3.]  # absolute value of current lat acc

_MIN_LANE_PROB = 0.6  # Minimum lanes probability to allow curvature prediction based on lanes.

_DEBUG = False


def _debug(msg):
  if not _DEBUG:
    return
  print(msg)


TurnControllerState = log.ControlsState.TurnControllerState


def eval_curvature(poly, x_vals):
  """
  This function returns a vector with the curvature based on path defined by `poly`
  evaluated on distance vector `x_vals`
  """
  # https://en.wikipedia.org/wiki/Curvature#  Local_expressions
  def curvature(x):
    a = abs(2 * poly[1] + 6 * poly[0] * x) / (1 + (3 * poly[0] * x**2 + 2 * poly[1] * x + poly[2])**2)**(1.5)
    return a

  return np.vectorize(curvature)(x_vals)


def eval_lat_acc(v_ego, x_curv):
  """
  This function returns a vector with the lateral acceleration based
  for the provided speed `v_ego` evaluated over curvature vector `x_curv`
  """

  def lat_acc(curv):
    a = v_ego**2 * curv
    return a

  return np.vectorize(lat_acc)(x_curv)


def _description_for_state(turn_controller_state):
  if turn_controller_state == TurnControllerState.disabled:
    return 'DISABLED'
  if turn_controller_state == TurnControllerState.entering:
    return 'ENTERING'
  if turn_controller_state == TurnControllerState.turning:
    return 'TURNING'
  if turn_controller_state == TurnControllerState.leaving:
    return 'LEAVING'


class TurnController():
  def __init__(self, CP):
    self._params = Params()
    self._CP = CP
    self._op_enabled = False
    self._is_enabled = self._params.get_bool("TurnVisionControl")
    self._min_braking_acc = float(self._params.get("MaxDecelerationForTurns"))
    self._jerk_limits = [self._min_braking_acc, _MAX_JERK_ACC_INCREASE]
    self._last_params_update = 0.0
    self._v_cruise_setpoint = 0.0
    self._v_ego = 0.0
    self._state = TurnControllerState.disabled

    self._reset()

  @property
  def v_turn_future(self):
    return float(self._v_turn_future) if self.state != TurnControllerState.disabled else self._v_cruise_setpoint

  @property
  def state(self):
    return self._state

  @property
  def is_active(self):
    return self._state != TurnControllerState.disabled

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'TVC: TurnVisionController state: {_description_for_state(value)}')
      if value == TurnControllerState.disabled:
        self._reset()
    self._state = value

  def _reset(self):
    self._v_turn_future = 0.0
    self._current_curvature = 0.0
    self._max_pred_curvature = 0.0
    self._max_pred_lat_acc = 0.0
    self._v_target_distance = 200.0
    self._v_target = 0.0
    self._lat_acc_overshoot_ahead = False
    self._lat_planner_data = None
    self._model_data = None

    self.a_turn = 0.0
    self.v_turn = 0.0

  def _update_params(self):
    time = sec_since_boot()
    if time > self._last_params_update + 5.0:
      self._is_enabled = self._params.get_bool("TurnVisionControl")
      self._last_params_update = time

  def _update_calculations(self):
    # Get path polynomial aproximation for curvature estimation from model data.
    path_poly = None
    md = self._model_data

    # 1. When the probability of lanes is good enough, compute polynomial from lanes as they are way more stable
    # on current mode than drving path.
    if md is not None and len(md.laneLines) == 4 and len(md.laneLines[0].t) == TRAJECTORY_SIZE:
      ll_x = md.laneLines[1].x  # left and right ll x is the same
      lll_y = np.array(md.laneLines[1].y)
      rll_y = np.array(md.laneLines[2].y)
      l_prob = md.laneLineProbs[1]
      r_prob = md.laneLineProbs[2]
      lll_std = md.laneLineStds[1]
      rll_std = md.laneLineStds[2]

      # Reduce reliance on lanelines that are too far apart or will be in a few seconds
      width_pts = rll_y - lll_y
      prob_mods = []
      for t_check in [0.0, 1.5, 3.0]:
        width_at_t = interp(t_check * (self._v_ego + 7), ll_x, width_pts)
        prob_mods.append(interp(width_at_t, [4.0, 5.0], [1.0, 0.0]))
      mod = min(prob_mods)
      l_prob *= mod
      r_prob *= mod

      # Reduce reliance on uncertain lanelines
      l_std_mod = interp(lll_std, [.15, .3], [1.0, 0.0])
      r_std_mod = interp(rll_std, [.15, .3], [1.0, 0.0])
      l_prob *= l_std_mod
      r_prob *= r_std_mod

      # Find path from lanes as the average center lane only if min probability on both lanes is above threshold.
      if l_prob > _MIN_LANE_PROB and r_prob > _MIN_LANE_PROB:
        c_y = width_pts / 2 + lll_y
        path_poly = np.polyfit(ll_x, c_y, 3)
        _debug(f'TVC: Poly for curvature derived from lanes: {path_poly}')

    # 2. If not polynomial derived from lanes, then derive it from compensated driving path with lanes as
    # provided by `lateralPlanner`.
    if path_poly is None and self._lat_planner_data is not None and len(self._lat_planner_data.dPathWLinesX) > 0:
      path_poly = np.polyfit(self._lat_planner_data.dPathWLinesX, self._lat_planner_data.dPathWLinesY, 3)
      _debug(f'TVC: Poly for curvature derived from compensated driving path: {path_poly}')

    # 3. If no polynomial derived from lanes or driving path, then provide a straight line poly.
    if path_poly is None:
      path_poly = np.array([0., 0., 0., 0.])

    pred_curvatures = eval_curvature(path_poly, _EVAL_RANGE)
    self._max_pred_curvature = np.amax(pred_curvatures)
    self._max_pred_lat_acc = self._v_ego**2 * self._max_pred_curvature

    a_lat_reg_max = interp(self._v_ego, _A_LAT_REG_MAX_BP, _A_LAT_REG_MAX_V)
    max_curvature_for_vego = a_lat_reg_max / max(self._v_ego, 0.1)**2
    lat_acc_overshoot_idxs = np.nonzero(pred_curvatures >= max_curvature_for_vego)[0]
    self._lat_acc_overshoot_ahead = len(lat_acc_overshoot_idxs) > 0

    if self._lat_acc_overshoot_ahead:
      self._v_target_distance = max(lat_acc_overshoot_idxs[0] * _EVAL_STEP + _EVAL_START, _EVAL_STEP)
      self._v_target = min(math.sqrt(a_lat_reg_max / self._max_pred_curvature), self._v_cruise_setpoint)
      _debug(f'TVC: High Lat Acc ahead. Distance: {self._v_target_distance:.2f}, target v: {self._v_target:.2f}')

  def _state_transition(self):
    # In any case, if system is disabled or the feature is disabeld or min braking param has been
    # set to non negative value, disable.
    if not self._op_enabled or not self._is_enabled or self._min_braking_acc >= 0.0:
      self.state = TurnControllerState.disabled
      return

    # DISABLED
    if self.state == TurnControllerState.disabled:
      # Do not enter a turn control cycle if speed is low.
      if self._v_ego <= _MIN_V:
        pass
      # If substantial curvature ahead is detected, and a minimum lateral
      # acceleration is predicted, then move to Entering turn state.
      elif self._max_pred_curvature >= _ENTERING_PRED_CURVATURE_TH \
              and self._max_pred_lat_acc >= _ENTERING_PRED_LAT_ACC_TH:
        self.state = TurnControllerState.entering
    # ENTERING
    elif self.state == TurnControllerState.entering:
      # Transition to Turning if current curvature over threshold.
      if self._current_curvature >= _TURNING_CURVATURE_TH:
        self.state = TurnControllerState.turning
      # Abort if road straightens.
      elif self._max_pred_curvature < _ABORT_ENTERING_CURVATURE_TH:
        self.state = TurnControllerState.disabled
    # TURNING
    elif self.state == TurnControllerState.turning:
      # Transition to Leaving if current curvature under threshold.
      if self._current_curvature < _LEAVING_CURVATURE_TH:
        self.state = TurnControllerState.leaving
    # LEAVING
    elif self.state == TurnControllerState.leaving:
      # Transition back to Turning if current curvature over threshold.
      if self._current_curvature >= _TURNING_CURVATURE_TH:
        self.state = TurnControllerState.turning
      elif self._current_curvature < _FINISH_CURVATURE_TH:
        self.state = TurnControllerState.disabled

  def _update_solution(self):
    # Calculate target acceleration based on turn state.
    # DISABLED
    if self.state == TurnControllerState.disabled:
      a_target = self._a_ego
    # ENTERING
    elif self.state == TurnControllerState.entering:
      entering_smooth_decel = interp(self._max_pred_lat_acc, _ENTERING_SMOOTH_DECEL_BP, _ENTERING_SMOOTH_DECEL_V)
      _debug(f'TVC: Overshooting {self._lat_acc_overshoot_ahead}, _entering_smooth_decel {entering_smooth_decel:.2f}')
      if self._lat_acc_overshoot_ahead:
        a_target = min((self._v_target**2 - self._v_ego**2) / (2 * self._v_target_distance), entering_smooth_decel)
      else:
        a_target = entering_smooth_decel
    # TURNING
    elif self.state == TurnControllerState.turning:
      current_lat_acc = self._current_curvature * self._v_ego**2
      a_target = interp(current_lat_acc, _TURNING_ACC_BP, _TURNING_ACC_V)
    # LEAVING
    elif self.state == TurnControllerState.leaving:
      a_target = _LEAVING_ACC

    # smooth out acceleration using jerk limits.
    j_limits = np.array(self._jerk_limits)
    a_limits = self._a_ego + j_limits * _LON_MPC_STEP
    a_target = max(min(a_target, a_limits[1]), a_limits[0])

    # calculate solution values.
    self.a_turn = max(a_target, self._min_braking_acc)  # acceleration in next Longitudinal control step.
    self.v_turn = self._v_ego + self.a_turn * _LON_MPC_STEP  # speed in next Longitudinal control step.
    self._v_turn_future = self._v_ego + self.a_turn * 4.  # speed in 4 seconds.

  def update(self, enabled, v_ego, a_ego, v_cruise_setpoint, sm):
    self._op_enabled = enabled
    self._v_ego = v_ego
    self._a_ego = a_ego
    self._v_cruise_setpoint = v_cruise_setpoint
    self._current_curvature = abs(
        sm['carState'].steeringAngleDeg * CV.DEG_TO_RAD / (self._CP.steerRatio * self._CP.wheelbase))
    self._lat_planner_data = sm['lateralPlan'] if sm.valid.get('lateralPlan', False) else None
    self._model_data = sm['modelV2'] if sm.valid.get('modelV2', False) else None

    self._update_params()
    self._update_calculations()
    self._state_transition()
    self._update_solution()
