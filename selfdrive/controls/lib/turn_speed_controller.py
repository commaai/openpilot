import numpy as np
import time
from common.params import Params
from cereal import log
from common.realtime import sec_since_boot
from selfdrive.controls.lib.speed_smoother import speed_smoother

_LON_MPC_STEP = 0.2  # Time stemp of longitudinal control (5 Hz)

_MIN_ADAPTING_BRAKE_ACC = -1.5  # Minimum acceleration allowed when adapting to lower speed limit.
_MIN_ADAPTING_BRAKE_JERK = -1.0  # Minimum jerk allowed when adapting to lower speed limit.
_SPEED_OFFSET_TH = -3.0  # m/s Maximum offset between speed limit and current speed for adapting state.
_LIMIT_ADAPT_ACC = -1.0  # Ideal acceleration for the adapting (braking) phase when approaching speed limits.
_MIN_SPEED_LIMIT = 8.33  # m/s, Minimum speed limit to provide as solution.

_MAX_MAP_DATA_AGE = 10.0  # s Maximum time to hold to map data, then consider it invalid.

_DEBUG = False

TurnSpeedControlState = log.ControlsState.SpeedLimitControlState


def _debug(msg):
  if not _DEBUG:
    return
  print(msg)


def _description_for_state(turn_speed_control_state):
  if turn_speed_control_state == TurnSpeedControlState.inactive:
    return 'INACTIVE'
  if turn_speed_control_state == TurnSpeedControlState.adapting:
    return 'ADAPTING'
  if turn_speed_control_state == TurnSpeedControlState.active:
    return 'ACTIVE'


class TurnSpeedController():
  def __init__(self):
    self._params = Params()
    self._last_params_update = 0.0
    self._is_enabled = self._params.get("TurnSpeedControl", encoding='utf8') == "1"
    self._op_enabled = False
    self._active_jerk_limits = [0.0, 0.0]
    self._active_accel_limits = [0.0, 0.0]
    self._adapting_jerk_limits = [_MIN_ADAPTING_BRAKE_JERK, 1.0]
    self._v_ego = 0.0
    self._a_ego = 0.0

    self._v_offset = 0.0
    self._speed_limit = 0.0
    self._distance = 0.0
    self._turn_sign = 0
    self._state = TurnSpeedControlState.inactive

    self._next_speed_limit_prev = 0.
    self._adapting_cycles = 0
    self._adapting_time = 0.

    self.v_turn_limit = 0.0
    self.a_turn_limit = 0.0
    self.v_turn_limit_future = 0.0

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'Turn Speed Controller state: {_description_for_state(value)}')

      if value == TurnSpeedControlState.adapting:
        self._adapting_cycles = 0  # Reset adapting state cycle count when entereing state.
        # Adapting time must be  calculated at the moment we enter adapting state.
        self._adapting_time = self._v_offset / _LIMIT_ADAPT_ACC

    self._state = value

  @property
  def is_active(self):
    return self.state > TurnSpeedControlState.tempInactive

  @property
  def speed_limit(self):
    return max(self._speed_limit, _MIN_SPEED_LIMIT) if self._speed_limit > 0. else 0.

  @property
  def distance(self):
    return self._distance

  @property
  def turn_sign(self):
    return self._turn_sign

  def _get_limit_from_map_data(self, sm):
    """Provides the speed limit, distance and turn sign to it for turns based on map data.
    """
    # Ignore if no live map data
    sock = 'liveMapData'
    if sm.logMonoTime[sock] is None:
      _debug('TS: No map data for turn speed limit')
      return 0., 0., 0

    # Load limits from map_data
    map_data = sm[sock]
    speed_limit = 0.
    turn_sign = map_data.turnSpeedLimitSign if map_data.turnSpeedLimitValid else 0
    next_speed_limit = map_data.turnSpeedLimitAhead if map_data.turnSpeedLimitAheadValid else 0.
    next_turn_sign = map_data.turnSpeedLimitAheadSign if map_data.turnSpeedLimitAheadValid else 0

    # Calculate the age of the gps fix. Ignore if too old.
    gps_fix_age = time.time() - map_data.lastGpsTimestamp * 1e-3
    if gps_fix_age > _MAX_MAP_DATA_AGE:
      _debug(f'TS: Ignoring map data as is too old. Age: {gps_fix_age}')
      return 0., 0., 0

    # Ensure current speed limit is considered only if we are inside the section.
    if map_data.turnSpeedLimitValid and self._v_ego > 0.:
      speed_limit_end_time = (map_data.turnSpeedLimitEndDistance / self._v_ego) - gps_fix_age
      if speed_limit_end_time > 0.:
        speed_limit = map_data.turnSpeedLimit

    # When we have no ahead speed limit to consider or it is greater than current speed limit
    # or car has stopped, then provide current value and reset tracking.
    if next_speed_limit == 0. or self._v_ego <= 0. or (speed_limit > 0 and next_speed_limit > speed_limit):
      self._next_speed_limit_prev = 0.
      return speed_limit, 0., turn_sign

    # Calculate the distance to the next speed limit ahead corrected by gps_fix_age
    distance_since_fix = self._v_ego * gps_fix_age
    distance_to_limit_ahead = max(0., map_data.turnSpeedLimitAheadDistance - distance_since_fix)

    # When we have a next_speed_limit value that has not changed from a provided next speed limit value
    # in previous resolutions, we keep providing it along with the udpated distance to it.
    if next_speed_limit == self._next_speed_limit_prev:
      return next_speed_limit, distance_to_limit_ahead, next_turn_sign

    # Reset tracking
    self._next_speed_limit_prev = 0.

    # Calculated the time needed to adapt to the new limit and the corresponding distance.
    adapt_time = (max(next_speed_limit, _MIN_SPEED_LIMIT) - self._v_ego) / _LIMIT_ADAPT_ACC
    adapt_distance = self._v_ego * adapt_time + 0.5 * _LIMIT_ADAPT_ACC * adapt_time**2

    # When we detect we are close enough, we provide the next limit value and track it.
    if distance_to_limit_ahead <= adapt_distance:
      self._next_speed_limit_prev = next_speed_limit
      return next_speed_limit, distance_to_limit_ahead, next_turn_sign

    # Otherwise we just provide the calculated speed_limit
    return speed_limit, 0., turn_sign

  def _update_params(self):
    time = sec_since_boot()
    if time > self._last_params_update + 5.0:
      self._is_enabled = self._params.get("TurnSpeedControl", encoding='utf8') == "1"
      self._last_params_update = time

  def _update_calculations(self):
    # Update current velocity offset (error)
    self._v_offset = self.speed_limit - self._v_ego

  def _state_transition(self):
    # In any case, if op is disabled, or speed limit control is disabled
    # or the reported speed limit is 0, deactivate.
    if not self._op_enabled or not self._is_enabled or self.speed_limit == 0.:
      self.state = TurnSpeedControlState.inactive
      return

    # inactive
    if self.state == TurnSpeedControlState.inactive:
      # If the limit speed offset is negative (i.e. reduce speed) and lower than threshold
      # we go to adapting state to quickly reduce speed, otherwise we go directly to active
      if self._v_offset < _SPEED_OFFSET_TH:
        self.state = TurnSpeedControlState.adapting
      else:
        self.state = TurnSpeedControlState.active
    # adapting
    elif self.state == TurnSpeedControlState.adapting:
      self._adapting_cycles += 1
      # Go to active once the speed offset is over threshold.
      if self._v_offset >= _SPEED_OFFSET_TH:
        self.state = TurnSpeedControlState.active
    # active
    elif self.state == TurnSpeedControlState.active:
      # Go to adapting if the speed offset goes below threshold.
      if self._v_offset < _SPEED_OFFSET_TH:
        self.state = TurnSpeedControlState.adapting

  def _update_solution(self):
    # inactive
    if self.state == TurnSpeedControlState.inactive:
      # Preserve values
      self.v_turn_limit = self._v_ego
      self.a_turn_limit = self._a_ego
      self.v_turn_limit_future = self._v_ego
    # adapting
    elif self.state == TurnSpeedControlState.adapting:
      # Calculate to adapt speed on target time.
      adapting_time = max(self._adapting_time - self._adapting_cycles * _LON_MPC_STEP, 1.0)  # min adapt time 1 sec.
      a_target = (self.speed_limit - self._v_ego) / adapting_time
      # smooth out acceleration using jerk limits.
      j_limits = np.array(self._adapting_jerk_limits)
      a_limits = self._a_ego + j_limits * _LON_MPC_STEP
      a_target = max(min(a_target, a_limits[1]), a_limits[0])
      # calculate the solution values
      self.a_turn_limit = max(a_target, _MIN_ADAPTING_BRAKE_ACC)  # acceleration in next Longitudinal control step.
      self.v_turn_limit = self._v_ego + self.a_turn_limit * _LON_MPC_STEP  # speed in next Longitudinal control step.
      self.v_turn_limit_future = max(self._v_ego + self.a_turn_limit * 4., self.speed_limit)  # speed in 4 seconds.
    # active
    elif self.state == TurnSpeedControlState.active:
      # Calculate following same cruise logic in planner.py
      self.v_turn_limit, self.a_turn_limit = \
          speed_smoother(self._v_ego, self._a_ego, self.speed_limit, self._active_accel_limits[1],
                         self._active_accel_limits[0], self._active_jerk_limits[1], self._active_jerk_limits[0],
                         _LON_MPC_STEP)
      self.v_turn_limit = max(self.v_turn_limit, 0.)
      self.v_turn_limit_future = self.speed_limit

  def update(self, enabled, v_ego, a_ego, sm, accel_limits, jerk_limits):
    self._op_enabled = enabled
    self._v_ego = v_ego
    self._a_ego = a_ego
    self._active_accel_limits = accel_limits
    self._active_jerk_limits = jerk_limits

    # Get the speed limit from Map Data
    self._speed_limit, self._distance, self._turn_sign = self._get_limit_from_map_data(sm)

    self._update_params()
    self._update_calculations()
    self._state_transition()
    self._update_solution()

  def deactivate(self):
    self.state = TurnSpeedControlState.inactive
