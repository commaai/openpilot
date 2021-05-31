import numpy as np
import time
from common.numpy_fast import interp
from enum import IntEnum
from cereal import log, car
from common.params import Params
from common.realtime import sec_since_boot
from selfdrive.controls.lib.speed_smoother import speed_smoother
from selfdrive.controls.lib.events import Events

_LON_MPC_STEP = 0.2  # Time stemp of longitudinal control (5 Hz)
_WAIT_TIME_LIMIT_RISE = 2.0  # Waiting time before raising the speed limit.

_MIN_ADAPTING_BRAKE_ACC = -1.5  # Minimum acceleration allowed when adapting to lower speed limit.
_MIN_ADAPTING_BRAKE_JERK = -1.0  # Minimum jerk allowed when adapting to lower speed limit.
_SPEED_OFFSET_TH = -3.0  # m/s Maximum offset between speed limit and current speed for adapting state.
_LIMIT_ADAPT_TIME = 5.0  # Ideal time (s) to adapt to lower speed limit. i.e. braking.

_MAX_SPEED_OFFSET_DELTA = 1.0  # m/s Maximum delta for speed limit changes.

SpeedLimitControlState = log.ControlsState.SpeedLimitControlState
EventName = car.CarEvent.EventName


def _description_for_state(speed_limit_control_state):
  if speed_limit_control_state == SpeedLimitControlState.inactive:
    return 'INACTIVE'
  if speed_limit_control_state == SpeedLimitControlState.tempInactive:
    return 'TEMP_INACTIVE'
  if speed_limit_control_state == SpeedLimitControlState.adapting:
    return 'ADAPTING'
  if speed_limit_control_state == SpeedLimitControlState.active:
    return 'ACTIVE'


class SpeedLimitResolver():
  class Source(IntEnum):
    none = 0
    car_state = 1
    map_data = 2

  class Policy(IntEnum):
    car_state_only = 0
    map_data_only = 1
    car_state_priority = 2
    map_data_priority = 3
    combined = 4

  def __init__(self, policy=Policy.map_data_priority):
    self._limit_solutions = {}  # Store for speed limit solutions from different sources
    self._distance_solutions = {}  # Store for distance to current speed limit start for different sources
    self._v_ego = 0.
    self._current_speed_limit = 0.
    self._policy = policy
    self._next_speed_limit_prev = 0.
    self.speed_limit = 0.
    self.distance = 0.
    self.source = SpeedLimitResolver.Source.none

  def resolve(self, v_ego, current_speed_limit, sm):
    self._v_ego = v_ego
    self._current_speed_limit = current_speed_limit
    self._sm = sm

    self._get_from_car_state()
    self._get_from_map_data()
    self._consolidate()

    return self.speed_limit, self.distance, self.source

  def _get_from_car_state(self):
    self._limit_solutions[SpeedLimitResolver.Source.car_state] = self._sm['carState'].cruiseState.speedLimit
    self._distance_solutions[SpeedLimitResolver.Source.car_state] = 0.

  def _get_from_map_data(self):
    # Ignore if no live map data
    sock = 'liveMapData'
    if self._sm.logMonoTime[sock] is None:
      self._limit_solutions[SpeedLimitResolver.Source.map_data] = 0.
      self._distance_solutions[SpeedLimitResolver.Source.map_data] = 0.
      _debug('SL: No map data for speed limit')
      return

    # Load limits from map_data
    map_data = self._sm[sock]
    speed_limit = map_data.speedLimit if map_data.speedLimitValid else 0.
    next_speed_limit = map_data.speedLimitAhead if map_data.speedLimitAheadValid else 0.

    # Calculate the age of the gps fix. Ignore if too old.
    gps_fix_age = time.time() - map_data.lastGpsTimestamp * 1e-3
    if gps_fix_age > _MAX_MAP_DATA_AGE:
      self._limit_solutions[SpeedLimitResolver.Source.map_data] = 0.
      self._distance_solutions[SpeedLimitResolver.Source.map_data] = 0.
      _debug(f'SL: Ignoring map data as is too old. Age: {gps_fix_age}')
      return

    # When we have no ahead speed limit to consider or it is greater than current speed limit
    # or car has stopped, then provide current value and reset tracking.
    if next_speed_limit == 0. or self._v_ego <= 0. or next_speed_limit > self._current_speed_limit:
      self._limit_solutions[SpeedLimitResolver.Source.map_data] = speed_limit
      self._distance_solutions[SpeedLimitResolver.Source.map_data] = 0.
      self._next_speed_limit_prev = 0.
      return

    # Calculate the actual distance to the speed limit ahead corrected by gps_fix_age
    distance_since_fix = self._v_ego * gps_fix_age
    distance_to_speed_limit_ahead = max(0., map_data.speedLimitAheadDistance - distance_since_fix)

    # When we have a next_speed_limit value that has not changed from a provided next speed limit value
    # in previous resolutions, we keep providing it.
    if next_speed_limit == self._next_speed_limit_prev:
      self._limit_solutions[SpeedLimitResolver.Source.map_data] = next_speed_limit
      self._distance_solutions[SpeedLimitResolver.Source.map_data] = distance_to_speed_limit_ahead
      return

    # Reset tracking
    self._next_speed_limit_prev = 0.

    # Calculate the time to the next speed limit and the adapt (braking)
    next_speed_limit_time = (map_data.speedLimitAheadDistance / self._v_ego) - gps_fix_age
    adapt_time = _LIMIT_ADAPT_TIME_PER_MS * (self._v_ego - next_speed_limit)

    # When we detect we are close enough, we provide the next limit value and track it.
    if next_speed_limit_time <= adapt_time:
      self._limit_solutions[SpeedLimitResolver.Source.map_data] = next_speed_limit
      self._distance_solutions[SpeedLimitResolver.Source.map_data] = distance_to_speed_limit_ahead
      self._next_speed_limit_prev = next_speed_limit
      return

    # Otherwise we just provide the map data speed limit.
    self.distance_to_map_speed_limit = 0.
    self._limit_solutions[SpeedLimitResolver.Source.map_data] = speed_limit
    self._distance_solutions[SpeedLimitResolver.Source.map_data] = 0.

  def _consolidate(self):
    limits = np.array([], dtype=float)
    distances = np.array([], dtype=float)
    sources = np.array([], dtype=int)

    if self._policy == SpeedLimitResolver.Policy.car_state_only or \
       self._policy == SpeedLimitResolver.Policy.car_state_priority or \
       self._policy == SpeedLimitResolver.Policy.combined:
      limits = np.append(limits, self._limit_solutions[SpeedLimitResolver.Source.car_state])
      distances = np.append(distances, self._distance_solutions[SpeedLimitResolver.Source.car_state])
      sources = np.append(sources, SpeedLimitResolver.Source.car_state.value)

    if self._policy == SpeedLimitResolver.Policy.map_data_only or \
       self._policy == SpeedLimitResolver.Policy.map_data_priority or \
       self._policy == SpeedLimitResolver.Policy.combined:
      limits = np.append(limits, self._limit_solutions[SpeedLimitResolver.Source.map_data])
      distances = np.append(distances, self._distance_solutions[SpeedLimitResolver.Source.map_data])
      sources = np.append(sources, SpeedLimitResolver.Source.map_data.value)

    if np.amax(limits) == 0.:
      if self._policy == SpeedLimitResolver.Policy.car_state_priority:
        limits = np.append(limits, self._limit_solutions[SpeedLimitResolver.Source.map_data])
        distances = np.append(distances, self._distance_solutions[SpeedLimitResolver.Source.map_data])
        sources = np.append(sources, SpeedLimitResolver.Source.map_data.value)

      elif self._policy == SpeedLimitResolver.Policy.map_data_priority:
        limits = np.append(limits, self._limit_solutions[SpeedLimitResolver.Source.car_state])
        distances = np.append(distances, self._distance_solutions[SpeedLimitResolver.Source.car_state])
        sources = np.append(sources, SpeedLimitResolver.Source.car_state.value)

    # Get all non-zero values and set the minimum if any, otherwise 0.
    mask = limits > 0.
    limits = limits[mask]
    distances = distances[mask]
    sources = sources[mask]

    if len(limits) > 0:
      min_idx = np.argmin(limits)
      self.speed_limit = limits[min_idx]
      self.distance = distances[min_idx]
      self.source = SpeedLimitResolver.Source(sources[min_idx])
    else:
      self.speed_limit = 0.
      self.distance = 0.
      self.source = SpeedLimitResolver.Source.none

    _debug(f'SL: *** Speed Limit set: {self.speed_limit}, distance: {self.distance}, source: {self.source}')


class SpeedLimitController():
  def __init__(self, CP):
    self._params = Params()
    self._last_params_update = 0.0
    self._is_metric = self._params.get_bool("IsMetric")
    self._is_enabled = self._params.get_bool("SpeedLimitControl")
    self._speed_limit_perc_offset = float(self._params.get("SpeedLimitPercOffset"))
    self._CP = CP
    self._op_enabled = False
    self._active_jerk_limits = [0.0, 0.0]
    self._active_accel_limits = [0.0, 0.0]
    self._adapting_jerk_limits = [_MIN_ADAPTING_BRAKE_JERK, 1.0]
    self._v_ego = 0.0
    self._a_ego = 0.0
    self._v_offset = 0.0
    self._v_cruise_setpoint = 0.0
    self._v_cruise_setpoint_prev = 0.0
    self._v_cruise_setpoint_changed = False
    self._speed_limit_set = 0.0
    self._speed_limit_set_prev = 0.0
    self._speed_limit_set_change = 0.0
    self._distance_set = 0.0
    self._speed_limit = 0.0
    self._speed_limit_prev = 0.0
    self._speed_limit_changed = False
    self._distance = 0.
    self._source = SpeedLimitResolver.Source.none
    self._last_speed_limit_set_change_ts = 0.0
    self._state = SpeedLimitControlState.inactive
    self._state_prev = SpeedLimitControlState.inactive
    self._adapting_cycles = 0

    self.v_limit = 0.0
    self.a_limit = 0.0
    self.v_limit_future = 0.0

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      print(f'Speed Limit Controller state: {_description_for_state(value)}')
      if value == SpeedLimitControlState.adapting:
        self._adapting_cycles = 0  # Reset adapting state cycle count when entereing state.
      elif value == SpeedLimitControlState.tempInactive:
        # Make sure speed limit is set to `set` value, this will have the effect
        # of canceling delayed increase limit, if pending.
        self._speed_limit = self._speed_limit_set
        self._speed_limit_prev = self._speed_limit
        self._distance = self._distance_set

    self._state = value

  @property
  def is_active(self):
    return self.state > SpeedLimitControlState.tempInactive

  @property
  def speed_limit(self):
    return self._speed_limit * (1.0 + self._speed_limit_perc_offset / 100.0)

  @property
  def distance(self):
    return self._distance

  @property
  def source(self):
    return self._source

  def _update_params(self):
    time = sec_since_boot()
    if time > self._last_params_update + 5.0:
      self._speed_limit_perc_offset = float(self._params.get("SpeedLimitPercOffset"))
      self._is_enabled = self._params.get_bool("SpeedLimitControl")
      print(f'Updated Speed limit params. enabled: {self._is_enabled}, \
              perc_offset: {self._speed_limit_perc_offset:.1f}')
      self._last_params_update = time

  def _update_calculations(self):
    # Track the time when speed limit set value changes.
    time = sec_since_boot()
    if self._speed_limit_set != self._speed_limit_set_prev:
      self._last_speed_limit_set_change_ts = time

    # Set distance to speed limit to 0 by default. i.e. active speed limit.
    # If the speed limit is ahead, we will update it below.
    self._distance = 0.

    # If not change on limit, we just update the distance to it.
    if self._speed_limit == self._speed_limit_set:
      self._distance = self._distance_set

    # Otherwise update speed limit from the set value.
    # - Imediate when changing from 0 or when updating to a lower speed limit or when increasing
    #   if delay increase is disabled.
    # - After a predefined period of time when increasing speed limit when delayed increase is enabled.
    else:
      if self._speed_limit == 0.0 or self._speed_limit_set < self._speed_limit or \
         not self._delay_increase or time > self._last_speed_limit_set_change_ts + _WAIT_TIME_LIMIT_RISE:
        self._speed_limit = self._speed_limit_set
        self._distance = self._distance_set

    # Update current velocity offset (error)
    self._v_offset = self.speed_limit_offseted - self._v_ego

    # Update change tracking variables
    self._speed_limit_changed = self._speed_limit != self._speed_limit_prev
    self._v_cruise_setpoint_changed = self._v_cruise_setpoint != self._v_cruise_setpoint_prev
    self._speed_limit_set_change = self._speed_limit_set - self._speed_limit_set_prev
    self._speed_limit_prev = self._speed_limit
    self._v_cruise_setpoint_prev = self._v_cruise_setpoint
    self._speed_limit_set_prev = self._speed_limit_set

  def _state_transition(self):
    self._state_prev = self._state
    # In any case, if op is disabled, or speed limit control is disabled
    # or the reported speed limit is 0, deactivate.
    if not self._op_enabled or not self._is_enabled or self._speed_limit == 0:
      self.state = SpeedLimitControlState.inactive
      return

    # inactive
    if self.state == SpeedLimitControlState.inactive:
      # If the limit speed offset is negative (i.e. reduce speed) and lower than threshold
      # we go to adapting state to quickly reduce speed, otherwise we go directly to active
      if self._v_offset < _SPEED_OFFSET_TH:
        self.state = SpeedLimitControlState.adapting
      else:
        self.state = SpeedLimitControlState.active
    # tempInactive
    elif self.state == SpeedLimitControlState.tempInactive:
      # if speed limit changes, transition to inactive,
      # proper active state will be set on next iteration.
      if self._speed_limit_changed:
        self.state = SpeedLimitControlState.inactive
    # adapting
    elif self.state == SpeedLimitControlState.adapting:
      self._adapting_cycles += 1
      # If user changes the cruise speed, deactivate temporarely
      if self._v_cruise_setpoint_changed:
        self.state = SpeedLimitControlState.tempInactive
      # Go to active once the speed offset is over threshold.
      elif self._v_offset >= _SPEED_OFFSET_TH:
        self.state = SpeedLimitControlState.active
    # active
    elif self.state == SpeedLimitControlState.active:
      # If user changes the cruise speed, deactivate temporarely
      if self._v_cruise_setpoint_changed:
        self.state = SpeedLimitControlState.tempInactive
      # Go to adapting if the speed offset goes below threshold.
      elif self._v_offset < _SPEED_OFFSET_TH:
        self.state = SpeedLimitControlState.adapting

  def _update_solution(self):
    # inactive
    if self.state == SpeedLimitControlState.inactive:
      # Preserve values
      self.v_limit = self._v_ego
      self.a_limit = self._a_ego
      self.v_limit_future = self._v_ego
    # adapting
    elif self.state == SpeedLimitControlState.adapting:
      # Calculate to adapt speed on target time.
      adapting_time = max(_LIMIT_ADAPT_TIME - self._adapting_cycles * _LON_MPC_STEP, 1.0)  # min adapt time 1 sec.
      a_target = (self.speed_limit - self._v_ego) / adapting_time
      # smooth out acceleration using jerk limits.
      j_limits = np.array(self._adapting_jerk_limits)
      a_limits = self._a_ego + j_limits * _LON_MPC_STEP
      a_target = max(min(a_target, a_limits[1]), a_limits[0])
      # calculate the solution values
      self.a_limit = max(a_target, _MIN_ADAPTING_BRAKE_ACC)  # acceleration in next Longitudinal control step.
      self.v_limit = self._v_ego + self.a_limit * _LON_MPC_STEP  # speed in next Longitudinal control step.
      self.v_limit_future = max(self._v_ego + self.a_limit * 4., self.speed_limit_offseted)  # speed in 4 seconds.
    # active
    elif self.state == SpeedLimitControlState.active:
      # Calculate following same cruise logic in longitudinal_planner.py
      self.v_limit, self.a_limit = speed_smoother(self._v_ego, self._a_ego, self.speed_limit_offseted,
                                                  self._active_accel_limits[1], self._active_accel_limits[0],
                                                  self._active_jerk_limits[1], self._active_jerk_limits[0],
                                                  _LON_MPC_STEP)
      self.v_limit = max(self.v_limit, 0.)
      self.v_limit_future = self._speed_limit

  def _update_events(self, events):
    if not self.is_active:
      # no event while inactive or deactivating
      return

    if self._state_prev <= SpeedLimitControlState.tempInactive:
      events.add(EventName.speedLimitActive)
    elif self._speed_limit_set_change > 0:
      events.add(EventName.speedLimitIncrease)
    elif self._speed_limit_set_change < 0:
      events.add(EventName.speedLimitDecrease)

  def update(self, enabled, v_ego, a_ego, CS, v_cruise_setpoint, accel_limits, jerk_limits, events=Events()):
    self._op_enabled = enabled
    self._v_ego = v_ego
    self._a_ego = a_ego

    self._speed_limit_set, self._distance_set, self._source = self._resolver.resolve(v_ego, self.speed_limit, sm)
    self._v_cruise_setpoint = v_cruise_setpoint
    self._active_accel_limits = accel_limits
    self._active_jerk_limits = jerk_limits

    self._update_params()
    self._update_calculations()
    self._state_transition()
    self._update_solution()
    self._update_events(events)

  def deactivate(self):
    self.state = SpeedLimitControlState.inactive
