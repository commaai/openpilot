import numpy as np
import time
from common.numpy_fast import interp
from enum import IntEnum
from cereal import log, car
from common.params import Params
from common.realtime import sec_since_boot
from selfdrive.controls.lib.drive_helpers import LIMIT_ADAPT_ACC, LIMIT_MIN_ACC, LIMIT_MAX_ACC, LIMIT_SPEED_OFFSET_TH, \
  LIMIT_MAX_MAP_DATA_AGE, CONTROL_N
from selfdrive.controls.lib.events import Events
from selfdrive.modeld.constants import T_IDXS


_PARAMS_UPDATE_PERIOD = 2.  # secs. Time between parameter updates.
_TEMP_INACTIVE_GUARD_PERIOD = 1.  # secs. Time to wait after activation before considering temp deactivation signal.

# Lookup table for speed limit percent offset depending on speed.
_LIMIT_PERC_OFFSET_V = [0.1, 0.05, 0.038]  # 55, 105, 135 km/h
_LIMIT_PERC_OFFSET_BP = [13.9, 27.8, 36.1]  # 50, 100, 130 km/h

SpeedLimitControlState = log.LongitudinalPlan.SpeedLimitControlState
EventName = car.CarEvent.EventName

_DEBUG = False


def _debug(msg):
  if not _DEBUG:
    return
  print(msg)


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
    if gps_fix_age > LIMIT_MAX_MAP_DATA_AGE:
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

    # Calculated the time needed to adapt to the new limit and the corresponding distance.
    adapt_time = (next_speed_limit - self._v_ego) / LIMIT_ADAPT_ACC
    adapt_distance = self._v_ego * adapt_time + 0.5 * LIMIT_ADAPT_ACC * adapt_time**2

    # When we detect we are close enough, we provide the next limit value and track it.
    if distance_to_speed_limit_ahead <= adapt_distance:
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
  def __init__(self):
    self._params = Params()
    self._resolver = SpeedLimitResolver()
    self._last_params_update = 0.0
    self._last_op_enabled_time = 0.0
    self._is_metric = self._params.get_bool("IsMetric")
    self._is_enabled = self._params.get_bool("SpeedLimitControl")
    self._offset_enabled = self._params.get_bool("SpeedLimitPercOffset")
    self._op_enabled = False
    self._op_enabled_prev = False
    self._v_ego = 0.
    self._a_ego = 0.
    self._v_offset = 0.
    self._v_cruise_setpoint = 0.
    self._v_cruise_setpoint_prev = 0.
    self._v_cruise_setpoint_changed = False
    self._speed_limit = 0.
    self._speed_limit_prev = 0.
    self._speed_limit_changed = False
    self._distance = 0.
    self._source = SpeedLimitResolver.Source.none
    self._state = SpeedLimitControlState.inactive
    self._state_prev = SpeedLimitControlState.inactive
    self._gas_pressed = False
    self._a_target = 0.

  @property
  def a_target(self):
    return self._a_target if self.is_active else self._a_ego

  @property
  def state(self):
    return self._state

  @state.setter
  def state(self, value):
    if value != self._state:
      _debug(f'Speed Limit Controller state: {_description_for_state(value)}')

      if value == SpeedLimitControlState.tempInactive:
        # Reset previous speed limit to current value as to prevent going out of tempInactive in
        # a single cycle when the speed limit changes at the same time the user has temporarily deactivate it.
        self._speed_limit_prev = self._speed_limit

    self._state = value

  @property
  def is_active(self):
    return self.state > SpeedLimitControlState.tempInactive

  @property
  def speed_limit_offseted(self):
    return self._speed_limit + self.speed_limit_offset

  @property
  def speed_limit_offset(self):
    if self._offset_enabled:
      return interp(self._speed_limit, _LIMIT_PERC_OFFSET_BP, _LIMIT_PERC_OFFSET_V) * self._speed_limit
    return 0.

  @property
  def speed_limit(self):
    return self._speed_limit

  @property
  def distance(self):
    return self._distance

  @property
  def source(self):
    return self._source

  def _update_params(self):
    time = sec_since_boot()
    if time > self._last_params_update + _PARAMS_UPDATE_PERIOD:
      self._is_enabled = self._params.get_bool("SpeedLimitControl")
      self._offset_enabled = self._params.get_bool("SpeedLimitPercOffset")
      _debug(f'Updated Speed limit params. enabled: {self._is_enabled}, with offset: {self._offset_enabled}')
      self._last_params_update = time

  def _update_calculations(self):
    # Update current velocity offset (error)
    self._v_offset = self.speed_limit_offseted - self._v_ego

    # Track the time op becomes active to prevent going to tempInactive right away after
    # op enabling since controlsd will change the cruise speed every time on enabling and this will
    # cause a temp inactive transition if the controller is updated before controlsd sets actual cruise
    # speed.
    if not self._op_enabled_prev and self._op_enabled:
      self._last_op_enabled_time = sec_since_boot()

    # Update change tracking variables
    self._speed_limit_changed = self._speed_limit != self._speed_limit_prev
    self._v_cruise_setpoint_changed = self._v_cruise_setpoint != self._v_cruise_setpoint_prev
    self._speed_limit_prev = self._speed_limit
    self._v_cruise_setpoint_prev = self._v_cruise_setpoint
    self._op_enabled_prev = self._op_enabled

  def _state_transition(self):
    self._state_prev = self._state

    # In any case, if op is disabled, or speed limit control is disabled
    # or the reported speed limit is 0 or gas is pressed, deactivate.
    if not self._op_enabled or not self._is_enabled or self._speed_limit == 0 or self._gas_pressed:
      self.state = SpeedLimitControlState.inactive
      return

    # In any case, we deactivate the speed limit controller temporarily if the user changes the cruise speed.
    # Ignore if a minimum amount of time has not passed since activation. This is to prevent temp inactivations
    # due to controlsd logic changing cruise setpoint when going active.
    if self._v_cruise_setpoint_changed and \
       sec_since_boot() > (self._last_op_enabled_time + _TEMP_INACTIVE_GUARD_PERIOD):
      self.state = SpeedLimitControlState.tempInactive
      return

    # inactive
    if self.state == SpeedLimitControlState.inactive:
      # If the limit speed offset is negative (i.e. reduce speed) and lower than threshold
      # we go to adapting state to quickly reduce speed, otherwise we go directly to active
      if self._v_offset < LIMIT_SPEED_OFFSET_TH:
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
      # Go to active once the speed offset is over threshold.
      if self._v_offset >= LIMIT_SPEED_OFFSET_TH:
        self.state = SpeedLimitControlState.active
    # active
    elif self.state == SpeedLimitControlState.active:
      # Go to adapting if the speed offset goes below threshold.
      if self._v_offset < LIMIT_SPEED_OFFSET_TH:
        self.state = SpeedLimitControlState.adapting

  def _update_solution(self):
    # inactive or tempInactive state
    if self.state <= SpeedLimitControlState.tempInactive:
      # Preserve current values
      a_target = self._a_ego
    # adapting
    elif self.state == SpeedLimitControlState.adapting:
      # When adapting we target to achieve the speed limit on the distance if not there yet,
      # otherwise try to keep the speed constant around the control time horizon.
      if self.distance > 0:
        a_target = (self.speed_limit_offseted**2 - self._v_ego**2) / (2. * self.distance)
      else:
        a_target = self._v_offset / T_IDXS[CONTROL_N]
    # active
    elif self.state == SpeedLimitControlState.active:
      # When active we are trying to keep the speed constant around the control time horizon.
      a_target = self._v_offset / T_IDXS[CONTROL_N]

    # Keep solution limited.
    self._a_target = np.clip(a_target, LIMIT_MIN_ACC, LIMIT_MAX_ACC)

  def _update_events(self, events):
    if not self.is_active:
      # no event while inactive
      return

    if self._state_prev <= SpeedLimitControlState.tempInactive:
      events.add(EventName.speedLimitActive)
    elif self._speed_limit_changed != 0:
      events.add(EventName.speedLimitValueChange)

  def update(self, enabled, v_ego, a_ego, sm, v_cruise_setpoint, events=Events()):
    self._op_enabled = enabled
    self._v_ego = v_ego
    self._a_ego = a_ego
    self._v_cruise_setpoint = v_cruise_setpoint
    self._gas_pressed = sm['carState'].gasPressed

    self._speed_limit, self._distance, self._source = self._resolver.resolve(v_ego, self.speed_limit, sm)

    self._update_params()
    self._update_calculations()
    self._state_transition()
    self._update_solution()
    self._update_events(events)
