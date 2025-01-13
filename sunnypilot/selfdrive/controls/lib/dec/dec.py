# The MIT License
#
# Copyright (c) 2019-, Rick Lan, dragonpilot community, and a number of other of contributors.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
#
# Version = 2024-7-11

import numpy as np

from cereal import messaging
from openpilot.common.numpy_fast import interp
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL

# d-e2e, from modeldata.h
TRAJECTORY_SIZE = 33

LEAD_WINDOW_SIZE = 4
LEAD_PROB = 0.6

SLOW_DOWN_WINDOW_SIZE = 4
SLOW_DOWN_PROB = 0.6

SLOW_DOWN_BP = [0., 10., 20., 30., 40., 50., 55., 60.]
SLOW_DOWN_DIST = [25., 38., 55., 75., 95., 115., 130., 150.]

SLOWNESS_WINDOW_SIZE = 12
SLOWNESS_PROB = 0.5
SLOWNESS_CRUISE_OFFSET = 1.05

DANGEROUS_TTC_WINDOW_SIZE = 3
DANGEROUS_TTC = 2.3

HIGHWAY_CRUISE_KPH = 70

STOP_AND_GO_FRAME = 60

SET_MODE_TIMEOUT = 10

MPC_FCW_WINDOW_SIZE = 10
MPC_FCW_PROB = 0.5

V_ACC_MIN = 9.72


class SNG_State:
  off = 0
  stopped = 1
  going = 2


class GenericMovingAverageCalculator:
  def __init__(self, window_size):
    self.window_size = window_size
    self.data = []
    self.total = 0

  def add_data(self, value: float) -> None:
    if len(self.data) == self.window_size:
      self.total -= self.data.pop(0)
    self.data.append(value)
    self.total += value

  def get_moving_average(self) -> float | None:
    return None if len(self.data) == 0 else self.total / len(self.data)

  def reset_data(self) -> None:
    self.data = []
    self.total = 0


class WeightedMovingAverageCalculator:
  def __init__(self, window_size):
    self.window_size = window_size
    self.data = []
    self.weights = np.linspace(1, 3, window_size)  # Linear weights, adjust as needed

  def add_data(self, value: float) -> None:
    if len(self.data) == self.window_size:
      self.data.pop(0)
    self.data.append(value)

  def get_weighted_average(self) -> float | None:
    if len(self.data) == 0:
      return None
    weighted_sum: float = float(np.dot(self.data, self.weights[-len(self.data):]))
    weight_total: float = float(np.sum(self.weights[-len(self.data):]))
    return weighted_sum / weight_total

  def reset_data(self) -> None:
    self.data = []


class DynamicExperimentalController:
  def __init__(self, params=None):
    self._params = params or Params()
    self._is_enabled: bool = self._params.get_bool("DynamicExperimentalControl")
    self._mode: str = 'acc'
    self._mode_prev: str = 'acc'
    self._mode_changed: bool = False
    self._frame: int = 0

    # Use weighted moving average for filtering leads
    self._lead_gmac = WeightedMovingAverageCalculator(window_size=LEAD_WINDOW_SIZE)
    self._has_lead_filtered = False
    self._has_lead_filtered_prev = False

    self._slow_down_gmac = WeightedMovingAverageCalculator(window_size=SLOW_DOWN_WINDOW_SIZE)
    self._has_slow_down = False

    self._has_blinkers = False

    self._slowness_gmac = WeightedMovingAverageCalculator(window_size=SLOWNESS_WINDOW_SIZE)
    self._has_slowness = False

    self._has_nav_instruction = False

    self._dangerous_ttc_gmac = WeightedMovingAverageCalculator(window_size=DANGEROUS_TTC_WINDOW_SIZE)
    self._has_dangerous_ttc = False

    self._v_ego_kph = 0.
    self._v_cruise_kph = 0.

    self._has_lead = False

    self._has_standstill = False
    self._has_standstill_prev = False

    self._sng_transit_frame = 0
    self._sng_state = SNG_State.off

    self._mpc_fcw_gmac = WeightedMovingAverageCalculator(window_size=MPC_FCW_WINDOW_SIZE)
    self._has_mpc_fcw = False
    self._mpc_fcw_crash_cnt = 0

    self._set_mode_timeout = 0

  @staticmethod
  def _anomaly_detection(recent_data: list[float], threshold: float = 2.0, context_check: bool = True) -> bool:
    """
    Basic anomaly detection using standard deviation.
    """
    if len(recent_data) < 5:
      return False
    mean: float = float(np.mean(recent_data))
    std_dev: float = float(np.std(recent_data))
    anomaly: bool = bool(recent_data[-1] > mean + threshold * std_dev)

    # Context check to ensure repeated anomaly
    if context_check:
      return bool(np.count_nonzero(np.array(recent_data) > mean + threshold * std_dev) > 1)
    return anomaly

  def _adaptive_slowdown_threshold(self) -> float:
    """
    Adapts the slow-down threshold based on vehicle speed and recent behavior.
    """
    adaptive_threshold: float = float(
      interp(self._v_ego_kph, SLOW_DOWN_BP, SLOW_DOWN_DIST) * (1.0 + 0.05 * np.log(1 + len(self._slow_down_gmac.data)))
    )
    return adaptive_threshold

  def _smoothed_lead_detection(self, lead_prob: float, smoothing_factor: float = 0.2) -> bool:
    """
    Smoothing the lead detection to avoid erratic behavior.
    """
    self._has_lead_filtered = (1 - smoothing_factor) * self._has_lead_filtered + smoothing_factor * lead_prob
    return bool(self._has_lead_filtered > LEAD_PROB)

  def _adaptive_lead_prob_threshold(self) -> float:
    """
    Adapts lead probability threshold based on driving conditions.
    """
    if self._v_ego_kph > HIGHWAY_CRUISE_KPH:
      return float(LEAD_PROB + 0.1)  # Increase the threshold on highways
    return float(LEAD_PROB)

  def _update(self, sm: messaging.SubMaster) -> None:
    car_state = sm['carState']
    lead_one = sm['radarState'].leadOne
    md = sm['modelV2']

    self._v_ego_kph = car_state.vEgo * 3.6
    self._v_cruise_kph = car_state.vCruise
    self._has_lead = lead_one.status
    self._has_standstill = car_state.standstill

    # fcw detection
    self._mpc_fcw_gmac.add_data(self._mpc_fcw_crash_cnt > 0)
    self._has_mpc_fcw = self._mpc_fcw_gmac.get_weighted_average() > MPC_FCW_PROB

    # nav enable detection
    # self._has_nav_instruction = md.navEnabledDEPRECATED and maneuver_distance / max(car_state.vEgo, 1) < 13

    # lead detection with smoothing
    self._lead_gmac.add_data(lead_one.status)
    #self._has_lead_filtered = self._lead_gmac.get_weighted_average() > LEAD_PROB
    lead_prob = self._lead_gmac.get_weighted_average() or 0
    self._has_lead_filtered = self._smoothed_lead_detection(lead_prob)

    # adaptive slow down detection
    adaptive_threshold = self._adaptive_slowdown_threshold()
    slow_down_trigger = len(md.orientation.x) == len(md.position.x) == TRAJECTORY_SIZE and md.position.x[TRAJECTORY_SIZE - 1] < adaptive_threshold
    self._slow_down_gmac.add_data(slow_down_trigger)
    self._has_slow_down = self._slow_down_gmac.get_weighted_average() > SLOW_DOWN_PROB

    # anomaly detection for slow down events
    if self._anomaly_detection(self._slow_down_gmac.data):
      # Handle anomaly: potentially log it, adjust behavior, or issue a warning
      self._has_slow_down = False  # Reset slow down if anomaly detected

    # blinker detection
    self._has_blinkers = car_state.leftBlinker or car_state.rightBlinker

    # sng detection
    if self._has_standstill:
      self._sng_state = SNG_State.stopped
      self._sng_transit_frame = 0
    else:
      if self._sng_transit_frame == 0:
        if self._sng_state == SNG_State.stopped:
          self._sng_state = SNG_State.going
          self._sng_transit_frame = STOP_AND_GO_FRAME
        elif self._sng_state == SNG_State.going:
          self._sng_state = SNG_State.off
      elif self._sng_transit_frame > 0:
        self._sng_transit_frame -= 1

    # slowness detection
    if not self._has_standstill:
      self._slowness_gmac.add_data(self._v_ego_kph <= (self._v_cruise_kph * SLOWNESS_CRUISE_OFFSET))
      self._has_slowness = self._slowness_gmac.get_weighted_average() > SLOWNESS_PROB

    # dangerous TTC detection
    if not self._has_lead_filtered and self._has_lead_filtered_prev:
      self._dangerous_ttc_gmac.reset_data()
      self._has_dangerous_ttc = False

    if self._has_lead and car_state.vEgo >= 0.01:
      self._dangerous_ttc_gmac.add_data(lead_one.dRel / car_state.vEgo)

    self._has_dangerous_ttc = self._dangerous_ttc_gmac.get_weighted_average() is not None and self._dangerous_ttc_gmac.get_weighted_average() <= DANGEROUS_TTC

    # keep prev values
    self._has_standstill_prev = self._has_standstill
    self._has_lead_filtered_prev = self._has_lead_filtered

  def _radarless_mode(self) -> None:
    # when mpc fcw crash prob is high
    # use blended to slow down quickly
    if self._has_mpc_fcw:
      self._set_mode('blended')
      return

    # Nav enabled and distance to upcoming turning is 300 or below
    # if self._has_nav_instruction:
    #  self._set_mode('blended')
    #  return

    # when blinker is on and speed is driving below V_ACC_MIN: blended
    # we don't want it to switch mode at higher speed, blended may trigger hard brake
    # if self._has_blinkers and self._v_ego_kph < V_ACC_MIN:
    #  self._set_mode('blended')
    #  return

    # when at highway cruise and SNG: blended
    # ensuring blended mode is used because acc is bad at catching SNG lead car
    # especially those who accel very fast and then brake very hard.
    # if self._sng_state == SNG_State.going and self._v_cruise_kph >= V_ACC_MIN:
    #  self._set_mode('blended')
    #  return

    # when standstill: blended
    # in case of lead car suddenly move away under traffic light, acc mode won't brake at traffic light.
    if self._has_standstill:
      self._set_mode('blended')
      return

    # when detecting slow down scenario: blended
    # e.g. traffic light, curve, stop sign etc.
    if self._has_slow_down:
      self._set_mode('blended')
      return

    # when detecting lead slow down: blended
    # use blended for higher braking capability
    if self._has_dangerous_ttc:
      self._set_mode('blended')
      return

    # car driving at speed lower than set speed: acc
    if self._has_slowness:
      self._set_mode('acc')
      return

    self._set_mode('acc')

  def _radar_mode(self) -> None:
    # when mpc fcw crash prob is high
    # use blended to slow down quickly
    if self._has_mpc_fcw:
      self._set_mode('blended')
      return

    # If there is a filtered lead, the vehicle is not in standstill, and the lead vehicle's yRel meets the condition,
    if self._has_lead_filtered and not self._has_standstill:
      self._set_mode('acc')
      return

    # when blinker is on and speed is driving below V_ACC_MIN: blended
    # we don't want it to switch mode at higher speed, blended may trigger hard brake
    # if self._has_blinkers and self._v_ego_kph < V_ACC_MIN:
    #  self._set_mode('blended')
    #  return

    # when standstill: blended
    # in case of lead car suddenly move away under traffic light, acc mode won't brake at traffic light.
    if self._has_standstill:
      self._set_mode('blended')
      return

    # when detecting slow down scenario: blended
    # e.g. traffic light, curve, stop sign etc.
    if self._has_slow_down:
      self._set_mode('blended')
      return

    # car driving at speed lower than set speed: acc
    if self._has_slowness:
      self._set_mode('acc')
      return

    # Nav enabled and distance to upcoming turning is 300 or below
    # if self._has_nav_instruction:
    #  self._set_mode('blended')
    #  return

    self._set_mode('acc')

  def get_mpc_mode(self) -> str:
    return str(self._mode)

  def has_changed(self) -> bool:
    return bool(self._mode_changed)

  def set_enabled(self, enabled: bool) -> None:
    self._is_enabled = enabled

  def is_enabled(self) -> bool:
    return self._is_enabled

  def set_mpc_fcw_crash_cnt(self, crash_cnt: float) -> None:
    self._mpc_fcw_crash_cnt = crash_cnt

  def _set_mode(self, mode: str) -> None:
    if self._set_mode_timeout == 0:
      self._mode = mode
      if mode == 'blended':
        self._set_mode_timeout = SET_MODE_TIMEOUT

    if self._set_mode_timeout > 0:
      self._set_mode_timeout -= 1

  def _read_params(self) -> None:
    if self._frame % int(1. / DT_MDL) == 0:
      self._is_enabled = self._params.get_bool("DynamicExperimentalControl")

  def update(self, radar_unavailable: bool, sm: messaging.SubMaster) -> None:
    self._read_params()

    if self._is_enabled:
      self._update(sm)

      if radar_unavailable:
        self._radarless_mode()
      else:
        self._radar_mode()

    self._mode_changed = self._mode != self._mode_prev
    self._mode_prev = self._mode

    self._frame += 1
