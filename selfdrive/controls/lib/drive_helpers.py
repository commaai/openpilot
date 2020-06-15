from common.numpy_fast import clip, interp
from selfdrive.config import Conversions as CV
from cereal import car

ButtonType = car.CarState.ButtonEvent.Type
button_pressed_cnt = 0
long_pressed = False
button_prev = ButtonType.unknown

# kph
V_CRUISE_MAX = 144
V_CRUISE_MIN = 8
V_CRUISE_DELTA_MI = 5 * CV.MPH_TO_KPH
V_CRUISE_DELTA_KM = 10
V_CRUISE_ENABLE_MIN = 40


class MPC_COST_LAT:
  PATH = 1.0
  LANE = 3.0
  HEADING = 1.0
  STEER_RATE = 1.0


class MPC_COST_LONG:
  TTC = 5.0
  DISTANCE = 0.1
  ACCELERATION = 10.0
  JERK = 20.0


def rate_limit(new_value, last_value, dw_step, up_step):
  return clip(new_value, last_value + dw_step, last_value + up_step)


def get_steer_max(CP, v_ego):
  return interp(v_ego, CP.steerMaxBP, CP.steerMaxV)


def update_v_cruise(v_cruise_kph, buttonEvents, enabled, metric):
  # handle button presses. TODO: this should be in state_control, but a decelCruise press
  # would have the effect of both enabling and changing speed is checked after the state transition
  global button_pressed_cnt, long_pressed, button_prev
  if enabled:
    if button_pressed_cnt:
      button_pressed_cnt += 1
    for b in buttonEvents:
      if b.type == ButtonType.accelCruise or b.type == ButtonType.decelCruise:
        if b.pressed and not button_pressed_cnt:
          button_pressed_cnt = 1
          button_prev = b.type
        elif not b.pressed and button_pressed_cnt:
          if not long_pressed and b.type == ButtonType.accelCruise:
            v_cruise_kph += 1 if metric else 1 * CV.MPH_TO_KPH
          elif not long_pressed and b.type == ButtonType.decelCruise:
            v_cruise_kph -= 1 if metric else 1 * CV.MPH_TO_KPH
          long_pressed = False
          button_pressed_cnt = 0
    if button_pressed_cnt > 25:
      long_pressed = True
      V_CRUISE_DELTA = V_CRUISE_DELTA_KM if metric else V_CRUISE_DELTA_MI
      if button_prev == ButtonType.accelCruise:
        v_cruise_kph += V_CRUISE_DELTA - v_cruise_kph % V_CRUISE_DELTA
      elif button_prev == ButtonType.decelCruise:
        v_cruise_kph -= V_CRUISE_DELTA - -v_cruise_kph % V_CRUISE_DELTA
      button_pressed_cnt %= 25
    v_cruise_kph = clip(v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

  return v_cruise_kph


def initialize_v_cruise(v_ego, buttonEvents, v_cruise_last):
  for b in buttonEvents:
    # 250kph or above probably means we never had a set speed
    if b.type == ButtonType.accelCruise and v_cruise_last < 250:
      return v_cruise_last

  return int(round(clip(v_ego * CV.MS_TO_KPH, V_CRUISE_ENABLE_MIN, V_CRUISE_MAX)))
