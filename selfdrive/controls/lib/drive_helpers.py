import os
import signal
from cereal import car
from common.numpy_fast import clip
from selfdrive.config import Conversions as CV

# kph
V_CRUISE_MAX = 169
V_CRUISE_MIN = 7
V_CRUISE_DELTA = 7
V_CRUISE_ENABLE_MIN = 7

class MPC_COST_LAT:
  PATH = 1.0
  LANE = 3.0
  HEADING = 1.0
  STEER_RATE = 1.0


class MPC_COST_LONG:
  TTC = 5.0
  DISTANCE = 0.8
  ACCELERATION = 10.0
  JERK = 20.0


class EventTypes:
  ENABLE = 'enable'
  PRE_ENABLE = 'preEnable'
  NO_ENTRY = 'noEntry'
  WARNING = 'warning'
  USER_DISABLE = 'userDisable'
  SOFT_DISABLE = 'softDisable'
  IMMEDIATE_DISABLE = 'immediateDisable'
  PERMANENT = 'permanent'


def create_event(name, types):
  event = car.CarEvent.new_message()
  event.name = name
  for t in types:
    setattr(event, t, True)
  return event


def get_events(events, types):
  out = []
  for e in events:
    for t in types:
      if getattr(e, t):
        out.append(e.name)
  return out


def rate_limit(new_value, last_value, dw_step, up_step):
  return clip(new_value, last_value + dw_step, last_value + up_step)


def learn_angle_offset(lateral_control, v_ego, angle_offset, c_poly, c_prob, angle_steers, steer_override):
  # simple integral controller that learns how much steering offset to put to have the car going straight
  # while being in the middle of the lane
  min_offset = -5.  # deg
  max_offset =  5.  # deg
  alpha = 1./36000. # correct by 1 deg in 2 mins, at 30m/s, with 50cm of error, at 20Hz
  min_learn_speed = 1.

  # learn less at low speed or when turning
  slow_factor = 1. / (1. + 0.02 * abs(angle_steers) * v_ego)
  alpha_v = alpha * c_prob * (max(v_ego - min_learn_speed, 0.)) * slow_factor

  # only learn if lateral control is active and if driver is not overriding:
  if lateral_control and not steer_override:
    angle_offset += c_poly[3] * alpha_v
    angle_offset = clip(angle_offset, min_offset, max_offset)

  return angle_offset


def update_v_cruise(v_cruise_kph, buttonEvents, enabled):
  # handle button presses. TODO: this should be in state_control, but a decelCruise press
  # would have the effect of both enabling and changing speed is checked after the state transition
  for b in buttonEvents:
    if enabled and not b.pressed:
      if b.type == "accelCruise":
        v_cruise_kph += V_CRUISE_DELTA - (v_cruise_kph % V_CRUISE_DELTA)
      elif b.type == "decelCruise":
        v_cruise_kph -= V_CRUISE_DELTA - ((V_CRUISE_DELTA - v_cruise_kph) % V_CRUISE_DELTA)
      v_cruise_kph = clip(v_cruise_kph, V_CRUISE_MIN, V_CRUISE_MAX)

  return v_cruise_kph


def initialize_v_cruise(v_ego, buttonEvents, v_cruise_last):
  for b in buttonEvents:
    # 250kph or above probably means we never had a set speed
    if b.type == "accelCruise" and v_cruise_last < 250:
      return v_cruise_last

  return int(round(clip(v_ego * CV.MS_TO_KPH, V_CRUISE_ENABLE_MIN, V_CRUISE_MAX)))


def kill_defaultd():
  # defaultd is used to send can messages when controlsd is off to make car test easier
  if os.path.isfile("/tmp/defaultd_pid"):
    with open("/tmp/defaultd_pid") as f:
      ddpid = int(f.read())
    print("signalling defaultd with pid %d" % ddpid)
    os.kill(ddpid, signal.SIGUSR1)
