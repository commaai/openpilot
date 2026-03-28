#!/usr/bin/env python3
from __future__ import annotations

import json
import time

from cereal import messaging
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.tools.bodyteleop.autonomy.audio import play_wav
from openpilot.tools.bodyteleop.autonomy.config import AutonomyConfig
from openpilot.tools.bodyteleop.autonomy.controller import compute_axes
from openpilot.tools.bodyteleop.autonomy.state_machine import BehaviorInputs, BehaviorState, next_state


def _get_float(params: Params, key: str, default: float) -> float:
  value = params.get(key, encoding='utf8')
  if value is None:
    return default
  try:
    return float(value)
  except ValueError:
    return default


def estimate_attending(sm: messaging.SubMaster, now: float, cfg: AutonomyConfig, latched: dict[str, float]) -> bool:
  if not sm.updated['driverStateV2']:
    return True

  ds = sm['driverStateV2'].leftDriverData
  gaze_prob = float(ds.faceProb)
  eyes_open_prob = 0.5 * (float(ds.leftEyeProb) + float(ds.rightEyeProb))

  attending_now = (gaze_prob >= cfg.gaze_attend_threshold) and (eyes_open_prob >= cfg.eye_closed_threshold)
  if attending_now:
    latched['attending_since'] = now
  else:
    latched['inattentive_since'] = now

  attending_hold = now - latched['attending_since'] <= cfg.attention_hold_s
  inattentive_hold = now - latched['inattentive_since'] <= cfg.inattentive_hold_s

  if inattentive_hold and not attending_now:
    return False
  if attending_hold and attending_now:
    return True
  return attending_now


def get_runtime_inputs(params: Params, cfg: AutonomyConfig) -> tuple[bool, float, float, bool]:
  target_visible = params.get_bool("BodyAutonomyTargetVisible")
  obstacle_distance = _get_float(params, "BodyAutonomyObstacleDistance", 10.0)
  target_distance = _get_float(params, "BodyAutonomyTargetDistance", cfg.creep_target_distance_m)
  target_bearing = _get_float(params, "BodyAutonomyTargetBearingDeg", 0.0)

  obstacle_too_close = obstacle_distance < cfg.stop_obstacle_distance_m
  return target_visible, target_distance, target_bearing, obstacle_too_close


def publish_joystick(pm: messaging.PubMaster, forward_axis: float, turn_axis: float) -> None:
  msg = messaging.new_message('testJoystick')
  msg.testJoystick.axes = [float(forward_axis), float(turn_axis)]
  msg.testJoystick.buttons = [False]
  pm.send('testJoystick', msg)


def main() -> None:
  params = Params()
  cfg = AutonomyConfig()
  sm = messaging.SubMaster(['driverStateV2'])
  pm = messaging.PubMaster(['testJoystick', 'logMessage'])
  rk = Ratekeeper(cfg.loop_hz)

  state = BehaviorState.IDLE
  latched = {
    'attending_since': 0.0,
    'inattentive_since': 0.0,
  }
  acquire_sound_latched = False

  cloudlog.info("bodyautonomyd: started")

  while True:
    sm.update(0)
    now = time.monotonic()

    enabled = params.get_bool("BodyAutonomyEnabled")
    attending = estimate_attending(sm, now, cfg, latched)
    target_visible, target_distance, target_bearing, obstacle_too_close = get_runtime_inputs(params, cfg)

    inp = BehaviorInputs(enabled=enabled,
                         target_visible=target_visible,
                         attending=attending,
                         obstacle_too_close=obstacle_too_close)
    prev_state = state
    state = next_state(state, inp)

    if state == BehaviorState.ACQUIRE and not acquire_sound_latched:
      play_wav(cfg.acquire_sound_file)
      acquire_sound_latched = True
    elif state in (BehaviorState.IDLE, BehaviorState.LOST):
      acquire_sound_latched = False
      if prev_state == BehaviorState.ADVANCE:
        play_wav(cfg.lose_sound_file)

    forward_axis, turn_axis = compute_axes(state, target_bearing, target_distance, cfg)
    publish_joystick(pm, forward_axis, turn_axis)

    status = {
      "enabled": enabled,
      "state": state.value,
      "target_visible": target_visible,
      "attending": attending,
      "target_distance_m": target_distance,
      "target_bearing_deg": target_bearing,
      "obstacle_too_close": obstacle_too_close,
      "axes": [forward_axis, turn_axis],
    }
    params.put("BodyAutonomyStatus", json.dumps(status))

    rk.keep_time()


if __name__ == "__main__":
  main()
