#!/usr/bin/env python3
# carControl を jetracer の PWM に橋渡しする簡易デーモン
# panda/CAN を使わず、擬似的な pandaStates と carState を流しつつ PWM を出力する

import os
import time
from typing import Tuple

from cereal import messaging, car, log
from openpilot.common.realtime import Ratekeeper
from openpilot.common.swaglog import cloudlog
from openpilot.common.params import Params

from jetracer.nvidia_racecar import NvidiaRacecar


def _clip(x: float, lo: float, hi: float) -> float:
  return max(lo, min(hi, x))


def build_car_params() -> car.CarParams:
  """最小限の CarParams を組み立てて Params に書き込む。"""
  cp = car.CarParams.new_message()
  # 既存のinterfaceに確実にある指紋にしておく（例: HYUNDAI SONATA 2019）
  car_fp = os.getenv("RCD_CAR_FINGERPRINT", "LEXUS_RX_TSS2")
  cp.carFingerprint = car_fp
  cp.mass = 3.0
  cp.wheelbase = 0.26
  cp.centerToFront = cp.wheelbase * 0.5
  cp.steerRatio = 1.0
  cp.steerActuatorDelay = 0.0
  cp.openpilotLongitudinalControl = True

  safety = car.CarParams.SafetyConfig.new_message()
  safety.safetyModel = car.CarParams.SafetyModel.noOutput
  cp.safetyConfigs = [safety]

  return cp


def publish_car_params_once(pm: messaging.PubMaster, params: Params, cp_reader: car.CarParams, cp_bytes: bytes) -> None:
  params.put("CarParams", cp_bytes)
  params.put_nonblocking("CarParamsCache", cp_bytes)
  params.put_nonblocking("CarParamsPersistent", cp_bytes)

  msg = messaging.new_message("carParams")
  msg.valid = True
  msg.carParams = cp_reader
  pm.send("carParams", msg)


def make_panda_states_msg():
  """pandaStates を1要素だけ入れたメッセージを作る。"""
  msg = messaging.new_message("pandaStates", 1)
  ps = msg.pandaStates[0]
  ps.pandaType = log.PandaState.PandaType.redPanda
  ps.ignitionLine = True
  ps.ignitionCan = True
  ps.controlsAllowed = True
  ps.powerSaveEnabled = False
  ps.faultStatus = 0
  ps.safetyModel = car.CarParams.SafetyModel.noOutput
  ps.safetyParam = 0
  return msg


def make_car_state_msg(v_ego: float, steer_deg: float):
  msg = messaging.new_message("carState")
  msg.valid = True

  cs = msg.carState
  cs.vEgo = v_ego
  cs.vEgoRaw = v_ego
  cs.steeringAngleDeg = steer_deg
  cs.canValid = True
  cs.cruiseState.enabled = True
  cs.cruiseState.available = True
  cs.gearShifter = car.CarState.GearShifter.drive
  cs.steeringRateDeg = 0.0
  cs.yawRate = 0.0
  cs.standstill = False
  return msg


def map_actuators_to_pwm(actuators: car.CarControl.Actuators, steer_max_deg: float, accel_gain: float) -> Tuple[float, float]:
  """carControl のアクチュエータを -1〜1 に正規化して返す。"""
  steer_cmd = 0.0
  if steer_max_deg > 0:
    steer_cmd = _clip(actuators.steeringAngleDeg / steer_max_deg, -1.0, 1.0)

  # accel[m/s^2] を適当なゲインでスロットルにマップ（負値で後退とみなす）
  throttle_cmd = _clip(actuators.accel * accel_gain, -1.0, 1.0)
  return steer_cmd, throttle_cmd


def main():
  # 環境変数でパラメータ調整可能にする
  car_type = os.getenv("RCD_RACECAR_TYPE", "OPTION")
  steer_max_deg = float(os.getenv("RCD_STEER_MAX_DEG", "30.0"))
  accel_gain = float(os.getenv("RCD_ACCEL_GAIN", "0.1"))
  v_ego_stub = float(os.getenv("RCD_VEGO_MPS", "1.5"))
  rate_hz = float(os.getenv("RCD_RATE_HZ", "50.0"))

  cloudlog.info(f"rcd starting with type={car_type}, steer_max_deg={steer_max_deg}, accel_gain={accel_gain}, rate_hz={rate_hz}")

  params = Params()
  pm = messaging.PubMaster(["pandaStates", "carState", "carParams"])
  sm = messaging.SubMaster(["carControl"])

  racecar = NvidiaRacecar(type=car_type)

  cp = build_car_params()
  cp_bytes = cp.to_bytes()
  cp_reader = cp.as_reader()
  publish_car_params_once(pm, params, cp_reader, cp_bytes)
  params.put("PandaSignatures", b"fake_panda_signature")

  rk = Ratekeeper(rate_hz, print_delay_threshold=None)
  last_cp_pub = time.monotonic()

  while True:
    sm.update(0)

    # carControl が来ていれば PWM を更新
    if sm.all_valid(["carControl"]):
      cc = sm["carControl"]
      steer_cmd, throttle_cmd = map_actuators_to_pwm(cc.actuators, steer_max_deg, accel_gain)
      racecar.steering = steer_cmd
      racecar.throttle = throttle_cmd

    # pandaStates を毎ループ送る
    pm.send("pandaStates", make_panda_states_msg())

    # carState を擬似送信（定速・ゼロ舵角）
    pm.send("carState", make_car_state_msg(v_ego_stub, 0.0))

    # carParams はたまに再送（controlsd 初期化漏れを防ぐため）
    now = time.monotonic()
    if now - last_cp_pub > 5.0:
      publish_car_params_once(pm, params, cp_reader, cp_bytes)
      last_cp_pub = now

    rk.keep_time()


if __name__ == "__main__":
  main()
