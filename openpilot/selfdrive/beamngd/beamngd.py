#!/usr/bin/env python3
import time
import traceback
from dataclasses import dataclass

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from opendbc.car.honda.values import HondaSafetyFlags
from openpilot.common.params import Params
from openpilot.common.realtime import Ratekeeper
from openpilot.cereal import messaging
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp


@dataclass
class Vector3:
  x: float = 0.0
  y: float = 0.0
  z: float = 0.0


@dataclass
class SimulatorVehicleState:
  valid: bool = True
  ignition: bool = True
  speed: float = 0.0 # m/s
  steering_angle: float = 0.0 # deg
  user_gas: float = 0.0
  user_brake: float = 0.0
  user_torque: float = 0.0
  cruise_button: int = 0
  left_blinker: bool = False
  right_blinker: bool = False
  is_engaged: bool = False


class BeamNGSimulatedCar:
  packer = CANPacker("honda_bosch_radarless_generated")

  def __init__(self):
    self.pm = messaging.PubMaster(['can', 'pandaStates'])
    self.sm = messaging.SubMaster(['carControl', 'controlsState', 'carParams', 'selfdriveState'])
    self.idx = 0
    self.params = Params()
    self.obd_multiplexing = False

  def send_can_messages(self, v_state: SimulatorVehicleState):
    if not v_state.valid:
      return

    msg = []
    speed_kph = v_state.speed * 3.6

    # powertrain (bus 0)
    msg.append(self.packer.make_can_msg("ENGINE_DATA", 0, {"XMISSION_SPEED": speed_kph}))
    msg.append(self.packer.make_can_msg("POWERTRAIN_DATA", 0, {
      "ACC_STATUS": int(v_state.is_engaged),
      "PEDAL_GAS": v_state.user_gas,
      "BRAKE_PRESSED": v_state.user_brake > 0,
      "BRAKE_SWITCH": v_state.user_brake > 0
    }))
    msg.append(self.packer.make_can_msg("WHEEL_SPEEDS", 0, {
      "WHEEL_SPEED_FL": speed_kph,
      "WHEEL_SPEED_FR": speed_kph,
      "WHEEL_SPEED_RL": speed_kph,
      "WHEEL_SPEED_RR": speed_kph
    }))
    msg.append(self.packer.make_can_msg("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": v_state.cruise_button}))
    msg.append(self.packer.make_can_msg("GEARBOX_AUTO", 0, {"GEAR_SHIFTER": 4})) # Drive
    msg.append(self.packer.make_can_msg("GAS_PEDAL_2", 0, {}))
    msg.append(self.packer.make_can_msg("SEATBELT_STATUS", 0, {"SEATBELT_DRIVER_LATCHED": 1}))
    msg.append(self.packer.make_can_msg("STEER_STATUS", 0, {"STEER_TORQUE_SENSOR": v_state.user_torque}))
    msg.append(self.packer.make_can_msg("STEERING_SENSORS", 0, {"STEER_ANGLE": v_state.steering_angle}))
    msg.append(self.packer.make_can_msg("VSA_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("STANDSTILL", 0, {"WHEELS_MOVING": 1 if v_state.speed >= 1.0 else 0}))
    msg.append(self.packer.make_can_msg("STEER_MOTOR_TORQUE", 0, {}))
    msg.append(self.packer.make_can_msg("EPB_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("DOORS_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE_FAULT_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("SCM_FEEDBACK", 0, {
      "MAIN_ON": 1,
      "LEFT_BLINKER": v_state.left_blinker,
      "RIGHT_BLINKER": v_state.right_blinker
    }))
    msg.append(self.packer.make_can_msg("CAR_SPEED", 0, {}))

    # cameras
    msg.append(self.packer.make_can_msg("STEERING_CONTROL", 2, {}))
    msg.append(self.packer.make_can_msg("ACC_HUD", 2, {}))
    msg.append(self.packer.make_can_msg("LKAS_HUD", 2, {}))

    self.pm.send('can', can_list_to_can_capnp(msg))

  def send_panda_state(self, v_state: SimulatorVehicleState):
    self.sm.update(0)

    if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
      self.obd_multiplexing = not self.obd_multiplexing
      self.params.put_bool("ObdMultiplexingChanged", True, block=True)

    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': v_state.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaBosch',
      'alternativeExperience': self.sm["carParams"].alternativeExperience,
      'safetyParam': HondaSafetyFlags.RADARLESS.value | HondaSafetyFlags.BOSCH_LONG.value,
    }
    self.pm.send('pandaStates', dat)

  def update(self, v_state: SimulatorVehicleState):
    try:
      self.send_can_messages(v_state)

      if self.idx % 50 == 0:  # 2 Hz
        self.send_panda_state(v_state)

      self.idx += 1
    except Exception:
      traceback.print_exc()
      raise


def main():
  sim_car = BeamNGSimulatedCar()

  # publisher for the fake sensors
  pm = messaging.PubMaster([
    'accelerometer',
    'gyroscope',
    'gpsLocationExternal',
    'driverStateV2',
    'driverMonitoringState',
    'peripheralState'
  ])

  # carstate
  v_state = SimulatorVehicleState()

  rk = Ratekeeper(100, print_delay_threshold=None)
  print("[beamngd] Dummy car, CAN publisher, and fake hardware sensors running...")

  while True:
    # imu
    dat = messaging.new_message('accelerometer', valid=True)
    dat.accelerometer.timestamp = dat.logMonoTime
    dat.accelerometer.init('acceleration')
    dat.accelerometer.acceleration.v = [0.0, 0.0, 9.81]
    pm.send('accelerometer', dat)

    dat = messaging.new_message('gyroscope', valid=True)
    dat.gyroscope.timestamp = dat.logMonoTime
    dat.gyroscope.init('gyroUncalibrated')
    dat.gyroscope.gyroUncalibrated.v = [0.0, 0.0, 0.0]
    pm.send('gyroscope', dat)

    # gps
    if rk.frame % 10 == 0:
      dat = messaging.new_message('gpsLocationExternal', valid=True)
      dat.gpsLocationExternal = {
        "unixTimestampMillis": int(time.time() * 1000),
        "flags": 1,
        "horizontalAccuracy": 1.0,
        "verticalAccuracy": 1.0,
        "speedAccuracy": 0.1,
        "bearingAccuracyDeg": 0.1,
        "vNED": [0.0, 0.0, 0.0],
        "bearingDeg": 0.0,
        "latitude": 32.7,
        "longitude": -117.1,
        "altitude": 0.0,
        "speed": 0.0,
        "source": 1,
      }
      pm.send('gpsLocationExternal', dat)

    # DM 2hz
    # DOING THIS IN REAL OPENPILOT IS DANGEROUS AND AGAINST COMMA REQUIREMENTS!
    # BEAMPILOT IS SIMULATION USE ONLY, DO NOT USE IN REAL LIFE (not that it can. but yeah)
    if rk.frame % 50 == 0:
      dat = messaging.new_message('driverStateV2')
      dat.driverStateV2.leftDriverData.faceOrientation = [0., 0., 0.]
      dat.driverStateV2.leftDriverData.faceProb = 1.0
      dat.driverStateV2.rightDriverData.faceOrientation = [0., 0., 0.]
      dat.driverStateV2.rightDriverData.faceProb = 1.0
      pm.send('driverStateV2', dat)

      dat = messaging.new_message('driverMonitoringState', valid=True)
      dm = dat.driverMonitoringState
      dm.alertLevel = 0
      dm.activePolicy = 1
      dm.visionPolicyState.faceDetected = True
      dm.visionPolicyState.isDistracted = False
      dm.visionPolicyState.awarenessPercent = 100
      pm.send('driverMonitoringState', dat)

    # perf state
    if rk.frame % 25 == 0:
      dat = messaging.new_message('peripheralState', valid=True)
      dat.peripheralState = {
        'pandaType': 3,
        'voltage': 12000,
        'current': 5678,
        'fanSpeedRpm': 1000
      }
      pm.send('peripheralState', dat)

    # pack and send 100hz
    sim_car.update(v_state)

    rk.keep_time()


if __name__ == "__main__":
  main()