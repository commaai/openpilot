from dataclasses import dataclass

import openpilot.cereal.messaging as messaging
from opendbc.can.packer import CANPacker
from opendbc.car.honda.values import HondaSafetyFlags
from openpilot.common.params import Params
from openpilot.selfdrive.pandad.pandad_api_impl import can_list_to_can_capnp


@dataclass
class Driver:
  """What the person in the driver's seat is doing"""
  ignition: bool = True
  cruise_button: int = 0
  gas: float = 0.
  brake: float = 0.
  steer_torque: float = 0.
  left_blinker: bool = False
  right_blinker: bool = False


class SimulatedCar:
  """A Honda Civic 2022 as openpilot sees it: its CAN messages and the panda they come through"""
  packer = CANPacker("honda_bosch_radarless_generated")

  def __init__(self):
    self.pm = messaging.PubMaster(['can', 'pandaStates'])
    self.sm = messaging.SubMaster(['carControl', 'carParams', 'selfdriveState'])
    self.params = Params()
    self.obd_multiplexing = False
    self.frame = 0

  def send_can_messages(self, driver: Driver, speed: float, steering_angle: float, engaged: bool):
    msg = []

    # *** powertrain bus ***

    speed = speed * 3.6 # convert m/s to kph
    msg.append(self.packer.make_can_msg("ENGINE_DATA", 0, {"XMISSION_SPEED": speed}))
    msg.append(self.packer.make_can_msg("WHEEL_SPEEDS", 0, {
      "WHEEL_SPEED_FL": speed,
      "WHEEL_SPEED_FR": speed,
      "WHEEL_SPEED_RL": speed,
      "WHEEL_SPEED_RR": speed
    }))

    msg.append(self.packer.make_can_msg("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": driver.cruise_button}))

    msg.append(self.packer.make_can_msg("GEARBOX_AUTO", 0, {"GEAR_SHIFTER": 4}))
    msg.append(self.packer.make_can_msg("GAS_PEDAL_2", 0, {}))
    msg.append(self.packer.make_can_msg("SEATBELT_STATUS", 0, {"SEATBELT_DRIVER_LATCHED": 1}))
    msg.append(self.packer.make_can_msg("STEER_STATUS", 0, {"STEER_TORQUE_SENSOR": driver.steer_torque}))
    msg.append(self.packer.make_can_msg("STEERING_SENSORS", 0, {"STEER_ANGLE": steering_angle}))
    msg.append(self.packer.make_can_msg("VSA_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("STANDSTILL", 0, {"WHEELS_MOVING": 1 if speed >= 3.6 else 0}))
    msg.append(self.packer.make_can_msg("STEER_MOTOR_TORQUE", 0, {}))
    msg.append(self.packer.make_can_msg("EPB_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("DOORS_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE_FAULT_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("SCM_FEEDBACK", 0,
                                    {
                                      "MAIN_ON": 1,
                                      "LEFT_BLINKER": driver.left_blinker,
                                      "RIGHT_BLINKER": driver.right_blinker
                                    }))
    msg.append(self.packer.make_can_msg("POWERTRAIN_DATA", 0,
                                    {
                                    "ACC_STATUS": int(engaged),
                                    "PEDAL_GAS": driver.gas,
                                    "BRAKE_PRESSED": driver.brake > 0
                                    }))
    msg.append(self.packer.make_can_msg("CAR_SPEED", 0, {}))

    # *** cam bus ***
    msg.append(self.packer.make_can_msg("STEERING_CONTROL", 2, {}))
    msg.append(self.packer.make_can_msg("ACC_HUD", 2, {}))
    msg.append(self.packer.make_can_msg("LKAS_HUD", 2, {}))

    self.pm.send('can', can_list_to_can_capnp(msg))

  def send_panda_state(self, driver: Driver):
    if self.params.get_bool("ObdMultiplexingEnabled") != self.obd_multiplexing:
      self.obd_multiplexing = not self.obd_multiplexing
      self.params.put_bool("ObdMultiplexingChanged", True, block=True)

    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': driver.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaBosch',
      'alternativeExperience': self.sm["carParams"].alternativeExperience,
      'safetyParam': HondaSafetyFlags.RADARLESS.value | HondaSafetyFlags.BOSCH_LONG.value,
    }
    self.pm.send('pandaStates', dat)

  def update(self, driver: Driver, speed: float, steering_angle: float, engaged: bool):
    """Called at 100 Hz"""
    self.send_can_messages(driver, speed, steering_angle, engaged)
    if self.frame % 50 == 0:
      self.send_panda_state(driver)
    self.frame += 1
