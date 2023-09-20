import cereal.messaging as messaging

from opendbc.can.packer import CANPacker
from opendbc.can.parser import CANParser
from openpilot.selfdrive.boardd.boardd_api_impl import can_list_to_can_capnp
from openpilot.selfdrive.car import crc8_pedal
from openpilot.tools.sim.lib.common import SimulatorState


class SimulatedCar:
  """Simulates a honda civic 2016 (panda state + can messages) to OpenPilot"""
  packer = CANPacker("honda_civic_touring_2016_can_generated")
  rpacker = CANPacker("acura_ilx_2016_nidec")

  def __init__(self):
    self.pm = messaging.PubMaster(['can', 'pandaStates'])
    self.sm = messaging.SubMaster(['carControl', 'controlsState', 'carParams'])
    self.cp = self.get_car_can_parser()
    self.idx = 0

  @staticmethod
  def get_car_can_parser():
    dbc_f = 'honda_civic_touring_2016_can_generated'
    checks = [
      (0xe4, 100),
      (0x1fa, 50),
      (0x200, 50),
    ]
    return CANParser(dbc_f, checks, 0)

  def send_can_messages(self, simulator_state: SimulatorState):
    if not simulator_state.valid:
      return

    msg = []

    # *** powertrain bus ***

    speed = simulator_state.speed * 3.6 # convert m/s to kph
    msg.append(self.packer.make_can_msg("ENGINE_DATA", 0, {"XMISSION_SPEED": speed}))
    msg.append(self.packer.make_can_msg("WHEEL_SPEEDS", 0, {
      "WHEEL_SPEED_FL": speed,
      "WHEEL_SPEED_FR": speed,
      "WHEEL_SPEED_RL": speed,
      "WHEEL_SPEED_RR": speed
    }))

    msg.append(self.packer.make_can_msg("SCM_BUTTONS", 0, {"CRUISE_BUTTONS": simulator_state.cruise_button}))

    values = {
      "COUNTER_PEDAL": self.idx & 0xF,
      "INTERCEPTOR_GAS": simulator_state.user_gas * 2**12,
      "INTERCEPTOR_GAS2": simulator_state.user_gas * 2**12,
    }
    checksum = crc8_pedal(self.packer.make_can_msg("GAS_SENSOR", 0, values)[2][:-1])
    values["CHECKSUM_PEDAL"] = checksum
    msg.append(self.packer.make_can_msg("GAS_SENSOR", 0, values))

    msg.append(self.packer.make_can_msg("GEARBOX", 0, {"GEAR": 4, "GEAR_SHIFTER": 8}))
    msg.append(self.packer.make_can_msg("GAS_PEDAL_2", 0, {}))
    msg.append(self.packer.make_can_msg("SEATBELT_STATUS", 0, {"SEATBELT_DRIVER_LATCHED": 1}))
    msg.append(self.packer.make_can_msg("STEER_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("STEERING_SENSORS", 0, {"STEER_ANGLE": simulator_state.steering_angle}))
    msg.append(self.packer.make_can_msg("VSA_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("STANDSTILL", 0, {"WHEELS_MOVING": 1 if simulator_state.speed >= 1.0 else 0}))
    msg.append(self.packer.make_can_msg("STEER_MOTOR_TORQUE", 0, {}))
    msg.append(self.packer.make_can_msg("EPB_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("DOORS_STATUS", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE_PARAMS", 0, {}))
    msg.append(self.packer.make_can_msg("CRUISE", 0, {}))
    msg.append(self.packer.make_can_msg("SCM_FEEDBACK", 0, {"MAIN_ON": 1}))
    msg.append(self.packer.make_can_msg("POWERTRAIN_DATA", 0,
                                    {
                                    "ACC_STATUS": int(simulator_state.is_engaged),
                                    "PEDAL_GAS": simulator_state.user_gas,
                                    "BRAKE_PRESSED": simulator_state.user_brake > 0
                                    }))
    msg.append(self.packer.make_can_msg("HUD_SETTING", 0, {}))
    msg.append(self.packer.make_can_msg("CAR_SPEED", 0, {}))

    # *** cam bus ***
    msg.append(self.packer.make_can_msg("STEERING_CONTROL", 2, {}))
    msg.append(self.packer.make_can_msg("ACC_HUD", 2, {}))
    msg.append(self.packer.make_can_msg("LKAS_HUD", 2, {}))
    msg.append(self.packer.make_can_msg("BRAKE_COMMAND", 2, {}))

    # *** radar bus ***
    if self.idx % 5 == 0:
      msg.append(self.rpacker.make_can_msg("RADAR_DIAGNOSTIC", 1, {"RADAR_STATE": 0x79}))
      for i in range(16):
        msg.append(self.rpacker.make_can_msg("TRACK_%d" % i, 1, {"LONG_DIST": 255.5}))

    self.pm.send('can', can_list_to_can_capnp(msg))

  def send_panda_state(self, simulator_state):
    self.sm.update(0)
    dat = messaging.new_message('pandaStates', 1)
    dat.valid = True
    dat.pandaStates[0] = {
      'ignitionLine': simulator_state.ignition,
      'pandaType': "blackPanda",
      'controlsAllowed': True,
      'safetyModel': 'hondaNidec',
      'alternativeExperience': self.sm["carParams"].alternativeExperience
    }
    self.pm.send('pandaStates', dat)

  def update(self, simulator_state: SimulatorState):
    self.send_can_messages(simulator_state)
    self.send_panda_state(simulator_state)

    self.idx += 1