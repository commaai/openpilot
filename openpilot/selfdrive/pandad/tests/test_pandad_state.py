from unittest.mock import call

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.parameterized import parameterized
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.pandad.panda_safety import PandaSafety
from openpilot.selfdrive.pandad.pandad import panda_states_msg, update_safety
from openpilot.selfdrive.pandad.tests.helpers import HEALTH, TRES, SafetyModel, can_health, make_panda

FaultType = log.PandaState.FaultType
LEC_NAMES = ['noError', 'stuffError', 'formError', 'ackError', 'bit1Error', 'bit0Error', 'crcError', 'noChange']
# health keys that aren't copied over as they are, or at all
RENAMED = {"safety_mode": "safetyModel", "car_harness_status": "harnessStatus", "can_core_reset_count": "canCoreResetCnt"}
UNPUBLISHED = ("som_reset_triggered", "heartbeat_disabled", "temperature", "total_tx_checksum_error_cnt", "sbu1_voltage_mV", "sbu2_voltage_mV")


def camel_case(health: dict) -> dict:
  return {RENAMED.get(k, k.split('_')[0] + ''.join(w.title() for w in k.split('_')[1:])): v for k, v in health.items() if k not in UNPUBLISHED}


class TestPandaState(OpenpilotTestCase):
  def test_panda_states_msg(self):
    faults = (1 << int(FaultType.relayMalfunction)) | (1 << int(FaultType.safetyRedundancy)) | (1 << int(FaultType.heartbeatLoopWatchdog)) | (1 << 31)
    health = HEALTH | {"faults": faults, "alternative_experience": 0xFFFF}
    msg = panda_states_msg(TRES, health, [can_health(i) for i in range(3)])
    assert msg.valid and len(msg.pandaStates) == 1
    ps = msg.pandaStates[0].to_dict()
    for k in ('deprecated', 'controlsAllowedRESERVED1', 'controlsAllowedRESERVED2'):
      ps.pop(k)

    expected = camel_case(health) | {
      'pandaType': 'tres', 'safetyModel': 'noOutput', 'faultStatus': 'faultPerm', 'harnessStatus': 'normal',
      'alternativeExperience': -1,  # Int16
      'sbu1Voltage': 1.0, 'sbu2Voltage': 2.0,
      # ascending, and bits past the last fault type dropped
      'faults': ['relayMalfunction', 'heartbeatLoopWatchdog', 'safetyRedundancy'],
    }
    for i in range(3):
      expected[f'canState{i}'] = camel_case(can_health(i)) | {
        'lastError': LEC_NAMES[(i + 1) % 8], 'lastStoredError': LEC_NAMES[(i + 3) % 8],
        'lastDataError': LEC_NAMES[(i + 5) % 8], 'lastDataStoredError': LEC_NAMES[(i + 7) % 8],
      }
    assert ps == expected

  @parameterized.expand([
    # SILENT goes to NO_OUTPUT twice: to make the bus live, and because we're offroad
    ({"safety_mode": SafetyModel.silent, "power_save_enabled": False}, False,
     [call.set_safety_mode(SafetyModel.noOutput), call.set_power_save(True), call.set_safety_mode(SafetyModel.noOutput)]),
    # car safety mode stays, power save goes off
    ({"safety_mode": SafetyModel.hondaNidec, "ignition_can": True}, True, [call.set_power_save(False)]),
    # ignition gone while still onroad: relay closes
    ({"safety_mode": SafetyModel.hondaNidec, "power_save_enabled": False}, True, [call.set_power_save(True), call.set_safety_mode(SafetyModel.noOutput)]),
    # offroad with ignition on, e.g. too hot: relay closes
    ({"safety_mode": SafetyModel.hondaNidec, "ignition_line": True, "power_save_enabled": False}, False, [call.set_safety_mode(SafetyModel.noOutput)]),
    ({"safety_mode": SafetyModel.allOutput, "ignition_line": True, "power_save_enabled": False}, True, []),
  ])
  def test_update_safety(self, health, is_onroad, writes):
    panda = make_panda()
    update_safety(panda, HEALTH | health, is_onroad)
    assert panda.method_calls == writes


def car_params(safety_model=SafetyModel.allOutput, safety_param=0, alternative_experience=0, configs=1, not_car=False) -> bytes:
  cp = car.CarParams.new_message(alternativeExperience=alternative_experience, notCar=not_car)
  for sc in cp.init('safetyConfigs', configs):
    sc.safetyModel = safety_model
    sc.safetyParam = safety_param
  return cp.to_bytes()


class TestPandaSafety(OpenpilotTestCase):
  def setUp(self):
    super().setUp()
    self.params = Params()
    self.panda = make_panda()
    self.safety = PandaSafety(self.panda)

  def test_onroad_sequence(self):
    s, p, writes = self.safety, self.params, self.panda.method_calls
    elm327 = [call.set_safety_mode(SafetyModel.elm327, 1), call.set_safety_mode(SafetyModel.elm327, 0)]
    honda = [call.set_alternative_experience(0xFFFF), call.set_safety_mode(SafetyModel.hondaBosch, 5)]

    s.configure_safety_mode(False)
    assert writes == []

    # ELM327 without OBD multiplexing to fingerprint
    s.configure_safety_mode(True)
    assert writes == elm327[:1]

    p.put_bool("ObdMultiplexingEnabled", True, block=True)
    s.configure_safety_mode(True)
    assert writes == elm327
    # written before it returns, card waits on it
    assert p.get_bool("ObdMultiplexingChanged")

    # nothing until card is done
    p.put_bool("FirmwareQueryDone", True, block=True)
    p.put("CarParams", car_params(SafetyModel.hondaBosch, 5, alternative_experience=-1, not_car=True), block=True)
    s.configure_safety_mode(True)
    assert writes == elm327

    # alternative experience first, the panda only takes it outside of a car safety mode. set once per drive
    p.put_bool("ControlsReady", True, block=True)
    s.configure_safety_mode(True)
    s.configure_safety_mode(True)
    assert writes == elm327 + honda and s.not_car

    # and again on the next drive
    p.remove("ObdMultiplexingChanged")
    s.configure_safety_mode(False)
    assert not s.not_car
    s.configure_safety_mode(True)
    assert writes == (elm327 + honda) * 2
    assert p.get_bool("ObdMultiplexingChanged")

  @parameterized.expand([("no_safety_config", car_params(configs=0)), ("invalid", b"\x01" * 27)])
  def test_bad_car_params(self, name, cp):
    # noOutput, once
    for k in ("FirmwareQueryDone", "ControlsReady"):
      self.params.put_bool(k, True, block=True)
    self.params.put("CarParams", cp, block=True)
    self.safety.configure_safety_mode(True)
    self.safety.configure_safety_mode(True)
    assert self.panda.method_calls[-2:] == [call.set_alternative_experience(0), call.set_safety_mode(SafetyModel.noOutput, 0)]
    assert self.panda.set_safety_mode.call_count == 2
