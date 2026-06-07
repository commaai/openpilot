import random

from cereal import car, log
from panda import DLC_TO_LEN
from openpilot.selfdrive.pandad.pandad import PandaSafety, PandadPanda, _calculate_checksum, _unpack_can_buffer, send_panda_states


class FakeHandle:
  def __init__(self, reads=None):
    self.connected = True
    self.comms_healthy = True
    self.writes = []
    self.control_writes = []
    self.reads = list(reads or [])

  def bulkWrite(self, endpoint, data, timeout=0):
    self.writes.append((endpoint, bytes(data), timeout))
    return len(data)

  def bulkRead(self, endpoint, length, timeout=0):
    assert endpoint == 1
    assert length == 0x4000
    return self.reads.pop(0)

  def controlWrite(self, request_type, request, value, index, data, timeout=0):
    self.control_writes.append((request, value, index, timeout))


class FakePubMaster:
  def __init__(self):
    self.sent = []

  def send(self, service, msg):
    self.sent.append((service, msg.to_bytes()))


class FakeSafetySetter:
  def __init__(self):
    self.safety_modes = []
    self.alternative_experiences = []

  def set_alternative_experience(self, alternative_experience):
    self.alternative_experiences.append(alternative_experience)

  def set_safety_mode(self, mode, param=0):
    self.safety_modes.append((mode, param))


class FakeStatePanda:
  def __init__(self):
    self.hw_type = int(log.PandaState.PandaType.tres)
    self.comms_healthy = True
    self.safety_modes = []
    self.power_saves = []

  def health(self):
    return {
      "uptime": 123,
      "voltage": 12000,
      "current": 500,
      "safety_tx_blocked": 1,
      "safety_rx_invalid": 2,
      "tx_buffer_overflow": 3,
      "rx_buffer_overflow": 4,
      "faults": (1 << int(log.PandaState.FaultType.relayMalfunction)) |
                (1 << int(log.PandaState.FaultType.heartbeatLoopWatchdog)),
      "ignition_line": 1,
      "ignition_can": 0,
      "controls_allowed": 1,
      "car_harness_status": int(log.PandaState.HarnessStatus.normal),
      "safety_mode": int(car.CarParams.SafetyModel.noOutput),
      "safety_param": 7,
      "fault_status": int(log.PandaState.FaultStatus.none),
      "power_save_enabled": 1,
      "heartbeat_lost": 0,
      "alternative_experience": 3,
      "interrupt_load": 0.25,
      "fan_power": 33,
      "safety_rx_checks_invalid": 0,
      "spi_error_count": 9,
      "sbu1_voltage_mV": 1000,
      "sbu2_voltage_mV": 2000,
      "som_reset_triggered": 0,
      "sound_output_level": 11,
    }

  def can_health(self, can_number):
    return {
      "bus_off": can_number == 1,
      "bus_off_cnt": can_number,
      "error_warning": 0,
      "error_passive": 0,
      "last_error": 0,
      "last_stored_error": 0,
      "last_data_error": 0,
      "last_data_stored_error": 0,
      "receive_error_cnt": 1,
      "transmit_error_cnt": 2,
      "total_error_cnt": 3,
      "total_tx_lost_cnt": 4,
      "total_rx_lost_cnt": 5,
      "total_tx_cnt": 6,
      "total_rx_cnt": 7,
      "total_fwd_cnt": 8,
      "total_tx_checksum_error_cnt": 9,
      "can_speed": 500,
      "can_data_speed": 2000,
      "canfd_enabled": 1,
      "brs_enabled": 1,
      "canfd_non_iso": 0,
      "irq0_call_rate": 10,
      "irq1_call_rate": 11,
      "irq2_call_rate": 12,
      "can_core_reset_count": 13,
    }

  def set_safety_mode(self, mode, param=0):
    self.safety_modes.append((mode, param))

  def set_power_save(self, power_save):
    self.power_saves.append(power_save)


def make_panda(handle=None):
  panda = object.__new__(PandadPanda)
  panda._context = None
  panda._handle = handle or FakeHandle()
  panda.serial = "test"
  panda.bootstub = False
  panda.comms_healthy = True
  panda.can_rx_overflow_buffer = b""
  panda.hw_type = 9
  return panda


def random_can_messages(count):
  rng = random.Random(0)
  msgs = []
  for i in range(count):
    data_len = DLC_TO_LEN[rng.randrange(len(DLC_TO_LEN))]
    dat = bytes(rng.getrandbits(8) for _ in range(data_len))
    addr = rng.randrange(1 << 29)
    bus = i % 3
    msgs.append((addr, dat, bus))
  return msgs


def packed_from_panda(can_msgs):
  handle = FakeHandle()
  panda = make_panda(handle)
  panda.can_send_many(can_msgs)
  return b"".join(dat for _, dat, _ in handle.writes), handle


def test_can_send_round_trip_and_chunking():
  can_msgs = random_can_messages(200)
  packed, handle = packed_from_panda(can_msgs)
  unpacked, overflow = _unpack_can_buffer(packed)

  assert overflow == b""
  assert unpacked == can_msgs
  assert len(handle.writes) > 1
  assert all(endpoint == 3 for endpoint, _, _ in handle.writes)


def test_can_send_skips_messages_for_other_pandas():
  can_msgs = [(1, b"12345678", 0), (2, b"12345678", 4), (3, b"12345678", 0xc0)]
  packed, _ = packed_from_panda(can_msgs)
  unpacked, overflow = _unpack_can_buffer(packed)

  assert overflow == b""
  assert unpacked == [can_msgs[0]]


def test_can_receive_keeps_partial_packet_for_next_read():
  can_msgs = random_can_messages(10)
  packed, _ = packed_from_panda(can_msgs)
  split = 6 + DLC_TO_LEN[packed[0] >> 4] + 3
  handle = FakeHandle([packed[:split], packed[split:]])
  panda = make_panda(handle)

  healthy, first = panda.can_receive()
  assert healthy
  assert first
  assert panda.can_rx_overflow_buffer

  healthy, second = panda.can_receive()
  assert healthy
  assert first + second == can_msgs
  assert panda.can_rx_overflow_buffer == b""


def test_can_receive_bad_checksum_resets_comms():
  packed, _ = packed_from_panda([(1, b"12345678", 0)])
  bad = bytearray(packed)
  bad[-1] ^= 0xff
  handle = FakeHandle([bytes(bad)])
  panda = make_panda(handle)

  healthy, msgs = panda.can_receive()

  assert not healthy
  assert msgs == []
  assert handle.control_writes == [(0xc0, 0, 0, 100)]


def test_set_safety_mode_accepts_capnp_enum():
  panda = make_panda()

  panda.set_safety_mode(car.CarParams.SafetyModel.elm327, 1)

  assert panda._handle.control_writes[-1] == (0xdc, 3, 1, 100)


def test_checksum_xor_matches_zeroed_packet_property():
  packed, _ = packed_from_panda([(0x123, b"abcdef", 2)])
  assert _calculate_checksum(packed) == 0


def test_panda_safety_sets_raw_safety_model_from_car_params():
  params = car.CarParams.new_message()
  safety_config = params.init("safetyConfigs", 1)[0]
  safety_config.safetyModel = car.CarParams.SafetyModel.elm327
  safety_config.safetyParam = 42
  params.alternativeExperience = 7

  panda = FakeSafetySetter()
  safety = object.__new__(PandaSafety)
  safety.panda = panda

  safety._set_safety_mode(params.to_bytes())

  assert panda.alternative_experiences == [7]
  assert panda.safety_modes == [(3, 42)]


def test_send_panda_states_maps_health_and_faults():
  panda = FakeStatePanda()
  pm = FakePubMaster()

  ignition = send_panda_states(pm, panda, is_onroad=True, spoofing_started=False)

  assert ignition is True
  assert panda.power_saves == [False]
  assert len(pm.sent) == 1
  service, dat = pm.sent[0]
  assert service == "pandaStates"

  with log.Event.from_bytes(dat) as msg:
    ps = msg.pandaStates[0]
    assert msg.valid
    assert ps.pandaType == log.PandaState.PandaType.tres
    assert ps.voltage == 12000
    assert ps.current == 500
    assert ps.safetyModel == car.CarParams.SafetyModel.noOutput
    assert ps.safetyParam == 7
    assert ps.controlsAllowed
    assert ps.canState1.busOff
    assert ps.canState2.canCoreResetCnt == 13
    assert list(ps.faults) == [
      log.PandaState.FaultType.relayMalfunction,
      log.PandaState.FaultType.heartbeatLoopWatchdog,
    ]
