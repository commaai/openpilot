from types import SimpleNamespace

import pytest
import usb1

from openpilot.system.hardware.chestnut.monitoring import ChestnutMonitoring, ChestnutUsb, GPU_FIELDS


class FakeUsb:
  def __init__(self, connected=True):
    self.connected = connected
    self.ina = (12000, 1500, False)
    self.ltssm = 0x78
    self.ina_error = None
    self.pcie_error = None
    self.closed = False
    self.connect_count = 0

  def connect(self):
    self.connect_count += 1
    return self.connected

  def close(self):
    self.closed = True

  def read_ina(self):
    if self.ina_error is not None:
      raise self.ina_error
    return self.ina

  def read_pcie_ltssm(self):
    if self.pcie_error is not None:
      raise self.pcie_error
    return self.ltssm


def gpu_state(**kwargs):
  defaults = dict.fromkeys(GPU_FIELDS, 0)
  defaults.update(kwargs)
  return SimpleNamespace(**defaults)


class FakeSubMaster:
  def __init__(self, state, *, updated=True, valid=True, recv_time=1., modeld_running=True):
    self.state = state
    self.modeld_running = modeld_running
    self.updated = {'chestnutGpuState': updated}
    self.valid = {'chestnutGpuState': valid}
    self.recv_time = {'chestnutGpuState': recv_time}

  def __getitem__(self, key):
    if key == 'managerState':
      return SimpleNamespace(processes=[SimpleNamespace(name='modeld', shouldBeRunning=True, running=self.modeld_running)])
    assert key == 'chestnutGpuState'
    return self.state


def test_no_messages_without_chestnut_or_retries_in_cycle():
  usb = FakeUsb(connected=False)
  monitoring = ChestnutMonitoring(usb)
  monitoring.set_enabled(True)
  assert monitoring.build_message() is None
  assert monitoring.build_message() is None
  assert monitoring.usb_failed
  assert usb.connect_count == 1


def test_offroad_closes_usb_and_stops_messages():
  usb = FakeUsb()
  monitoring = ChestnutMonitoring(usb)
  monitoring.set_enabled(True)
  assert monitoring.build_message() is not None
  monitoring.set_enabled(False)
  assert usb.closed
  assert monitoring.build_message() is None


def test_ina_pcie_and_gpu_metrics():
  monitoring = ChestnutMonitoring(FakeUsb())
  monitoring.set_enabled(True)
  gpu = gpu_state(tempC=72., memoryTempC=80., powerDrawW=45., powerLimitW=55.,
                  gpuUsagePercent=91, gpuClockMhz=2200, fanSpeedRpm=3100)
  monitoring.update_gpu_state(FakeSubMaster(gpu, recv_time=10.), 10.)
  msg = monitoring.build_message()
  assert msg.valid
  assert (msg.chestnutState.supplyVoltage, msg.chestnutState.supplyCurrent,
          msg.chestnutState.supplyFault, msg.chestnutState.pcieLtssm) == (12000, 1500, False, 0x78)
  for field in GPU_FIELDS:
    assert getattr(msg.chestnutState, field) == getattr(gpu, field)


def test_stale_and_invalid_model_metrics_are_cleared():
  monitoring = ChestnutMonitoring(FakeUsb())
  monitoring.set_enabled(True)
  monitoring.update_gpu_state(FakeSubMaster(gpu_state(tempC=72.), recv_time=10.), 10.)
  monitoring.update_gpu_state(FakeSubMaster(gpu_state(), updated=False, recv_time=10.), 11.1)
  assert monitoring.build_message().chestnutState.tempC == 0.
  monitoring.update_gpu_state(FakeSubMaster(gpu_state(tempC=75.), valid=False), 11.)
  assert monitoring.build_message().chestnutState.tempC == 0.
  monitoring.update_gpu_state(FakeSubMaster(gpu_state(tempC=76.)), 12.)
  assert monitoring.build_message().chestnutState.tempC == 76.


def test_poll_does_not_wait_for_model_updates():
  monitoring = ChestnutMonitoring(FakeUsb())
  monitoring.set_enabled(True)
  assert monitoring.update(FakeSubMaster(gpu_state(), recv_time=10.), 10.) is not None
  assert monitoring.update(FakeSubMaster(gpu_state(), updated=False, recv_time=10.), 10.1) is not None
  assert monitoring.update(FakeSubMaster(gpu_state(), updated=False, recv_time=10.), 11.1) is not None


def test_poll_resumes_immediately_when_modeld_exits():
  monitoring = ChestnutMonitoring(FakeUsb())
  monitoring.set_enabled(True)
  assert monitoring.update(FakeSubMaster(gpu_state(), updated=False, recv_time=10., modeld_running=False), 10.1) is not None


def test_supply_loss_and_recovery():
  usb = FakeUsb()
  monitoring = ChestnutMonitoring(usb)
  monitoring.set_enabled(True)
  usb.ina = (0, 0, True)
  lost = monitoring.build_message()
  assert (lost.chestnutState.supplyVoltage, lost.chestnutState.supplyCurrent, lost.chestnutState.supplyFault) == (0, 0, True)
  usb.ina = (12000, 1500, False)
  recovered = monitoring.build_message()
  assert (recovered.chestnutState.supplyVoltage, recovered.chestnutState.supplyCurrent,
          recovered.chestnutState.supplyFault) == (12000, 1500, False)


@pytest.mark.parametrize("failure", ["ina", "pcie"])
def test_usb_failure_stays_invalid_until_next_ignition_and_preserves_gpu(failure):
  usb = FakeUsb()
  monitoring = ChestnutMonitoring(usb)
  monitoring.set_enabled(True)
  gpu = gpu_state(tempC=72., gpuUsagePercent=91)
  monitoring.update_gpu_state(FakeSubMaster(gpu), 1.)
  assert monitoring.build_message().valid
  setattr(usb, f"{failure}_error", OSError("removed"))
  msg = monitoring.build_message()
  assert not msg.valid
  assert msg.chestnutState.supplyVoltage == 0
  assert msg.chestnutState.pcieLtssm == 0
  assert msg.chestnutState.tempC == 72.
  assert msg.chestnutState.gpuUsagePercent == 91
  assert monitoring.usb_failed
  assert usb.closed
  connect_count = usb.connect_count
  assert not monitoring.build_message().valid
  assert usb.connect_count == connect_count

  setattr(usb, f"{failure}_error", None)
  assert not monitoring.build_message().valid


def test_offroad_transition_recovers_after_failure():
  usb = FakeUsb()
  monitoring = ChestnutMonitoring(usb)
  monitoring.set_enabled(True)
  assert monitoring.build_message().valid
  usb.ina_error = OSError("removed")
  assert not monitoring.build_message().valid
  usb.ina_error = None
  monitoring.set_enabled(False)
  monitoring.set_enabled(True)
  assert not monitoring.usb_failed
  assert monitoring.build_message().valid


def test_usb_timeout_latches_during_loading():
  usb = FakeUsb()
  monitoring = ChestnutMonitoring(usb)
  monitoring.set_enabled(True)
  assert monitoring.build_message().valid
  usb.closed = False
  usb.ina_error = usb1.USBErrorTimeout()
  assert not monitoring.build_message().valid
  assert monitoring.usb_failed
  assert usb.closed


class FakeHandle:
  def __init__(self, responses):
    self.responses = iter(responses)

  def controlRead(self, *args, **kwargs):
    return next(self.responses)


def connected_usb(*responses):
  usb = ChestnutUsb()
  usb.handle = FakeHandle(responses)
  return usb


def test_usb_decodes_complete_responses():
  usb = connected_usb(b'\xe0.\xdc\x05\x01', b'\x78')
  assert usb.read_ina() == (12000, 1500, True)
  assert usb.read_pcie_ltssm() == 0x78


@pytest.mark.parametrize("response,read", [(b'\x00' * 4, "read_ina"), (b'', "read_pcie_ltssm")])
def test_usb_rejects_malformed_response(response, read):
  usb = connected_usb(response)
  with pytest.raises(ValueError, match="short chestnut USB response"):
    getattr(usb, read)()
