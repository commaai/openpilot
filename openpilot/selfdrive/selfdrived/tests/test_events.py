import math
from types import SimpleNamespace

from openpilot.selfdrive.selfdrived.events import overheat_alert


class FakeSubMaster:
  def __init__(self, device_state):
    self.device_state = device_state

  def __getitem__(self, service):
    assert service == "deviceState"
    return self.device_state


def make_device_state(cpu_temp, gpu_temp, memory_temp, max_temp):
  return SimpleNamespace(cpuTempC=cpu_temp, gpuTempC=gpu_temp, memoryTempC=memory_temp, maxTempC=max_temp)


def test_overheat_alert_ignores_nan_temps():
  sm = FakeSubMaster(make_device_state([math.nan], [64.0], math.nan, 72.0))

  alert = overheat_alert(None, None, sm, False, 100, None)

  assert alert.alert_text_2 == "72 °C"


def test_overheat_alert_handles_only_nan_temps():
  sm = FakeSubMaster(make_device_state([math.nan], [math.nan], math.nan, math.nan))

  alert = overheat_alert(None, None, sm, False, 100, None)

  assert alert.alert_text_2 == "Temperature unavailable"
