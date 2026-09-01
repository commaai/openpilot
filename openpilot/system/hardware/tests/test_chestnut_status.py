from itertools import product
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT
from openpilot.system.hardware.chestnut.status import STARTUP_STABILIZATION_TIME, ChestnutStatus, update_modeld_state


FAILURE_ALERTS = {
  "Offroad_ChestnutModelError", "Offroad_ChestnutNotDetected", "Offroad_ChestnutOverheated",
  "Offroad_ChestnutPcieUnavailable", "Offroad_ChestnutUncompiled", "Offroad_ChestnutUpdateFailed", "Offroad_ChestnutUsbSlow",
}
DEVICE = {"vendorId": 0xADD1, "productId": 1, "product": CHESTNUT_USB_PRODUCT, "speedMbps": 10000}
STATE = SimpleNamespace(supplyVoltage=12000, supplyFault=False, pcieLtssm=0x78, tempC=50., memoryTempC=60.)


class Alerts:
  def __init__(self):
    self.values = {}
    self.triggers = dict.fromkeys(FAILURE_ALERTS, 0)
    self.clears = dict.fromkeys(FAILURE_ALERTS, 0)

  def set(self, name, enabled, extra=None):
    previous = self.values.get(name, (False, None))[0]
    self.values[name] = (enabled, extra)
    if name in FAILURE_ALERTS and enabled != previous:
      (self.triggers if enabled else self.clears)[name] += 1

  @property
  def active(self):
    return {name for name in FAILURE_ALERTS if self.values[name][0]}


def process_state(*, running=False, should_run=True):
  return SimpleNamespace(name="modeld", running=running, shouldBeRunning=should_run)


@pytest.mark.parametrize(("states", "expected"), [
  ([(False, True, True), (True, True, True)], [False, False]),
  ([(True, True, True), (False, True, True)], [False, True]),
  ([(True, True, True), (False, False, True)], [False, False]),
  ([(True, True, False), (False, True, False)], [False, False]),
])
def test_modeld_process_transitions(states, expected):
  modeld_seen = False
  failed = []
  for running, should_run, chestnut_started in states:
    modeld_seen, process_failed = update_modeld_state([process_state(running=running, should_run=should_run)],
                                                      modeld_seen, chestnut_started)
    failed.append(process_failed)
  assert failed == expected


def update(status, alerts=None, *, offroad=True, branch="release-chestnut", devices=None, firmware_failed=False,
           error=False, model_recovered=False, state=STATE, compiled=True):
  alerts = alerts or Alerts()
  status.update(offroad, branch, [DEVICE] if devices is None else devices, firmware_failed, compiled,
                error, model_recovered, state, alerts.set)
  assert len(alerts.active) <= 1
  return alerts


def test_software_failure():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, error=True)
  assert alerts.active == {"Offroad_ChestnutModelError"}
  update(status, alerts)
  assert not alerts.active


def test_non_chestnut_ignores_model_error():
  assert not update(ChestnutStatus(), devices=[], error=True).active


def test_usb_cold_boot_and_pull_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  with patch("openpilot.system.hardware.chestnut.status.time.monotonic", return_value=status.started + 9):
    update(status, alerts, devices=[])
  assert not alerts.active
  with patch("openpilot.system.hardware.chestnut.status.time.monotonic", return_value=status.started + 11):
    update(status, alerts, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert alerts.values["Offroad_ChestnutNotDetected"][1] == "Chestnut USB disconnected. Check USB connection. Restart the car to retry."

  with patch("openpilot.system.hardware.chestnut.status.time.monotonic", return_value=status.started + 11):
    update(status, alerts, offroad=False, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}

  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=False, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert alerts.values["Offroad_ChestnutNotDetected"][1] == "Chestnut USB disconnected. Check USB connection. Restart the car to retry."
  update(status, alerts, offroad=False)
  assert not alerts.active
  assert alerts.triggers["Offroad_ChestnutNotDetected"] == 1
  assert alerts.clears["Offroad_ChestnutNotDetected"] == 1


def test_usb_disconnection_clears_after_recovery_and_offroad():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=False, devices=[], state=None)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}

  update(status, alerts, offroad=False)
  assert not alerts.active
  update(status, alerts, offroad=False, devices=[], state=None)
  update(status, alerts, offroad=True)
  assert not alerts.active
  update(status, alerts, offroad=False)
  assert not alerts.active


def test_usb_io_failure_does_not_report_disconnection():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=False, state=None)
  assert not alerts.active
  update(status, alerts, offroad=False, error=True, state=None)
  assert alerts.active == {"Offroad_ChestnutModelError"}


def test_usb_disconnection_remains_model_failure_cause():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=False, devices=[], state=None)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  update(status, alerts, offroad=False)
  assert not alerts.active
  update(status, alerts, offroad=False, error=True)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert "USB reconnected" in alerts.values["Offroad_ChestnutNotDetected"][1]
  update(status, alerts, offroad=False, model_recovered=True)
  assert not alerts.active


def test_detected_and_not_detected_are_mutually_exclusive():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, branch="master")
  assert alerts.values["Offroad_ChestnutBranch"][0]
  update(status, alerts, offroad=False, branch="master", devices=[], state=None)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert not alerts.values["Offroad_ChestnutBranch"][0]


@pytest.mark.parametrize("present", list(product((False, True), repeat=4)))
def test_usb_transition_histories_report_current_state(present):
  status, alerts = ChestnutStatus(), Alerts()
  seen = False
  for connected in present:
    seen |= connected
    update(status, alerts, offroad=False, branch="master", devices=[DEVICE] if connected else [])
    missing = "Offroad_ChestnutNotDetected" in alerts.active
    assert missing == (seen and not connected)
    assert not (missing and alerts.values["Offroad_ChestnutBranch"][0])

  update(status, alerts, offroad=True, branch="master", devices=[DEVICE] if present[-1] else [])
  assert ("Offroad_ChestnutNotDetected" in alerts.active) == (seen and not present[-1])


def test_delayed_usb_detection_then_removal():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, devices=[])
  assert not alerts.active
  update(status, alerts, offroad=False)
  assert not alerts.active
  update(status, alerts, offroad=False, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}


def test_seen_chestnut_remains_expected_across_ignition_cycles():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, branch="master")
  update(status, alerts, offroad=False, branch="master")
  update(status, alerts, branch="master")
  update(status, alerts, offroad=False, branch="master", devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert alerts.values["Offroad_ChestnutNotDetected"][1] == "Chestnut USB disconnected. Check USB connection. Restart the car to retry."


def test_slow_usb_transition():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, devices=[DEVICE | {"speedMbps": 480}])
  assert alerts.active == {"Offroad_ChestnutUsbSlow"}
  update(status, alerts)
  assert not alerts.active
  assert (alerts.triggers["Offroad_ChestnutUsbSlow"], alerts.clears["Offroad_ChestnutUsbSlow"]) == (1, 1)

  update(status, alerts, offroad=False, devices=[DEVICE | {"speedMbps": 480}])
  assert alerts.active == {"Offroad_ChestnutUsbSlow"}


@pytest.mark.parametrize(("firmware_failed", "compiled", "expected"), [
  (True, True, "Offroad_ChestnutUpdateFailed"), (False, False, "Offroad_ChestnutUncompiled"),
])
def test_setup_failure_transitions(firmware_failed, compiled, expected):
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, firmware_failed=firmware_failed, compiled=compiled)
  assert alerts.active == {expected}
  update(status, alerts)
  assert not alerts.active
  assert (alerts.triggers[expected], alerts.clears[expected]) == (1, 1)
  update(status, alerts, offroad=False, firmware_failed=firmware_failed, compiled=compiled)
  assert alerts.active == {expected}


def start_model(status, alerts, state=STATE):
  update(status, alerts, offroad=False, state=state)
  status.onroad_since -= STARTUP_STABILIZATION_TIME
  update(status, alerts, offroad=False, state=state)


def test_pcie_failure_transition():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=link_down)
  update(status, alerts, offroad=False, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "PCIe link unavailable" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  update(status, alerts, offroad=True)
  assert not alerts.active


def test_pcie_failure_without_model_state():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  status.onroad_since -= STARTUP_STABILIZATION_TIME
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=link_down)
  update(status, alerts, offroad=False, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}


def test_initial_power_absence_transition():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  start_model(status, alerts, low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_power_absence_before_model_load():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, error=True, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_offroad_power_and_pcie_do_not_report_normal_power_down():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, state=low)
  assert not alerts.active
  update(status, alerts)
  assert not alerts.active

  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, state=link_down)
  update(status, alerts, state=link_down)
  assert not alerts.active
  update(status, alerts)
  assert not alerts.active


def test_onroad_power_failure_clears_offroad():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, error=True, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  update(status, alerts, error=True, state=low)
  assert not alerts.active


@pytest.mark.parametrize("kwargs", [
  {"firmware_failed": True},
  {"compiled": False},
  {"devices": [DEVICE | {"speedMbps": 480}]},
])
def test_setup_alerts_clear_offroad(kwargs):
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, **kwargs)
  update(status, alerts, offroad=True, **kwargs)
  assert not alerts.active
  assert not alerts.values["Offroad_ChestnutBranch"][0]


def test_model_error_remains_offroad_after_drive():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, error=True)
  update(status, alerts, offroad=True, error=True)
  assert alerts.active == {"Offroad_ChestnutModelError"}


def test_branch_alert_remains_offroad_after_drive():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, branch="master")
  update(status, alerts, offroad=True, branch="master")
  assert alerts.values["Offroad_ChestnutBranch"][0]


def test_overheat_and_disconnected_remain_offroad():
  hot = SimpleNamespace(**(vars(STATE) | {"tempC": 105.}))

  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=True, state=hot)
  assert alerts.active == {"Offroad_ChestnutOverheated"}

  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=True, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}


def test_crank_power_loss_and_recovery_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  assert "engine-crank voltage drop" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False, error=True, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False, model_recovered=True, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=True, state=STATE)
  assert not alerts.active
  assert alerts.triggers["Offroad_ChestnutPcieUnavailable"] == 1


def test_single_startup_power_sample_is_ignored():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  assert not alerts.active
  update(status, alerts, offroad=False, state=STATE)
  assert not alerts.active


def test_crank_power_loss_recovers_during_stabilization():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  update(status, alerts, offroad=False, state=low)
  update(status, alerts, offroad=False, state=STATE)
  assert not alerts.active
  status.onroad_since -= STARTUP_STABILIZATION_TIME
  update(status, alerts, offroad=False, model_recovered=True, state=STATE)
  assert not alerts.active
  assert status.hardware_failure is None
  assert status.startup_failure is None


def test_startup_power_loss_causing_model_failure_is_reported():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  assert not alerts.active
  update(status, alerts, offroad=False, error=True, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_startup_power_cause_remains_until_model_load_finishes():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  update(status, alerts, offroad=False, state=STATE)
  status.onroad_since -= STARTUP_STABILIZATION_TIME
  update(status, alerts, offroad=False, state=STATE)
  assert not alerts.active
  update(status, alerts, offroad=False, error=True, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_sustained_startup_power_loss_is_reported():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  update(status, alerts, offroad=False, state=low)
  assert not alerts.active
  status.onroad_since -= STARTUP_STABILIZATION_TIME
  update(status, alerts, offroad=False, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_hardware_alerts_remain_until_offroad():
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  for failed_state, failure_text in ((low, "lost power"), (link_down, "PCIe link unavailable")):
    status, alerts = ChestnutStatus(), Alerts()
    start_model(status, alerts)
    update(status, alerts, offroad=False, state=failed_state)
    update(status, alerts, offroad=False, state=failed_state)
    update(status, alerts, offroad=False, error=True, state=STATE)
    assert failure_text in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
    update(status, alerts, offroad=False, model_recovered=True, state=STATE)
    assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
    update(status, alerts, offroad=True, error=True, state=STATE)
    assert not alerts.active


def test_usb_recovery_does_not_report_power_recovery():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  update(status, alerts, offroad=False, devices=[], state=None)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  update(status, alerts, offroad=False, error=True, state=STATE)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert "USB reconnected" in alerts.values["Offroad_ChestnutNotDetected"][1]
  update(status, alerts, error=True, state=STATE)
  assert not alerts.active


def test_usb_reconnection_does_not_become_power_failure():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  update(status, alerts, offroad=False, devices=[], state=None, error=True)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 0, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, error=True, state=low)
  update(status, alerts, offroad=False, error=True, state=low)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert "USB reconnected" in alerts.values["Offroad_ChestnutNotDetected"][1]
  assert status.hardware_failure is None


@pytest.mark.parametrize(("usb", "powered", "pcie", "expected"), [
  (False, False, False, "Offroad_ChestnutNotDetected"),
  (False, False, True, "Offroad_ChestnutNotDetected"),
  (False, True, False, "Offroad_ChestnutNotDetected"),
  (False, True, True, "Offroad_ChestnutNotDetected"),
  (True, False, False, "Offroad_ChestnutPcieUnavailable"),
  (True, False, True, "Offroad_ChestnutPcieUnavailable"),
  (True, True, False, "Offroad_ChestnutPcieUnavailable"),
  (True, True, True, None),
])
def test_hardware_state_precedence(usb, powered, pcie, expected):
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  state = SimpleNamespace(**(vars(STATE) | {
    "supplyVoltage": 12000 if powered else 3000,
    "supplyFault": not powered,
    "pcieLtssm": 0x78 if pcie else 0,
  }))
  devices = [DEVICE] if usb else []
  update(status, alerts, offroad=False, devices=devices, state=state if usb else None)
  update(status, alerts, offroad=False, devices=devices, state=state if usb else None)
  assert alerts.active == ({expected} if expected is not None else set())


def test_hardware_cause_precedes_resulting_model_error():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, error=True, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  update(status, alerts, offroad=False, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  update(status, alerts, offroad=False, error=True, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "lost power" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False, model_recovered=True)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}


@pytest.mark.parametrize("powered", list(product((False, True), repeat=4)))
def test_power_transition_histories_latch_failures(powered):
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  failed = False
  for available in powered:
    update(status, alerts, offroad=False, state=STATE if available else low)
    failed |= not available
    assert ("Offroad_ChestnutPcieUnavailable" in alerts.active) == failed


def test_overheat_hysteresis_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 100.})))
  assert alerts.active == {"Offroad_ChestnutOverheated"}
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 96.})))
  assert alerts.active == {"Offroad_ChestnutOverheated"}
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 94.})))
  assert not alerts.active
  assert (alerts.triggers["Offroad_ChestnutOverheated"], alerts.clears["Offroad_ChestnutOverheated"]) == (1, 1)


def test_overheat_clears_without_current_telemetry():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 100.})))
  assert alerts.active == {"Offroad_ChestnutOverheated"}
  update(status, alerts, devices=[], state=None)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  assert not status.overheated


@pytest.mark.parametrize(("kwargs", "expected"), [
  ({"devices": []}, None),
  ({"firmware_failed": True, "compiled": False}, "Offroad_ChestnutUpdateFailed"),
  ({"compiled": False, "devices": [DEVICE | {"speedMbps": 480}]}, "Offroad_ChestnutUncompiled"),
  ({"state": SimpleNamespace(**(vars(STATE) | {"tempC": 105.})), "devices": [DEVICE | {"speedMbps": 480}]}, "Offroad_ChestnutOverheated"),
])
def test_alert_precedence(kwargs, expected):
  alerts = update(ChestnutStatus(), **kwargs)
  assert alerts.active == ({expected} if expected is not None else set())


def test_branch_alert_is_suppressed_by_failure():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, branch="master")
  assert alerts.values["Offroad_ChestnutBranch"][0]
  update(status, alerts, branch="master", error=True)
  assert alerts.active == {"Offroad_ChestnutModelError"}
  assert not alerts.values["Offroad_ChestnutBranch"][0]


def test_alert_transition_clears_previous_cause_first():
  status = ChestnutStatus()
  active = set()

  def set_alert(name, enabled, extra=None):
    active.discard(name)
    if enabled:
      active.add(name)
    assert len(active) <= 1

  status.update(True, "master", [DEVICE], False, True, True, False, STATE, set_alert)
  assert active == {"Offroad_ChestnutModelError"}
  status.update(True, "master", [DEVICE], False, True, False, False, STATE, set_alert)
  assert active == {"Offroad_ChestnutBranch"}
