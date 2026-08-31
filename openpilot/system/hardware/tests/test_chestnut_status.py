from itertools import product
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT
from openpilot.system.hardware.chestnut.status import ChestnutStatus, update_modeld_state


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
  was_running = False
  failed = []
  for running, should_run, chestnut_started in states:
    was_running, process_failed = update_modeld_state([process_state(running=running, should_run=should_run)],
                                                      was_running, chestnut_started)
    failed.append(process_failed)
  assert failed == expected


def update(status, alerts=None, *, offroad=True, branch="release-chestnut", devices=None, firmware_failed=False,
           error=False, state=STATE, usb_failed=False, compiled=True):
  alerts = alerts or Alerts()
  status.update(offroad, branch, [DEVICE] if devices is None else devices, firmware_failed, compiled,
                error, state, usb_failed, alerts.set)
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

  with patch("openpilot.system.hardware.chestnut.status.time.monotonic", return_value=status.started + 11):
    update(status, alerts, offroad=False, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}

  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=False, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}
  update(status, alerts, offroad=False)
  assert not alerts.active
  assert alerts.triggers["Offroad_ChestnutNotDetected"] == 1
  assert alerts.clears["Offroad_ChestnutNotDetected"] == 1


def test_peripheral_failure_clears_after_recovery_and_offroad():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  update(status, alerts, offroad=False, state=None, usb_failed=True)
  assert alerts.active == {"Offroad_ChestnutNotDetected"}

  update(status, alerts, offroad=False)
  assert not alerts.active
  update(status, alerts, offroad=False, state=None, usb_failed=True)
  update(status, alerts, offroad=True)
  assert not alerts.active
  update(status, alerts, offroad=False)
  assert not alerts.active


def test_detected_and_not_detected_are_mutually_exclusive():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, branch="master")
  assert alerts.values["Offroad_ChestnutBranch"][0]
  update(status, alerts, offroad=False, branch="master", state=None, usb_failed=True)
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
  update(status, alerts, offroad=False, state=state)


def test_pcie_failure_transition():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=link_down)
  update(status, alerts, offroad=False, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "PCIe link is not up" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False)
  assert not alerts.active


def test_pcie_failure_without_model_state():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False)
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=link_down)
  update(status, alerts, offroad=False, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}


def test_initial_power_absence_transition():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  start_model(status, alerts, low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "power disconnected" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_power_absence_before_model_load():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, error=True, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "power disconnected" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_offroad_power_and_pcie_follow_current_state():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  update(status, alerts)
  assert not alerts.active

  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, state=link_down)
  update(status, alerts, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  update(status, alerts)
  assert not alerts.active


def test_crank_power_loss_and_recovery_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  assert "engine-crank voltage drop" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False, state=STATE)
  assert not alerts.active
  update(status, alerts, offroad=False, state=low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "power lost" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  assert "power restored" not in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  assert alerts.triggers["Offroad_ChestnutPcieUnavailable"] == 2


@pytest.mark.parametrize("powered", list(product((False, True), repeat=4)))
def test_power_transition_histories_report_current_state(powered):
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  for available in powered:
    update(status, alerts, offroad=False, state=STATE if available else low)
    assert ("Offroad_ChestnutPcieUnavailable" in alerts.active) == (not available)


def test_overheat_hysteresis_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 100.})))
  assert alerts.active == {"Offroad_ChestnutOverheated"}
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 96.})))
  assert alerts.active == {"Offroad_ChestnutOverheated"}
  update(status, alerts, state=SimpleNamespace(**(vars(STATE) | {"tempC": 94.})))
  assert not alerts.active
  assert (alerts.triggers["Offroad_ChestnutOverheated"], alerts.clears["Offroad_ChestnutOverheated"]) == (1, 1)


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
