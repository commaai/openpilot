from types import SimpleNamespace
from unittest.mock import patch

import pytest

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT
from openpilot.system.hardware.chestnut.status import ChestnutStatus


FAILURE_ALERTS = {
  "Offroad_ChestnutNotDetected", "Offroad_ChestnutOverheated",
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


def update(status, alerts=None, *, offroad=True, branch="release-chestnut", devices=None, firmware_failed=False,
           active=False, state=STATE, usb_failed=False, compiled=True):
  alerts = alerts or Alerts()
  with patch("openpilot.system.hardware.chestnut.status.chestnut_compiled", return_value=compiled):
    status.update(offroad, branch, [DEVICE] if devices is None else devices, firmware_failed,
                  active, state, usb_failed, alerts.set)
  assert len(alerts.active) <= 1
  return alerts


def test_usb_cold_boot_and_pull_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  with patch("openpilot.system.hardware.chestnut.status.time.monotonic", return_value=status.started + 9):
    update(status, alerts, devices=[])
  assert not alerts.active
  with patch("openpilot.system.hardware.chestnut.status.time.monotonic", return_value=status.started + 11):
    update(status, alerts, devices=[])
  assert alerts.active == {"Offroad_ChestnutNotDetected"}

  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, active=None)
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


def start_model(status, alerts, state=STATE):
  update(status, alerts, offroad=False, active=None, state=state)
  update(status, alerts, offroad=False, active=True, state=state)


def test_pcie_failure_transition():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=link_down)
  update(status, alerts, offroad=False, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "PCIe link is not up" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_pcie_failure_without_loading_sample():
  status, alerts = ChestnutStatus(), Alerts()
  update(status, alerts, offroad=False, active=True)
  link_down = SimpleNamespace(**(vars(STATE) | {"pcieLtssm": 0}))
  update(status, alerts, offroad=False, active=True, state=link_down)
  update(status, alerts, offroad=False, active=True, state=link_down)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}


def test_initial_power_absence_transition():
  status, alerts = ChestnutStatus(), Alerts()
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  start_model(status, alerts, low)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "power disconnected" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]


def test_crank_power_loss_and_recovery_transitions():
  status, alerts = ChestnutStatus(), Alerts()
  start_model(status, alerts)
  low = SimpleNamespace(**(vars(STATE) | {"supplyVoltage": 3000, "supplyFault": True, "pcieLtssm": 0}))
  update(status, alerts, offroad=False, state=low)
  assert "engine-crank voltage drop" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  update(status, alerts, offroad=False, state=STATE)
  assert alerts.active == {"Offroad_ChestnutPcieUnavailable"}
  assert "power restored" in alerts.values["Offroad_ChestnutPcieUnavailable"][1]
  assert alerts.triggers["Offroad_ChestnutPcieUnavailable"] == 1


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
