import time

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, is_chestnut_usb_id
from openpilot.selfdrive.modeld.helpers import CHESTNUT_PCIE_READY, CHESTNUT_POWERED_VOLTAGE


CHESTNUT_RELEASE_BRANCHES = ("release-chestnut", "release-chestnut-staging")
GPU_TEMP_LIMIT = 100.
MEMORY_TEMP_LIMIT = 95.
TEMP_HYSTERESIS = 5.


def update_modeld_state(processes, modeld_seen: bool, chestnut_started: bool) -> tuple[bool, bool]:
  modeld = next((p for p in processes if p.name == "modeld"), None)
  if modeld is not None and not modeld.shouldBeRunning:
    return False, False
  modeld_seen |= modeld is not None and modeld.running
  failed = modeld is not None and chestnut_started and modeld_seen and not modeld.running
  return modeld_seen, failed


class ChestnutStatus:
  def __init__(self):
    self.started = time.monotonic()
    self.onroad_ts = self.started
    self.offroad = True
    self.post_drive = False
    self.power_failures = 0
    self.power_seen = False
    self.link_failures = 0
    self.overheated = False
    self.usb_seen = False
    self.usb_disconnected = False
    self.hardware_failure: str | None = None
    self.startup_failure: str | None = None
    self.usb_lost = False

  def update(self, offroad: bool, branch: str, usb_state: list[dict], firmware_failed: bool, model_compiled: bool,
             model_error: bool, model_recovered: bool, state, set_alert) -> str | None:
    detected = [d for d in usb_state if is_chestnut_usb_id(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_chestnut_usb_id(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == CHESTNUT_USB_PRODUCT

    if self.offroad and not offroad:
      self.onroad_ts = time.monotonic()
      self.post_drive = False
      self.power_failures = 0
      self.power_seen = False
      self.link_failures = 0
      self.usb_disconnected = False
      self.hardware_failure = None
      self.startup_failure = None
      self.usb_lost = False
    elif not self.offroad and offroad:
      self.post_drive = True
      self.hardware_failure = None

    self.usb_seen |= firmware_ok
    self.usb_disconnected = self.usb_seen and not firmware_ok

    if self.usb_disconnected or state is None:
      self.power_failures = 0
      self.power_seen = False
      self.link_failures = 0
    else:
      power_lost = state.supplyFault or state.supplyVoltage < CHESTNUT_POWERED_VOLTAGE
      self.power_failures = self.power_failures + 1 if power_lost else 0
      self.power_seen |= not power_lost
      self.link_failures = self.link_failures + 1 if state.pcieLtssm != CHESTNUT_PCIE_READY else 0

    if state is None:
      self.overheated = False
    else:
      gpu_limit = GPU_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      memory_limit = MEMORY_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      self.overheated = state.tempC >= gpu_limit or state.memoryTempC >= memory_limit

    release = branch in CHESTNUT_RELEASE_BRANCHES
    missing = self.usb_disconnected or (release and time.monotonic() - self.started > 10. and len(detected) != 1)
    compiled = firmware_ok and model_compiled
    software_failed = model_error and compiled
    show_setup_alerts = not offroad or not self.post_drive
    update_failed = firmware_failed and show_setup_alerts
    uncompiled = firmware_ok and not compiled and show_setup_alerts
    slow_usb = len(devices) == 1 and devices[0]["speedMbps"] < 5000 and show_setup_alerts

    stabilizing = not offroad and time.monotonic() - self.onroad_ts < 5.
    if stabilizing:
      if self.power_failures > 0:
        self.startup_failure = "power"
      elif self.link_failures > 0 and self.startup_failure is None:
        self.startup_failure = "pcie"
    usb_model_failure = self.usb_lost and model_error
    power_failed = (self.power_failures > 0 and (self.power_seen or self.power_failures >= 2 or model_error) and
                    not offroad and not stabilizing and not usb_model_failure)
    pcie_failed = self.link_failures >= 2 and not power_failed and not offroad and not stabilizing and not usb_model_failure
    if power_failed:
      self.hardware_failure = "power"
    elif model_error and self.startup_failure is not None and not usb_model_failure:
      self.hardware_failure = self.startup_failure
    elif pcie_failed and self.hardware_failure is None:
      self.hardware_failure = "pcie"
    hardware_failed = self.hardware_failure is not None and not offroad
    pcie_alert = ("Chestnut lost power. Check 12 V connection. This may be caused by an engine-crank voltage drop. Restart the car to retry."
                  if self.hardware_failure == "power" else
                  "Chestnut PCIe link unavailable. Check the GPU is securely seated. Restart the car to retry.")
    missing_alert = "Chestnut USB disconnected. Check USB connection. Restart the car to retry."

    if missing:
      self.usb_lost = True
    elif model_recovered:
      self.startup_failure = None
      self.usb_lost = False

    usb_reconnected = (model_error and self.usb_lost and not offroad and
                       not any((missing, update_failed, uncompiled, hardware_failed, self.overheated, slow_usb)))
    usb_alert = "Chestnut USB reconnected. Restart the car to retry." if usb_reconnected else missing_alert
    alerts = (
      ("Offroad_ChestnutNotDetected", missing or usb_reconnected, usb_alert),
      ("Offroad_ChestnutUpdateFailed", update_failed, None),
      ("Offroad_ChestnutUncompiled", uncompiled, None),
      ("Offroad_ChestnutPcieUnavailable", hardware_failed, pcie_alert),
      ("Offroad_ChestnutOverheated", self.overheated, f"{state.tempC:.0f} °C" if state is not None else None),
      ("Offroad_ChestnutUsbSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None),
      ("Offroad_ChestnutModelError", software_failed and not (stabilizing and self.startup_failure is not None) and
       not self.usb_lost and not hardware_failed and show_setup_alerts, None),
    )
    selected = next(((name, text) for name, active, text in alerts if active), None)
    if selected is None and not release and len(devices) == 1:
      selected = ("Offroad_ChestnutBranch", None)
    selected_alert = selected[0] if selected is not None else None

    for name in ("Offroad_ChestnutBranch", *(name for name, _, _ in alerts)):
      if name != selected_alert:
        set_alert(name, False)
    if selected is not None:
      set_alert(selected[0], True, selected[1])
    self.offroad = offroad
    return selected_alert
