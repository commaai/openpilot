import time

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, is_chestnut_usb_id
from openpilot.selfdrive.modeld.helpers import CHESTNUT_PCIE_READY, CHESTNUT_POWERED_VOLTAGE


CHESTNUT_RELEASE_BRANCHES = ("release-chestnut", "release-chestnut-staging")
GPU_TEMP_LIMIT = 100.
MEMORY_TEMP_LIMIT = 95.
TEMP_HYSTERESIS = 5.
STARTUP_STABILIZATION_TIME = 5.
USB_RECONNECTED_ALERT = ("Offroad_ChestnutNotDetected", "Chestnut USB reconnected. Restart the car to retry.")


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
    self.onroad_since = self.started
    self.offroad = True
    self.offroad_after_drive = False
    self.power_failures = 0
    self.power_seen = False
    self.link_failures = 0
    self.overheated = False
    self.usb_seen = False
    self.usb_failed = False
    self.hardware_failure: str | None = None
    self.startup_failure: str | None = None
    self.usb_failure = False

  def update(self, offroad: bool, branch: str, usb_state: list[dict], firmware_failed: bool, model_compiled: bool,
             model_error: bool, model_recovered: bool, state, set_alert) -> str | None:
    detected = [d for d in usb_state if is_chestnut_usb_id(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_chestnut_usb_id(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == CHESTNUT_USB_PRODUCT

    if self.offroad and not offroad:
      self.onroad_since = time.monotonic()
      self.offroad_after_drive = False
      self.power_failures = 0
      self.power_seen = False
      self.link_failures = 0
      self.usb_failed = False
      self.hardware_failure = None
      self.startup_failure = None
      self.usb_failure = False
    elif not self.offroad and offroad:
      self.offroad_after_drive = True
      self.hardware_failure = None

    self.usb_seen |= firmware_ok
    self.usb_failed = self.usb_seen and not firmware_ok

    if self.usb_failed or state is None:
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
    missing = self.usb_failed or (release and time.monotonic() - self.started > 10. and len(detected) != 1)
    slow_usb = len(devices) == 1 and devices[0]["speedMbps"] < 5000
    update_failed = firmware_failed
    compiled = firmware_ok and model_compiled
    uncompiled = firmware_ok and not compiled
    software_failed = model_error and compiled
    show_setup_alerts = not offroad or not self.offroad_after_drive

    stabilizing = not offroad and time.monotonic() - self.onroad_since < STARTUP_STABILIZATION_TIME
    if stabilizing:
      if self.power_failures > 0:
        self.startup_failure = "power"
      elif self.link_failures > 0 and self.startup_failure is None:
        self.startup_failure = "pcie"
    usb_model_failure = self.usb_failure and model_error
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

    # Only report one model failure cause, ordered from direct setup/hardware failures to software failures.
    causes = (
      ("Offroad_ChestnutNotDetected", missing, missing_alert),
      ("Offroad_ChestnutUpdateFailed", update_failed and show_setup_alerts, None),
      ("Offroad_ChestnutUncompiled", uncompiled and show_setup_alerts, None),
      ("Offroad_ChestnutPcieUnavailable", hardware_failed, pcie_alert),
      ("Offroad_ChestnutOverheated", self.overheated, f"{state.tempC:.0f} °C" if state is not None else None),
      ("Offroad_ChestnutUsbSlow", slow_usb and show_setup_alerts, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None),
    )
    current_cause = next(((name, text) for name, active, text in causes if active), None)
    if current_cause is not None and current_cause[0] == "Offroad_ChestnutNotDetected":
      self.usb_failure = True
    elif model_recovered:
      self.startup_failure = None
      self.usb_failure = False

    retained_cause = USB_RECONNECTED_ALERT if model_error and self.usb_failure and current_cause is None and not offroad else None
    alerts = []
    for name, active, text in causes:
      if retained_cause is not None and name == retained_cause[0]:
        active, text = True, retained_cause[1]
      alerts.append((name, active, text))
    alerts.append(("Offroad_ChestnutModelError", software_failed and not (stabilizing and self.startup_failure is not None) and
                   not self.usb_failure and not hardware_failed and show_setup_alerts, None))
    active_alert = next((name for name, active, _ in alerts if active), None)
    branch_alert = "Offroad_ChestnutBranch" if not release and len(devices) == 1 and active_alert is None else None
    selected_alert = active_alert or branch_alert

    # Clear the previous cause before setting the current one so readers never observe both.
    for name in ("Offroad_ChestnutBranch", *(name for name, _, _ in alerts)):
      if name != selected_alert:
        set_alert(name, False)
    if selected_alert is not None:
      extra_text = next((text for name, _, text in alerts if name == selected_alert), None)
      set_alert(selected_alert, True, extra_text)
    self.offroad = offroad
    return selected_alert
