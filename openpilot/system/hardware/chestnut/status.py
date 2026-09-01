import time

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, is_chestnut_usb_id
from openpilot.selfdrive.modeld.helpers import CHESTNUT_PCIE_READY, CHESTNUT_POWERED_VOLTAGE


CHESTNUT_RELEASE_BRANCHES = ("release-chestnut", "release-chestnut-staging")
GPU_TEMP_LIMIT = 100.
MEMORY_TEMP_LIMIT = 95.
TEMP_HYSTERESIS = 5.


class ChestnutStatus:
  def __init__(self):
    self.started = time.monotonic()
    self.onroad_ts = self.started
    self.power_lost_count = 0
    self.power_seen = False
    self.link_fail_count = 0
    self.overheated = False
    self.usb_seen = False
    self.failure: str | None = None
    self.usb_lost = False

  def start_drive(self) -> None:
    self.onroad_ts = time.monotonic()
    self.power_lost_count = 0
    self.power_seen = False
    self.link_fail_count = 0
    self.failure = None
    self.usb_lost = False

  def end_drive(self) -> None:
    self.failure = None
    self.overheated = False

  def update_hardware(self, firmware_ok: bool, state) -> bool:
    self.usb_seen |= firmware_ok
    usb_disconnected = self.usb_seen and not firmware_ok
    if usb_disconnected or state is None:
      self.power_lost_count = 0
      self.power_seen = False
      self.link_fail_count = 0
    else:
      power_lost = state.supplyFault or state.supplyVoltage < CHESTNUT_POWERED_VOLTAGE
      self.power_lost_count = self.power_lost_count + 1 if power_lost else 0
      self.power_seen |= not power_lost
      self.link_fail_count = self.link_fail_count + 1 if state.pcieLtssm != CHESTNUT_PCIE_READY else 0

    if state is not None:
      gpu_limit = GPU_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      memory_limit = MEMORY_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      self.overheated = state.tempC >= gpu_limit or state.memoryTempC >= memory_limit
    return usb_disconnected

  def update_failure(self, offroad: bool, model_unavailable: bool, model_error: bool) -> tuple[bool, bool]:
    stabilizing = not offroad and time.monotonic() - self.onroad_ts < 5.
    if stabilizing and self.failure is None:
      if self.power_lost_count > 0:
        self.failure = "power"
      elif self.link_fail_count > 0:
        self.failure = "pcie"

    usb_failed = self.usb_lost and model_unavailable
    power_failed = (self.power_lost_count > 0 and (self.power_seen or self.power_lost_count >= 2 or model_error) and
                    not offroad and not stabilizing and not usb_failed)
    pcie_failed = self.link_fail_count >= 2 and not power_failed and not offroad and not stabilizing and not usb_failed
    if power_failed:
      self.failure = "power"
    elif pcie_failed and self.failure is None:
      self.failure = "pcie"
    hardware_failed = (self.failure is not None and not offroad and not usb_failed and
                       (model_unavailable or power_failed or pcie_failed))
    return stabilizing, hardware_failed

  def update(self, *, offroad: bool, show_setup_alerts: bool, branch: str, usb_state: list[dict], firmware_failed: bool, model_compiled: bool,
             model_active: bool | None, model_error: bool, state, set_alert) -> bool:
    detected = [d for d in usb_state if is_chestnut_usb_id(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_chestnut_usb_id(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == CHESTNUT_USB_PRODUCT

    usb_disconnected = self.update_hardware(firmware_ok, state)

    release = branch in CHESTNUT_RELEASE_BRANCHES
    missing = usb_disconnected or (release and time.monotonic() - self.started > 10. and len(detected) != 1)
    compiled = firmware_ok and model_compiled
    update_failed = firmware_failed and show_setup_alerts
    uncompiled = firmware_ok and not compiled and show_setup_alerts
    slow_usb = len(devices) == 1 and devices[0]["speedMbps"] < 5000 and show_setup_alerts

    if missing:
      self.usb_lost = True

    model_unavailable = model_active is False
    stabilizing, hardware_failed = self.update_failure(offroad, model_unavailable, model_error)
    usb_reconnected = (model_unavailable and self.usb_lost and not offroad and
                       not any((missing, update_failed, uncompiled, hardware_failed, self.overheated, slow_usb)))
    model_failed = (model_error and compiled and not (stabilizing and self.failure is not None) and
                    not self.usb_lost and not hardware_failed and show_setup_alerts)
    alerts = (
      ("Offroad_ChestnutAbsent", missing, None),
      ("Offroad_ChestnutUsbReconnected", usb_reconnected, None),
      ("Offroad_ChestnutUpdateFailed", update_failed, None),
      ("Offroad_ChestnutUncompiled", uncompiled, None),
      ("Offroad_ChestnutPowerLost", hardware_failed and self.failure == "power", None),
      ("Offroad_ChestnutPcieUnavailable", hardware_failed and self.failure == "pcie", None),
      ("Offroad_ChestnutOverheated", self.overheated, f"{state.tempC:.0f} °C" if state is not None else None),
      ("Offroad_ChestnutUsbSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None),
      ("Offroad_ChestnutModelError", model_failed, None),
    )
    for name, active, text in alerts:
      set_alert(name, active, text)
    set_alert("Offroad_ChestnutBranch", not release and len(devices) == 1 and not any(active for _, active, _ in alerts))
    return missing or hardware_failed
