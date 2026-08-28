import time

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, is_chestnut_usb_device
from openpilot.selfdrive.modeld.helpers import chestnut_compiled


CHESTNUT_RELEASE_BRANCHES = ("release-chestnut", "release-chestnut-staging")
CHESTNUT_POWERED_VOLTAGE = 5000
GPU_TEMP_LIMIT = 100.
MEMORY_TEMP_LIMIT = 95.
TEMP_HYSTERESIS = 5.
FAN_START_GPU_TEMP = 60.
FAN_STOP_GPU_TEMP = 50.
FAN_START_MEMORY_TEMP = 70.
FAN_STOP_MEMORY_TEMP = 60.
FAN_STALLED_RPM = 250


class ChestnutStatus:
  def __init__(self):
    self.started = time.monotonic()
    self.offroad = True
    self.pcie_failed = False
    self.power_lost = False
    self.power_restored = False
    self.link_failures = 0
    self.model_loading_seen = False
    self.model_attempted = False
    self.overheated = False
    self.fans_obstructed = False
    self.usb_seen = False
    self.usb_failed = False

  def update(self, offroad: bool, branch: str, usb_state: list[dict], firmware_failed: bool,
             model_loading: bool, model_active: bool | None, state, set_alert) -> None:
    detected = [d for d in usb_state if is_chestnut_usb_device(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_chestnut_usb_device(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == CHESTNUT_USB_PRODUCT

    if self.offroad and not offroad:
      self.pcie_failed = False
      self.power_lost = False
      self.power_restored = False
      self.link_failures = 0
      self.model_loading_seen = False
      self.model_attempted = False
      self.usb_seen = firmware_ok
      self.usb_failed = False

    self.model_loading_seen |= model_loading
    self.model_attempted |= self.model_loading_seen and not model_loading and model_active is not None

    if not offroad and self.usb_seen and not firmware_ok:
      self.usb_failed = True

    if not offroad and self.model_attempted and state is not None:
      power_lost = state.supplyFault or state.supplyVoltage < CHESTNUT_POWERED_VOLTAGE
      self.link_failures = self.link_failures + 1 if state.pcieLtssm != 0x78 else 0
      self.pcie_failed |= self.link_failures >= 2 or power_lost
      self.power_lost |= power_lost

    if self.pcie_failed and self.power_lost and state is not None:
      self.power_restored |= not state.supplyFault and state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE
    if self.usb_failed:
      self.pcie_failed = False
      self.power_lost = False
      self.power_restored = False

    if state is not None:
      gpu_limit = GPU_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      memory_limit = MEMORY_TEMP_LIMIT - (TEMP_HYSTERESIS if self.overheated else 0.)
      self.overheated = state.tempC >= gpu_limit or state.memoryTempC >= memory_limit
      fan_hot = (state.tempC >= (FAN_STOP_GPU_TEMP if self.fans_obstructed else FAN_START_GPU_TEMP) or
                 state.memoryTempC >= (FAN_STOP_MEMORY_TEMP if self.fans_obstructed else FAN_START_MEMORY_TEMP))
      self.fans_obstructed = fan_hot and state.fanSpeedRpm < FAN_STALLED_RPM

    release = branch in CHESTNUT_RELEASE_BRANCHES
    missing = self.usb_failed or (offroad and release and time.monotonic() - self.started > 10. and len(detected) != 1)
    slow_usb = offroad and len(devices) == 1 and devices[0]["speedMbps"] < 5000
    set_alert("Offroad_ChestnutBranch", not release and len(devices) == 1)
    set_alert("Offroad_ChestnutNotDetected", missing)
    set_alert("Offroad_ChestnutFansObstructed", self.fans_obstructed)
    set_alert("Offroad_ChestnutOverheated", self.overheated, f"{state.tempC:.0f} °C" if state is not None else None)
    set_alert("Offroad_ChestnutUsbSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None)
    if self.power_lost:
      pcie_action = ("12V power restored. Cycle ignition to reload the model." if self.power_restored else
                     "12V power was interrupted, possibly by engine start-stop. Check 12V, then cycle ignition to reload the model.")
    else:
      pcie_action = "Check 12V connection."
    set_alert("Offroad_ChestnutPcieUnavailable", self.pcie_failed, pcie_action)
    set_alert("Offroad_ChestnutUncompiled", offroad and firmware_ok and not chestnut_compiled())
    set_alert("Offroad_ChestnutUpdateFailed", offroad and firmware_failed)
    self.offroad = offroad
