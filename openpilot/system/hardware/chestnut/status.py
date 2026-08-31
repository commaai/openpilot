import time

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, is_chestnut_usb_id
from openpilot.selfdrive.modeld.helpers import CHESTNUT_PCIE_READY, CHESTNUT_POWERED_VOLTAGE


CHESTNUT_RELEASE_BRANCHES = ("release-chestnut", "release-chestnut-staging")
GPU_TEMP_LIMIT = 100.
MEMORY_TEMP_LIMIT = 95.
TEMP_HYSTERESIS = 5.


def update_modeld_state(processes, was_running: bool, chestnut_started: bool) -> tuple[bool, bool]:
  modeld = next((p for p in processes if p.name == "modeld"), None)
  if modeld is not None and not modeld.shouldBeRunning:
    return False, False
  was_running |= modeld is not None and modeld.running
  failed = modeld is not None and chestnut_started and was_running and not modeld.running
  return was_running, failed


class ChestnutStatus:
  def __init__(self):
    self.started = time.monotonic()
    self.offroad = True
    self.pcie_failed = False
    self.power_seen = False
    self.power_unavailable = False
    self.power_lost = False
    self.link_failures = 0
    self.overheated = False
    self.usb_seen = False
    self.usb_failed = False

  def update(self, offroad: bool, branch: str, usb_state: list[dict], firmware_failed: bool, model_compiled: bool,
             model_error: bool, state, usb_failed: bool, set_alert) -> None:
    detected = [d for d in usb_state if is_chestnut_usb_id(d["vendorId"], d["productId"], include_bootloader=True)]
    devices = [d for d in detected if is_chestnut_usb_id(d["vendorId"], d["productId"])]
    firmware_ok = len(devices) == 1 and devices[0]["product"] == CHESTNUT_USB_PRODUCT

    if self.offroad and not offroad:
      self.pcie_failed = False
      self.power_seen = False
      self.power_unavailable = False
      self.power_lost = False
      self.link_failures = 0
      self.usb_seen = firmware_ok
      self.usb_failed = False

    self.usb_seen |= firmware_ok
    self.usb_failed = self.usb_seen and (not firmware_ok or usb_failed)

    if state is not None:
      powered = state.supplyVoltage >= CHESTNUT_POWERED_VOLTAGE
      power_lost = state.supplyFault or not powered
      if power_lost and not self.power_lost:
        self.power_unavailable = not self.power_seen
      self.power_seen |= powered

    if state is not None:
      self.link_failures = self.link_failures + 1 if state.pcieLtssm != CHESTNUT_PCIE_READY else 0
      self.pcie_failed = self.link_failures >= 2 or power_lost
      self.power_lost = power_lost
    if self.usb_failed:
      self.pcie_failed = False
      self.power_seen = False
      self.power_unavailable = False
      self.power_lost = False

    if state is not None:
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

    if self.power_lost:
      pcie_alert = ("Chestnut power disconnected. Check 12V connection, then cycle ignition." if self.power_unavailable else
                    "Chestnut power lost. Possibly caused by an engine-crank voltage drop. Check 12V connection, then cycle ignition.")
    else:
      pcie_alert = "Chestnut GPU unavailable. PCIe link is not up. Check the GPU is securely seated."

    # Only report one model failure cause, ordered from direct setup/hardware failures to software failures.
    alerts = (
      ("Offroad_ChestnutNotDetected", missing, None),
      ("Offroad_ChestnutUpdateFailed", update_failed, None),
      ("Offroad_ChestnutUncompiled", uncompiled, None),
      ("Offroad_ChestnutPcieUnavailable", self.pcie_failed, pcie_alert),
      ("Offroad_ChestnutOverheated", self.overheated, f"{state.tempC:.0f} °C" if state is not None else None),
      ("Offroad_ChestnutUsbSlow", slow_usb, f"{devices[0]['speedMbps']} Mbps" if slow_usb else None),
      ("Offroad_ChestnutModelError", software_failed, None),
    )
    active_alert = next((name for name, active, _ in alerts if active), None)

    set_alert("Offroad_ChestnutBranch", not release and len(devices) == 1 and active_alert is None)
    for name, _, extra_text in alerts:
      set_alert(name, name == active_alert, extra_text)
    self.offroad = offroad
