import struct
from contextlib import suppress

import usb1

import openpilot.cereal.messaging as messaging
from openpilot.common.hardware.usb import CHESTNUT_USB_IDS


USB_TIMEOUT_MS = 100
PCIE_LTSSM_ADDRESS = 0xB450


class ChestnutUsb:
  def __init__(self):
    self.context: usb1.USBContext | None = None
    self.handle = None

  def close(self) -> None:
    handle, context = self.handle, self.context
    self.handle = None
    self.context = None
    with suppress(Exception):
      if handle is not None:
        handle.close()
    with suppress(Exception):
      if context is not None:
        context.close()

  def connect(self) -> bool:
    if self.handle is not None:
      return True

    context = usb1.USBContext()
    for vendor_id, product_id in CHESTNUT_USB_IDS:
      handle = context.openByVendorIDAndProductID(vendor_id, product_id, skip_on_error=True)
      if handle is not None:
        self.context = context
        self.handle = handle
        return True
    context.close()
    return False

  def _read(self, request: int, value: int, length: int) -> bytes:
    if self.handle is None:
      raise usb1.USBErrorNoDevice
    raw = bytes(self.handle.controlRead(0xC0, request, value, 0, length, timeout=USB_TIMEOUT_MS))
    if len(raw) != length:
      raise ValueError(f"short chestnut USB response: {len(raw)}/{length}")
    return raw

  def read_ina(self) -> tuple[int, int, bool]:
    return struct.unpack('<Hh?', self._read(0xC0, 0, 5))

  def read_pcie_ltssm(self) -> int:
    return self._read(0xE4, PCIE_LTSSM_ADDRESS, 1)[0]


class ChestnutMonitoring:
  def __init__(self, usb: ChestnutUsb | None = None):
    self.usb = usb or ChestnutUsb()
    self.gpu_state = None
    self.seen = False
    self.enabled = False
    self.usb_failed = False

  def set_enabled(self, enabled: bool) -> None:
    if self.enabled == enabled:
      return
    self.enabled = enabled
    self.usb.close()
    self.usb_failed = False

  def retry(self) -> None:
    self.usb_failed = False

  def update_gpu_state(self, sm: messaging.SubMaster) -> None:
    if sm.updated['chestnutGpuState']:
      self.gpu_state = sm['chestnutGpuState'] if sm.valid['chestnutGpuState'] else None
    elif not sm.alive['chestnutGpuState']:
      self.gpu_state = None

  def update(self, sm: messaging.SubMaster, model_loading: bool = False):
    self.update_gpu_state(sm)
    return self.build_message(model_loading)

  def build_message(self, model_loading: bool = False):
    if not self.enabled:
      return None

    msg = messaging.new_message('chestnutState')
    if self.gpu_state is not None:
      msg.chestnutState = self.gpu_state
    state = msg.chestnutState

    if self.usb_failed:
      return msg if self.seen else None

    try:
      if not self.usb.connect():
        self.usb_failed = True
        return msg if self.seen else None

      self.seen = True
      voltage, current, fault = self.usb.read_ina()
      pcie_ltssm = self.usb.read_pcie_ltssm()
      state.supplyVoltage = voltage
      state.supplyCurrent = current
      state.supplyFault = fault
      state.pcieLtssm = pcie_ltssm
      msg.valid = True
    except Exception as e:
      if not model_loading or not isinstance(e, usb1.USBErrorTimeout):
        self.usb.close()
        self.usb_failed = True
    return msg
