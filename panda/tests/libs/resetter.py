import time
import usb1


class Resetter():
  def __init__(self):
    self._handle = None
    self.connect()

  def __enter__(self):
    return self

  def __exit__(self, *args):
    self.close()

  def close(self):
    self._handle.close()
    self._context.close()
    self._handle = None

  def connect(self):
    if self._handle:
      self.close()

    self._handle = None

    self._context = usb1.USBContext()
    self._context.open()
    for device in self._context.getDeviceList(skip_on_error=True):
      if device.getVendorID() == 0xbbaa and device.getProductID() == 0xddc0:
        try:
          self._handle = device.open()
          self._handle.claimInterface(0)
          break
        except Exception as e:
          print(e)
    assert self._handle

  def enable_power(self, port, enabled):
    self._handle.controlWrite((usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE), 0xff, port, enabled, b'')

  def enable_boot(self, enabled):
    self._handle.controlWrite((usb1.ENDPOINT_OUT | usb1.TYPE_VENDOR | usb1.RECIPIENT_DEVICE), 0xff, 0, enabled, b'')

  def cycle_power(self, dfu=False, ports=None):
    if ports is None:
      ports = [1, 2, 3]

    self.enable_boot(dfu)
    for port in ports:
      self.enable_power(port, False)
    time.sleep(0.05)

    for port in ports:
      self.enable_power(port, True)
    time.sleep(0.05)
    self.enable_boot(False)
    time.sleep(0.12)  # takes the kernel this long to detect the disconnect
