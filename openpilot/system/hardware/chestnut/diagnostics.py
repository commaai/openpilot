import os
import subprocess
import sys

from openpilot.common.basedir import BASEDIR
from openpilot.common.hardware.usb import get_usb_state, is_chestnut_usb_device
from openpilot.system.hardware.chestnut.flash import link_up


def _chestnut():
  devices = [d for d in get_usb_state() if is_chestnut_usb_device(d["vendorId"], d["productId"])]
  return devices[0] if len(devices) == 1 else None


def check_chestnut() -> str | None:
  device = _chestnut()
  if device is None:
    return "USB not connected"
  if device["speedMbps"] < 5000:
    return f"USB link {device['speedMbps']} Mbps"
  link_errors = device["linkErrorCount"]

  if not link_up():
    return "12V not connected"

  env = {**os.environ, "DEV": "USB+AMD:LLVM", "GMMU": "0", "PYTHONPATH": os.path.join(BASEDIR, "tinygrad_repo")}
  code = "from tinygrad import Tensor; x = Tensor.rand(1 << 20).realize(); [x.numpy() for _ in range(8)]"
  try:
    result = subprocess.run([sys.executable, "-c", code], env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                            timeout=15., check=False)
  except subprocess.TimeoutExpired:
    return "GPU check timed out"
  if result.returncode != 0:
    return "GPU incompatible"

  device = _chestnut()
  return "USB link errors" if device is None or device["linkErrorCount"] > link_errors else None
