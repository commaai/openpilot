import mmap
import pickle
import struct
from pathlib import Path

from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def load_oob(f):
  data = memoryview(mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY))
  opcode_size = struct.unpack_from('<q', data)[0]
  def buffers():
    offset = 8 + opcode_size
    while offset < len(data):
      size = struct.unpack_from('<q', data, offset)[0]
      offset += 8
      if offset + size > len(data):
        raise EOFError("incomplete model buffer")
      yield pickle.PickleBuffer(data[offset:offset + size])
      offset += size
  return pickle.loads(data[8:8 + opcode_size], buffers=buffers())

def chestnut_present() -> bool:
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if is_chestnut_usb_id(*usb_id) and product == CHESTNUT_USB_PRODUCT:
        return True
    except Exception:
      pass
  return False

def chestnut_compiled() -> bool:
  path = modeld_pkl_path(chestnut=True)
  return path.is_file() and all(
    (MODELS_DIR / f'big_driving_warp_{size}_tinygrad.pkl').is_file() for size in ('1344x760', '1928x1208'))
