import io
import json
import pickle
import shutil
import struct
import tempfile
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openpilot.common.file_chunker import get_manifest_path
from openpilot.common.hardware.usb import CHESTNUT_FW_VERSION, CHESTNUT_USB_IDS, USB_DEVICES_PATH

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'


def get_tg_input_devices(process_name: str, usbgpu: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not usbgpu else 'usbgpu']

def modeld_pkl_path(usbgpu: bool):
  prefix = 'big_' if usbgpu else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def dump_oob(obj, f):
  with tempfile.TemporaryFile(dir=".") as tmp:
    def buffer_callback(pb: pickle.PickleBuffer):
      m = pb.raw()
      tmp.write(struct.pack('<q', m.nbytes))
      tmp.write(m)
      pb.release() # keep peak ram at ~1 buffer
    stream = io.BytesIO()
    pickle.Pickler(stream, protocol=5, buffer_callback=buffer_callback).dump(obj)
    opcodes = stream.getvalue()
    f.write(struct.pack('<q', len(opcodes)))
    f.write(opcodes)
    tmp.seek(0)
    shutil.copyfileobj(tmp, f)

def load_oob(f):
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    def read_buffer():
      return pickle.PickleBuffer(f.read(struct.unpack('<q', h)[0])) if (h := f.read(8)) else None
    with ThreadPoolExecutor(max_workers=1) as pool:
      pending = deque(pool.submit(read_buffer) for _ in range(2))
      while (pb := pending.popleft().result()) is not None:
        pending.append(pool.submit(read_buffer))
        yield pb
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())

def usbgpu_present() -> bool:
  for d in USB_DEVICES_PATH.glob("*"):
    try:
      usb_id = (int((d / "idVendor").read_text(), 16), int((d / "idProduct").read_text(), 16))
      product = (d / "product").read_text().strip()
      if usb_id in CHESTNUT_USB_IDS and product == f"custom {CHESTNUT_FW_VERSION}-CLEAN":
        return True
    except Exception:
      pass
  return False

def usbgpu_compiled() -> bool:
  return Path(get_manifest_path(modeld_pkl_path(usbgpu=True))).is_file()
