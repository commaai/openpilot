import io
import json
import pickle
import shutil
import struct
import tempfile
from pathlib import Path

from openpilot.common.file_chunker import get_chunk_name, get_manifest_path, open_file_chunked
from openpilot.common.hardware.usb import CHESTNUT_FW_VERSION, CHESTNUT_USB_IDS, USB_DEVICES_PATH

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'
DRIVING_MODEL_PATH = MODELS_DIR / 'driving_tinygrad.pkl'
BIG_DRIVING_MODEL_PATH = MODELS_DIR / 'big_driving_tinygrad.pkl'
BIG_DRIVING_POLICY_PATH = MODELS_DIR / 'big_driving_policy_usb_amd.pkl'


def get_tg_input_devices(process_name: str, usbgpu: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not usbgpu else 'usbgpu']

def modeld_pkl_path(usbgpu: bool):
  return BIG_DRIVING_MODEL_PATH if usbgpu else DRIVING_MODEL_PATH

def modeld_warps_pkl_path(warp_device: str):
  platform = warp_device.split(':', 1)[0].lower()
  if not platform.isalnum():
    raise ValueError(f"invalid warp device {warp_device!r}")
  return MODELS_DIR / f'driving_warps_{platform}.pkl'

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
    while (h := f.read(8)):
      pb = pickle.PickleBuffer(bytearray(struct.unpack('<q', h)[0]))
      f.readinto(pb)
      yield pb
  return pickle.load(io.BytesIO(opcodes), buffers=buffers())

def load_modeld_jits(usbgpu: bool, warp_device: str):
  if usbgpu and split_modeld_compiled(warp_device):
    # The desktop policy is a raw, dev-only override. Open it directly so a
    # stale chunk manifest from an older experiment cannot shadow the rsync.
    with open(BIG_DRIVING_POLICY_PATH, 'rb') as f:
      policy = load_oob(f)
    with open(modeld_warps_pkl_path(warp_device), 'rb') as f:
      warps = load_oob(f)
    if set(policy) != {'metadata', 'run_policy'}:
      raise ValueError(f"unexpected policy artifact keys: {set(policy)!r}")
    if not warps or not all(isinstance(key, tuple) and len(key) == 2 for key in warps):
      raise ValueError(f"unexpected warp artifact keys: {set(warps)!r}")
    return policy | warps

  model_path = BIG_DRIVING_MODEL_PATH if usbgpu else DRIVING_MODEL_PATH
  with open_file_chunked(model_path) as f:
    return load_oob(f)

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

def modeld_artifact_exists(path: Path) -> bool:
  try:
    manifest = Path(get_manifest_path(path))
    if not manifest.is_file():
      return path.is_file()
    num_chunks = int(manifest.read_text())
    return num_chunks > 0 and all(Path(get_chunk_name(path, i, num_chunks)).is_file() for i in range(num_chunks))
  except (OSError, ValueError):
    return False

def split_modeld_compiled(warp_device: str) -> bool:
  return BIG_DRIVING_POLICY_PATH.is_file() and modeld_warps_pkl_path(warp_device).is_file()

def usbgpu_compiled() -> bool:
  try:
    warp_device = get_tg_input_devices('openpilot.selfdrive.modeld.modeld', True)['WARP_DEV']
    return modeld_artifact_exists(BIG_DRIVING_MODEL_PATH) or split_modeld_compiled(warp_device)
  except (KeyError, OSError, TypeError, ValueError):
    return False
