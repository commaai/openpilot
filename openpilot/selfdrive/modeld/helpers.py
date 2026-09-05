import io
import json
import pickle
import shutil
import struct
import tempfile
from pathlib import Path

from openpilot.common.file_chunker import get_existing_chunks, get_manifest_path
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id

MODELS_DIR = Path(__file__).resolve().parent / 'models'
TG_INPUT_DEVICES_PATH = MODELS_DIR / 'tg_input_devices.json'


def get_tg_input_devices(process_name: str, chestnut: bool):
  with open(TG_INPUT_DEVICES_PATH) as f:
    return json.load(f)[process_name]['default' if not chestnut else 'chestnut']

def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def precompiled_modeld_marker(model_path: Path) -> Path:
  return model_path.with_name(f'{model_path.stem}.precompiled.json')

def get_precompiled_modeld_files(model_path: Path) -> list[Path]:
  marker_path = precompiled_modeld_marker(model_path)
  if not marker_path.is_file():
    return []

  metadata = json.loads(marker_path.read_text())
  if (
    metadata.get('version') != 1 or
    metadata.get('artifact') != model_path.name or
    metadata.get('target') != 'chestnut' or
    metadata.get('backend') != 'USB+AMD:LLVM' or
    not metadata.get('model_checkpoint')
  ):
    raise ValueError(f'invalid precompiled modeld marker: {marker_path}')

  artifact_paths = [Path(path) for path in get_existing_chunks(model_path)]
  missing_paths = [path for path in artifact_paths if not path.is_file()]
  if missing_paths:
    raise FileNotFoundError(f'precompiled modeld artifact is incomplete: {missing_paths}')
  if metadata.get('files') != [path.name for path in artifact_paths]:
    raise ValueError(f'precompiled modeld files do not match {marker_path}')

  payload_paths = artifact_paths[1:] if artifact_paths[0].name.endswith('.chunkmanifest') else artifact_paths
  artifact_size = sum(path.stat().st_size for path in payload_paths)
  if metadata.get('artifact_size') != artifact_size:
    raise ValueError(f'precompiled modeld size does not match {marker_path}')
  return [marker_path, *artifact_paths]

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
  return Path(get_manifest_path(modeld_pkl_path(chestnut=True))).is_file()
