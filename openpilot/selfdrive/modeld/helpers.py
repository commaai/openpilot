import io
import pickle
import shutil
import struct
import tempfile
import time
from collections.abc import Callable
from pathlib import Path
import numpy as np
from tinygrad import Tensor, TinyJit, Device
from tinygrad.nn.state import get_parameters

from openpilot.common.file_chunker import get_manifest_path
from openpilot.common.hardware.usb import CHESTNUT_USB_PRODUCT, USB_DEVICES_PATH, is_chestnut_usb_id

MODELS_DIR = Path(__file__).resolve().parent / 'models'


def modeld_pkl_path(chestnut: bool):
  prefix = 'big_' if chestnut else ''
  return MODELS_DIR / f'{prefix}driving_tinygrad.pkl'

def dump_pickle(obj, f, *, out_of_band=False):
  if not out_of_band:
    return pickle.dump(obj, f)
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

def load_pickle(f, *, out_of_band=False):
  if not out_of_band:
    return pickle.load(f)
  opcodes = f.read(struct.unpack('<q', f.read(8))[0])
  def buffers():
    while (h := f.read(8)):
      pb = pickle.PickleBuffer(bytearray(struct.unpack('<q', h)[0]))
      if f.readinto(pb) != pb.raw().nbytes:
        raise EOFError("incomplete model buffer")
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


def allocate_inputs(input_specs, packed_specs, initialize=None):
  """Allocate inputs and NumPy views, initializing before copying to devices."""
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in input_specs.items()}
  views = arrays.copy()
  if packed_specs:
    packed = views.pop('packed_inputs')
    views.update({name: packed[start:start+int(np.prod(shape))*np.dtype(dtype).itemsize].view(dtype).reshape(shape)
                  for name, (start, shape, dtype) in packed_specs.items()})
  if initialize is not None:
    initialize(views)
  return {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in input_specs.items()}, views


def compile_jit(function:Callable, make_inputs:Callable[[int], tuple[tuple, dict]], benchmark_runs=20, *, out_of_band=False):
  """The factory creates fresh inputs, including any mutable state, for each seed."""
  if benchmark_runs < 1:
    raise ValueError("benchmark_runs must be at least 1")
  jit = TinyJit(function, prune=True)

  def run(fn, seed, count):
    args, kwargs = make_inputs(seed)
    result = None
    for i in range(count):
      Device.default.synchronize()
      start = time.perf_counter()
      output = fn(*args, **kwargs)
      Device.default.synchronize()
      print(f"  [{i+1}/{count}] {(time.perf_counter()-start)*1e3:.2f} ms")
      if i == 0:
        result = [t.numpy().copy() for t in get_parameters(output)], [t.numpy().copy() for t in get_parameters((args, kwargs))]
    return result

  expected = run(jit, 42, 3)
  with tempfile.TemporaryFile(dir=".") as f:
    dump_pickle(jit, f, out_of_band=out_of_band)
    f.seek(0)
    loaded = load_pickle(f, out_of_band=out_of_band)
  for seed in (42, 43):
    reference = expected if seed == 42 else run(function, seed, 1)
    actual = run(loaded, seed, benchmark_runs)
    for ref_group, actual_group in zip(reference, actual, strict=True):
      for ref, value in zip(ref_group, actual_group, strict=True):
        np.testing.assert_array_equal(ref, value)
  # Preserve shared weight buffers when several JITs are saved in one artifact.
  return jit


def _patch_tinygrad_fetch_fw():
  import hashlib
  import pathlib
  import zstandard
  from tinygrad import helpers
  _orig = helpers.fetch_fw
  def fetch_fw(path, name, sha256):
    p = pathlib.Path(f"/lib/firmware/{path}/{name}.zst")
    if p.is_file():
      blob = zstandard.ZstdDecompressor().stream_reader(p.read_bytes()).read()
      if hashlib.sha256(blob).hexdigest() == sha256:
        return blob
    return _orig(path, name, sha256)
  helpers.fetch_fw = fetch_fw
_patch_tinygrad_fetch_fw()
