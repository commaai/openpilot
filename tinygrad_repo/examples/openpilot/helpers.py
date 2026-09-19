"""Shared compilation and input allocation for model and warp artifacts."""
import io, pathlib, pickle, shutil, struct, time, tempfile
import numpy as np
from typing import Callable
from tinygrad import Tensor, Device, Context, dtypes
from tinygrad.device import Buffer
from tinygrad.uop.ops import PatternMatcher, UPat, Ops, graph_rewrite
from tinygrad.nn.state import get_parameters

def allocate_inputs(input_specs, initialize=None):
  """Initialize inputs before copying to devices."""
  arrays = {name: np.zeros(shape, dtype=dtype) for name, (shape, dtype, _) in input_specs.items()}
  if initialize is not None: initialize(arrays)
  return {name: Tensor(arrays[name], device=device).realize() for name, (_, _, device) in input_specs.items()}

def dump_pickle(obj, path):
  with tempfile.TemporaryFile(dir=".") as buffers:
    opcodes, offset = io.BytesIO(), 0

    def persistent_id(b):
      nonlocal offset
      if not isinstance(b, Buffer) or b.device != Device.DEFAULT: return None
      offset += buffers.write(bytes((-offset) % 256))
      offset = (off:=offset) + buffers.write(b.as_memoryview())
      return b.size, b.dtype, off

    p = pickle.Pickler(opcodes)
    p.persistent_id = persistent_id
    p.dump(obj)
    with open(path, "wb") as f:
      f.write(struct.pack('<q', len(opcodes_b:=opcodes.getvalue())))
      f.write(opcodes_b)
      buffers.seek(0)
      shutil.copyfileobj(buffers, f)

def load_pickle(path, *, out_of_band=False):
  with open(path, "rb") as f:
    if not out_of_band: return pickle.load(f)
    opcodes, buffers = f.read(struct.unpack('<q', f.read(8))[0]), Tensor(pathlib.Path(path))[f.tell():].uop.buffer.ensure_allocated()

  # FIXME: we load in chunks here because hcq_submit for one large copy is very slow to compile
  arena, CHUNK_SIZE = Buffer(Device.DEFAULT, buffers.nbytes, dtypes.uchar, preallocate=True), 32 << 20
  for off in range(0, buffers.nbytes, CHUNK_SIZE):
    size = min(CHUNK_SIZE, buffers.nbytes-off)
    arena.view(size, dtypes.uchar, off).ensure_allocated().copy_from(buffers.view(size, dtypes.uchar, off).ensure_allocated())

  def persistent_load(pid): return arena.view(*pid)

  u = pickle.Unpickler(io.BytesIO(opcodes))
  u.persistent_load = persistent_load
  return u.load()

@Context(OPENPILOT_HACKS=1, **{'AMD': {'TC_OPT': 2, 'TC_MIN_GLOBALS': 32}}.get(Device.DEFAULT, {}))
def benchmark(fxn:Callable, cb=None, **kwargs):
  Device.default.synchronize()
  start = time.perf_counter()
  if (output := fxn(**kwargs)) is not None: output.realize()
  Device.default.synchronize()
  end = time.perf_counter()
  if cb: cb(end-start)
  return [t.numpy().copy() for t in get_parameters(kwargs.get('output_buffers', output))]

pm_retargetable = PatternMatcher([
  (UPat(Ops.PROGRAM, src=(UPat(), UPat(), UPat(), UPat()), name="p"), lambda p: p.replace(src=p.src[:-1]) if p.arg.target.device == "CPU" else None)
])

def make_retargetable(jit): jit.captured._linear = graph_rewrite(jit.captured._linear, pm_retargetable, walk=True, enter_calls=True)
