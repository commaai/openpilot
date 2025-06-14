import random, ctypes
import numpy as np
from tinygrad.device import Buffer, Device
from tinygrad.helpers import Context, getenv, from_mv
from tinygrad.dtype import dtypes
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.engine.realize import ExecItem, BufferXfer, get_runner
from tinygrad.engine.jit import apply_graph_to_jit

BUF_LEN = getenv("BUF_LEN", 128)

cached_prgs = {}
def gen_prg(device, inputs_cnt):
  if (device, inputs_cnt) in cached_prgs: return cached_prgs[(device, inputs_cnt)]

  with Context(DEBUG=0):
    fst = [Tensor.randn(BUF_LEN, dtype=dtypes.int).realize() for i in range(inputs_cnt)]
    s = fst[0]
    for i in range(1, inputs_cnt): s = s.bitwise_xor(fst[i])

    si = s.schedule()[-1]
    prg = get_runner(device, si.ast)
  cached_prgs[(device, inputs_cnt)] = prg
  return prg

def alloc_rawbuffer(device, fill=False):
  rawbuf = Buffer(device, BUF_LEN, dtypes.int).ensure_allocated()
  if fill:
    with Context(DEBUG=0):
      data = np.random.randint(-10000, 10000, size=rawbuf.size, dtype=_to_np_dtype(rawbuf.dtype))
      rawbuf.copyin(Tensor(data).realize().uop.base.realized.as_buffer())
  return rawbuf

def gen_kernel_ji(device, deps):
  assert len(deps) >= 2
  out = alloc_rawbuffer(device)
  prg = gen_prg(device, len(deps))
  return ExecItem(prg, [out] + deps)

def gen_copy_ji(device, deps):
  assert len(deps) == 1
  out = alloc_rawbuffer(device)
  prg = BufferXfer(deps[0].nbytes, device, deps[0].device)
  return ExecItem(prg, [out] + deps)

def gen_graph():
  input_buffers = []
  all_buffers = []
  jis = []

  last_n_deps = getenv("LAST_N_DEPS", 0)

  kernel_count = random.randint(2, getenv("MAX_KERNELS", 128))
  for i in range(kernel_count):
    target_device_id = random.randint(0, getenv("MAX_DEVICES", 6) - 1)
    target_device = f"{Device.DEFAULT}:{target_device_id}"
    is_copy = random.randint(0, 10) < 3

    if is_copy:
      deps_pool = [buf for buf in all_buffers[-last_n_deps:] if buf.device != target_device]
      if len(deps_pool) == 0: deps = []
      else: deps = random.sample(deps_pool, 1)
    else:
      deps_pool = [buf for buf in all_buffers[-last_n_deps:] if buf.device == target_device]
      deps_count = random.randint(0, min(getenv("MAX_DEPS_COUNT", 6), len(deps_pool)))
      if deps_count == 0: deps = []
      else: deps = random.sample(deps_pool, deps_count)

    if len(deps) == 0 or (not is_copy and len(deps) < 2):
      buf = alloc_rawbuffer(target_device, fill=True)
      input_buffers.append(buf)
      all_buffers.append(buf)
    elif is_copy:
      jis.append(gen_copy_ji(target_device, deps))
      all_buffers.append(jis[-1].bufs[0])
    else:
      jis.append(gen_kernel_ji(target_device, deps))
      all_buffers.append(jis[-1].bufs[0])

  return jis, all_buffers, input_buffers

def run_jit(jis, all_buffers, input_buffers, var_vals):
  with Context(DEBUG=0):
    for rawbuf in all_buffers:
      if rawbuf in input_buffers: continue
      mv = memoryview(bytearray(rawbuf.size * rawbuf.dtype.itemsize))
      ctypes.memset(from_mv(mv), 0, len(mv))
      rawbuf.copyin(mv)

  for ei in jis: ei.run(var_vals, jit=True)

  with Context(DEBUG=0):
    res_buffers = []
    for rawbuf in all_buffers: res_buffers.append(rawbuf.as_buffer())
    return res_buffers

def fuzz_graph(jis, all_buffers, input_buffers):
  ground_thruth_bufs = run_jit(jis, input_buffers, all_buffers, {})
  ground_truth_np = [np.frombuffer(x, _to_np_dtype(all_buffers[i].dtype)) for i,x in enumerate(ground_thruth_bufs)]

  for _ in range(getenv("FUZZ_GRAPH_SPLIT_RUNS", 64)):
    max_split_points = len(jis) // 3
    split_points = random.randint(0, min(max_split_points, getenv("FUZZ_GRAPH_MAX_SPLITS", 8)))
    split = [0]
    for i in range(split_points - 1):
      split.append(random.randint(split[-1] + 2, len(jis) - 2 * (max_split_points - i)))
    split.append(len(jis))

    graphed_jit = []
    for sp in range(len(split)-1):
      graphed_jit += apply_graph_to_jit(jis[split[sp]:split[sp+1]], [], {})

    for _ in range(getenv("FUZZ_GRAPH_SPLIT_RETRY_RUNS", 4)):
      test_bufs = run_jit(graphed_jit, input_buffers, all_buffers, {})
      test_bufs_np = [np.frombuffer(x, _to_np_dtype(all_buffers[i].dtype)) for i,x in enumerate(test_bufs)]
      for i in range(len(ground_thruth_bufs)): np.testing.assert_equal(ground_truth_np[i], test_bufs_np[i])

if __name__ == "__main__":
  SEED = getenv("SEED", 42)
  random.seed(SEED)
  np.random.seed(SEED)

  next_graph_id = 0
  for i in range(getenv("ITERS", 1000)):
    print("Running graph", next_graph_id)
    jis, all_buffers, input_buffers = gen_graph()
    fuzz_graph(jis, all_buffers, input_buffers)
    next_graph_id += 1
