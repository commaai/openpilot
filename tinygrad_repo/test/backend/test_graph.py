import numpy as np
import functools, unittest, ctypes

from tinygrad.device import Device, Buffer
from tinygrad.tensor import Tensor
from tinygrad.helpers import Context, from_mv
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.engine.realize import run_linear, compile_linear
from tinygrad.uop.ops import UOp, Ops, buffers

from test.helpers import needs_second_gpu

np.random.seed(1337)
Tensor.manual_seed(1337)
BUF_SIZE = 4096
RUN_CNT = 5

# cache AST by (device, num_inputs)
cached_asts: dict[tuple[str, int], UOp] = {}
def get_ast(device:str, num_inputs:int) -> UOp:
  if (device, num_inputs) not in cached_asts:
    with Context(DEBUG=0):
      fst = [Tensor.randn(BUF_SIZE, dtype=dtypes.int).realize() for _ in range(num_inputs)]
      s = fst[0]
      for i in range(1, num_inputs): s = s.bitwise_xor(fst[i])
      cached_asts[(device, num_inputs)] = s.schedule_linear().src[-1].src[0]
  return cached_asts[(device, num_inputs)]

def make_buffer(device, size=BUF_SIZE, fill=False):
  buf = Buffer(device, size, dtypes.int).ensure_allocated()
  if fill:
    with Context(DEBUG=0):
      buf.copyin(Tensor(np.random.randint(-10000, 10000, size=size, dtype=np.int32)).realize().uop.base.realized.as_memoryview())
  return buf

def make_view(base, offset_elems, size_elems):
  return Buffer(base.device, size_elems, base.dtype, base=base, offset=offset_elems * base.dtype.itemsize).ensure_allocated()

def get_buf_uop(buf:Buffer, cache:dict[Buffer,UOp]) -> UOp:
  if buf not in cache:
    cache[buf] = u = UOp.new_buffer(buf.device, buf.size, buf.dtype)
    buffers[u] = buf
  return cache[buf]

def make_graph(graph_cls, calls:list[UOp]):
  linear = compile_linear(UOp(Ops.LINEAR, src=tuple(calls)))
  cf = UOp(Ops.CUSTOM_FUNCTION, dtypes.void, src=(linear,), arg="graph")
  return graph_cls(cf, [])

def run_schedule(calls:list[UOp]):
  run_linear(UOp(Ops.LINEAR, src=tuple(calls)))

def zero_bufs(bufs):
  for b in bufs:
    mv = memoryview(bytearray(b.nbytes))
    ctypes.memset(from_mv(mv), 0, len(mv))
    b.copyin(mv)

@unittest.skipUnless(Device[Device.DEFAULT].graph is not None, "graph support required")
class TestGraph(unittest.TestCase):
  def skip_if_no_offset(self):
    if not hasattr(Device[Device.DEFAULT].allocator, "_offset"): self.skipTest("device does not support _offset")

  def skip_if_not_multigraph(self):
    graph = g.func if isinstance(g:=(d:=Device[Device.DEFAULT]).graph, functools.partial) else g
    if not issubclass(graph, MultiGraphRunner): self.skipTest("graph is not supported (not MultiGraphRunner)")
    if not hasattr(d.allocator, '_transfer') or not d.allocator.supports_transfer: self.skipTest("device is not supported (no transfers)")

  def test_order_2_writes_to_same_buf(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(5)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[3],c), get_buf_uop(b[4],c), metadata=()),
    ]

    zero_bufs([b[0]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_order_read_write_same_buf(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(5)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(b[1],c), get_buf_uop(b[3],c), get_buf_uop(b[4],c), metadata=()),
    ]

    zero_bufs([b[0], b[1]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0], b[1]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_order_write_read_same_buf(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(5)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(b[1],c), get_buf_uop(b[0],c), get_buf_uop(b[4],c), metadata=()),
    ]

    zero_bufs([b[0], b[1]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0], b[1]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_order_copy_writed(self):
    self.skip_if_not_multigraph()
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(4)]
    c: dict[Buffer,UOp] = {}

    calls = [
      get_ast(d0, 2).call(get_buf_uop(b[0],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c), metadata=()),
      UOp(Ops.COPY).call(get_buf_uop(b[3],c), get_buf_uop(b[0],c), metadata=()),
    ]

    zero_bufs([b[0], b[3]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[0], b[3]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_order_copy_then_read(self):
    self.skip_if_not_multigraph()
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(4)]
    c: dict[Buffer,UOp] = {}

    calls = [
      UOp(Ops.COPY).call(get_buf_uop(b[1],c), get_buf_uop(b[0],c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(b[3],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c), metadata=()),
    ]

    zero_bufs([b[1], b[3]])
    run_schedule(calls)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs([b[1], b[3]])
      make_graph(Device[d0].graph, calls)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_read_write_several_graphs(self):
    d0 = Device.DEFAULT
    b = [make_buffer(d0, fill=True) for _ in range(8)]
    c: dict[Buffer,UOp] = {}

    calls1 = [get_ast(d0, 2).call(get_buf_uop(b[3],c), get_buf_uop(b[1],c), get_buf_uop(b[2],c), metadata=())]
    calls2 = [get_ast(d0, 2).call(get_buf_uop(b[4],c), get_buf_uop(b[1],c), get_buf_uop(b[3],c), metadata=())]
    calls3 = [get_ast(d0, 2).call(get_buf_uop(b[5],c), get_buf_uop(b[4],c), get_buf_uop(b[2],c), metadata=())]

    out = [b[3], b[4], b[5]]
    zero_bufs(out)
    run_schedule(calls1 + calls2 + calls3)
    expected = [np.frombuffer(x.as_memoryview(), np.int32).copy() for x in b]

    for _ in range(RUN_CNT):
      zero_bufs(out)
      make_graph(Device[d0].graph, calls1)([], {})
      make_graph(Device[d0].graph, calls2)([], {})
      make_graph(Device[d0].graph, calls3)([], {})
      for i, buf in enumerate(b): np.testing.assert_equal(expected[i], np.frombuffer(buf.as_memoryview(), np.int32))

  @needs_second_gpu
  def test_copies_2_devs(self):
    self.skip_if_not_multigraph()
    d0, d1 = Device.DEFAULT, f"{Device.DEFAULT}:1"
    b0 = [make_buffer(d0, fill=True) for _ in range(3)]
    b1 = [make_buffer(d1, fill=True)]
    c: dict[Buffer,UOp] = {}

    calls = [
      UOp(Ops.COPY).call(get_buf_uop(b1[0],c), get_buf_uop(b0[0],c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(b0[2],c), get_buf_uop(b0[0],c), get_buf_uop(b0[1],c), metadata=()),
    ]

    out = [b1[0], b0[2]]
    zero_bufs(out)
    run_schedule(calls)
    expected = {buf: np.frombuffer(buf.as_memoryview(), np.int32).copy() for buf in b0 + b1}

    for _ in range(RUN_CNT):
      zero_bufs(out)
      make_graph(Device[d0].graph, calls)([], {})
      for buf in b0 + b1: np.testing.assert_equal(expected[buf], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_graph_offset_bufs(self):
    self.skip_if_not_multigraph()
    d0 = Device.DEFAULT
    if not hasattr(Device[d0].allocator, "_offset"): self.skipTest("device does not support _offset")

    b0 = make_buffer(d0, fill=True)
    b1 = make_view(b0, 0, b0.size)
    b2 = make_view(b0, 0, b0.size)
    c: dict[Buffer,UOp] = {}

    calls = [
      UOp(Ops.COPY).call(get_buf_uop(b0,c), get_buf_uop(b2,c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(b1,c), get_buf_uop(b0,c), get_buf_uop(b2,c), metadata=()),
    ]

    zero_bufs([b0])
    run_schedule(calls)
    expected = np.frombuffer(b0.as_memoryview(), np.int32).copy()

    for _ in range(RUN_CNT):
      zero_bufs([b0])
      make_graph(Device[d0].graph, calls)([], {})
      np.testing.assert_equal(expected, np.frombuffer(b0.as_memoryview(), np.int32))

  def test_partial_write_preserves_write_dep(self):
    self.skip_if_not_multigraph()
    self.skip_if_no_offset()
    d0 = Device.DEFAULT

    base = make_buffer(d0, BUF_SIZE * 2, fill=True)
    copy_src_full = make_buffer(d0, BUF_SIZE * 2, fill=True)
    copy_src_lo = make_buffer(d0, fill=True)
    v_lo, v_hi = make_view(base, 0, BUF_SIZE), make_view(base, BUF_SIZE, BUF_SIZE)
    a, out = make_buffer(d0, fill=True), make_buffer(d0, fill=True)
    c: dict[Buffer,UOp] = {}

    calls = [
      UOp(Ops.COPY).call(get_buf_uop(base,c), get_buf_uop(copy_src_full,c), metadata=()),
      UOp(Ops.COPY).call(get_buf_uop(v_lo,c), get_buf_uop(copy_src_lo,c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(out,c), get_buf_uop(v_hi,c), get_buf_uop(a,c), metadata=()),
    ]

    zero_bufs([base, out])
    run_schedule(calls)
    expected = {base: np.frombuffer(base.as_memoryview(), np.int32).copy(), out: np.frombuffer(out.as_memoryview(), np.int32).copy()}

    for _ in range(RUN_CNT):
      zero_bufs([base, out])
      make_graph(Device[d0].graph, calls)([], {})
      for buf in [base, out]: np.testing.assert_equal(expected[buf], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_partial_write_preserves_read_dep(self):
    self.skip_if_not_multigraph()
    self.skip_if_no_offset()
    d0 = Device.DEFAULT

    base = make_buffer(d0, BUF_SIZE * 2, fill=True)
    copy_dst = make_buffer(d0, BUF_SIZE * 2, fill=True)
    copy_src_lo = make_buffer(d0, fill=True)
    v_lo, v_hi = make_view(base, 0, BUF_SIZE), make_view(base, BUF_SIZE, BUF_SIZE)
    a, b = make_buffer(d0, fill=True), make_buffer(d0, fill=True)
    c: dict[Buffer,UOp] = {}

    calls = [
      UOp(Ops.COPY).call(get_buf_uop(copy_dst,c), get_buf_uop(base,c), metadata=()),
      UOp(Ops.COPY).call(get_buf_uop(v_lo,c), get_buf_uop(copy_src_lo,c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(v_hi,c), get_buf_uop(a,c), get_buf_uop(b,c), metadata=()),
    ]

    zero_bufs([copy_dst, base])
    run_schedule(calls)
    expected = {copy_dst: np.frombuffer(copy_dst.as_memoryview(), np.int32).copy(), base: np.frombuffer(base.as_memoryview(), np.int32).copy()}

    for _ in range(RUN_CNT):
      zero_bufs([copy_dst, base])
      make_graph(Device[d0].graph, calls)([], {})
      for buf in [copy_dst, base]: np.testing.assert_equal(expected[buf], np.frombuffer(buf.as_memoryview(), np.int32))

  def test_middle_write_splits_write_dep(self):
    self.skip_if_not_multigraph()
    self.skip_if_no_offset()
    d0 = Device.DEFAULT

    base = make_buffer(d0, BUF_SIZE * 3, fill=True)
    copy_src_full = make_buffer(d0, BUF_SIZE * 3, fill=True)
    copy_src_mid = make_buffer(d0, fill=True)
    v_lo, v_mid, v_hi = make_view(base, 0, BUF_SIZE), make_view(base, BUF_SIZE, BUF_SIZE), make_view(base, BUF_SIZE * 2, BUF_SIZE)
    a, out1, out2 = make_buffer(d0, fill=True), make_buffer(d0, fill=True), make_buffer(d0, fill=True)
    c: dict[Buffer,UOp] = {}

    calls = [
      UOp(Ops.COPY).call(get_buf_uop(base,c), get_buf_uop(copy_src_full,c), metadata=()),
      UOp(Ops.COPY).call(get_buf_uop(v_mid,c), get_buf_uop(copy_src_mid,c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(out1,c), get_buf_uop(v_lo,c), get_buf_uop(a,c), metadata=()),
      get_ast(d0, 2).call(get_buf_uop(out2,c), get_buf_uop(v_hi,c), get_buf_uop(a,c), metadata=()),
    ]

    outs = [base, out1, out2]
    zero_bufs(outs)
    run_schedule(calls)
    expected = {buf: np.frombuffer(buf.as_memoryview(), np.int32).copy() for buf in outs}

    for _ in range(RUN_CNT):
      zero_bufs(outs)
      make_graph(Device[d0].graph, calls)([], {})
      for buf in outs: np.testing.assert_equal(expected[buf], np.frombuffer(buf.as_memoryview(), np.int32))

if __name__ == '__main__':
  unittest.main()
