import numpy as np
import functools, unittest, ctypes

from tinygrad.device import Device, Buffer
from tinygrad.tensor import Tensor, _to_np_dtype
from tinygrad.helpers import Context, CI, dedup, from_mv
from tinygrad.dtype import dtypes
from tinygrad.engine.jit import MultiGraphRunner
from tinygrad.engine.realize import ExecItem, BufferXfer, get_runner, CompiledRunner

np.random.seed(1337)
Tensor.manual_seed(1337)
BUF_SIZE = 4096 if CI else 4096 * 128
RUN_CNT = 4 if CI else 32

cached_prgs = {}
def helper_exec_op(device, outbuf, inbufs):
  if (device, len(inbufs)) not in cached_prgs:
    with Context(DEBUG=0):
      fst = [Tensor.randn(BUF_SIZE, dtype=dtypes.int).realize() for i in range(len(inbufs))]
      s = fst[0]
      for i in range(1, len(inbufs)): s = s.bitwise_xor(fst[i])

      si = s.schedule()[-1]
      prg = get_runner(device, si.ast)
    cached_prgs[(device, len(inbufs))] = prg

  return ExecItem(cached_prgs[(device, len(inbufs))], [outbuf] + inbufs)

def helper_copy_op(device, dest, src):
  prg = BufferXfer(dest.nbytes, device, src.device)
  return ExecItem(prg, [dest, src])

def helper_alloc_rawbuffer(device, fill=False):
  rawbuf = Buffer(device, BUF_SIZE, dtypes.int).ensure_allocated()
  if fill:
    with Context(DEBUG=0):
      data = np.random.randint(-10000, 10000, size=rawbuf.size, dtype=_to_np_dtype(rawbuf.dtype))
      rawbuf.copyin(Tensor(data).realize().lazydata.base.realized.as_buffer())
  return rawbuf

def helper_create_offset_rawbuffer(base, offset=0):
  x = Buffer(base.device, base.size-offset, base.dtype, base=base, offset=offset)
  return x.ensure_allocated()

def helper_run_jit(jis, bufs, out_buffers):
  for rawbuf in out_buffers:
    mv = memoryview(bytearray(rawbuf.size * rawbuf.dtype.itemsize))
    ctypes.memset(from_mv(mv), 0, len(mv))
    rawbuf.copyin(mv)

  for ei in jis: ei.run({}, jit=True)
  return [rawbuf.as_buffer() for rawbuf in bufs]

def helper_test_graphs(graph_impl, graphs, runs=RUN_CNT):
  reg_ji = []
  bufs = []
  out_buffers = set()
  for graph in graphs:
    for ji in graph:
      out_buffers.update([ji.bufs[i] for i in (ji.prg.p.outs if isinstance(ji.prg, CompiledRunner) else [0])])
      bufs += ji.bufs
      reg_ji.append(ji)
  bufs = dedup(bufs)

  ground_thruth_bufs = helper_run_jit(reg_ji, bufs, out_buffers)
  ground_truth_np = [np.frombuffer(x, _to_np_dtype(bufs[i].dtype)) for i,x in enumerate(ground_thruth_bufs)]

  # Build graphs
  gr_ji = [ExecItem(graph_impl(graph, [], {}), []) for graph in graphs]

  for _ in range(runs):
    test_bufs = helper_run_jit(gr_ji, bufs, out_buffers)
    test_bufs_np = [np.frombuffer(x, _to_np_dtype(bufs[i].dtype)) for i,x in enumerate(test_bufs)]
    for i in range(len(ground_thruth_bufs)): np.testing.assert_equal(ground_truth_np[i], test_bufs_np[i])

@unittest.skipUnless(Device[Device.DEFAULT].graph is not None, "graph support required")
class TestGraph(unittest.TestCase):
  def test_order_2_writes_to_same_buf(self):
    d0 = Device.DEFAULT
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(5)]

    graphs = [
      [helper_exec_op(d0, b0[0], [b0[1], b0[2]]), helper_exec_op(d0, b0[0], [b0[3], b0[4]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_order_read_write_same_buf(self):
    d0 = Device.DEFAULT
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(5)]

    graphs = [
      [helper_exec_op(d0, b0[0], [b0[1], b0[2]]), helper_exec_op(d0, b0[1], [b0[3], b0[4]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_order_write_read_same_buf(self):
    d0 = Device.DEFAULT
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(5)]

    graphs = [
      [helper_exec_op(d0, b0[0], [b0[1], b0[2]]), helper_exec_op(d0, b0[1], [b0[0], b0[4]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def skip_if_not_multigraph(self):
    graph = g.func if isinstance(g:=Device[Device.DEFAULT].graph, functools.partial) else g
    if not issubclass(graph, MultiGraphRunner): self.skipTest("graph is not supported (not MultiGraphRunner)")

  def test_order_copy_writed(self):
    self.skip_if_not_multigraph()

    d0 = Device.DEFAULT
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(4)]

    graphs = [
      [helper_exec_op(d0, b0[0], [b0[1], b0[2]]), helper_copy_op(d0, b0[3], b0[0])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_order_copy_then_read(self):
    self.skip_if_not_multigraph()

    d0 = Device.DEFAULT
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(4)]

    graphs = [
      [helper_copy_op(d0, b0[1], b0[0]), helper_exec_op(d0, b0[3], [b0[1], b0[2]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_read_write_several_graphs(self):
    d0 = Device.DEFAULT
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(8)]

    graphs = [
      [helper_exec_op(d0, b0[3], [b0[1], b0[2]])],
      [helper_exec_op(d0, b0[4], [b0[1], b0[3]])],
      [helper_exec_op(d0, b0[5], [b0[4], b0[2]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

    graphs = [
      [helper_exec_op(d0, b0[3], [b0[1], b0[2]]), helper_exec_op(d0, b0[4], [b0[1], b0[2]]), helper_exec_op(d0, b0[5], [b0[1], b0[2]])],
      [helper_exec_op(d0, b0[2], [b0[6], b0[7]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_copies_2_devs(self):
    self.skip_if_not_multigraph()

    d0, d1 = Device.DEFAULT, f"{Device.DEFAULT}:1"
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(3)]
    b1 = [helper_alloc_rawbuffer(d1, fill=True) for _ in range(1)]

    graphs = [
      [helper_copy_op(d0, b1[0], b0[0]), helper_exec_op(d0, b0[2], [b0[0], b0[1]])]
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_copies_after_graph_global(self):
    self.skip_if_not_multigraph()

    d0, d1, d2, d3 = Device.DEFAULT, f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3"
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(8)]
    b1 = [helper_alloc_rawbuffer(d1, fill=True) for _ in range(6)]
    b2 = [helper_alloc_rawbuffer(d2, fill=True) for _ in range(6)]
    b3 = [helper_alloc_rawbuffer(d3, fill=True) for _ in range(6)]

    graphs = [
      [helper_exec_op(d0, b0[2], [b0[0], b0[1]]), helper_exec_op(d0, b0[3], [b0[0], b0[2]]), helper_exec_op(d0, b0[4], [b0[3], b0[2]]),
       helper_exec_op(d0, b0[5], [b0[0], b0[2]]), helper_exec_op(d0, b0[6], [b0[1], b0[2]]), helper_exec_op(d0, b0[7], [b0[0], b0[2]])],
      [helper_copy_op(d1, b0[2], b1[0])],
      [helper_exec_op(d0, b0[2], [b0[0], b0[1]]), helper_exec_op(d0, b0[3], [b0[0], b0[2]]), helper_exec_op(d0, b0[4], [b0[3], b0[2]]),
       helper_exec_op(d0, b0[5], [b0[0], b0[2]]), helper_exec_op(d0, b0[6], [b0[1], b0[2]]), helper_exec_op(d0, b0[7], [b0[0], b0[2]])],
      [helper_copy_op(d3, b0[2], b3[0])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

    graphs = [
      [helper_exec_op(d0, b0[2], [b0[0], b0[1]]), helper_exec_op(d0, b0[3], [b0[0], b0[2]]), helper_exec_op(d0, b0[4], [b0[3], b0[2]]),
       helper_exec_op(d0, b0[5], [b0[0], b0[2]]), helper_copy_op(d0, b2[0], b0[2]), helper_copy_op(d0, b2[1], b0[5]),
       helper_exec_op(d0, b0[7], [b0[0], b0[2]])],
      [helper_copy_op(d1, b0[2], b1[0])],
      [helper_exec_op(d0, b0[2], [b0[0], b0[1]])],
      [helper_copy_op(d3, b0[2], b3[0])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

    graphs = [
      [helper_exec_op(d0, b0[2], [b0[0], b0[1]]), helper_exec_op(d0, b0[3], [b0[0], b0[2]]), helper_exec_op(d0, b0[4], [b0[3], b0[2]]),
       helper_exec_op(d0, b0[5], [b0[0], b0[2]]), helper_copy_op(d0, b2[0], b0[2]), helper_copy_op(d0, b2[1], b0[5]),
       helper_exec_op(d0, b0[7], [b0[0], b0[2]])],
      [helper_copy_op(d1, b0[5], b1[0])],
      [helper_copy_op(d3, b0[5], b3[0])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

    graphs = [
      [helper_copy_op(d1, b0[5], b1[0])],
      [helper_copy_op(d3, b0[5], b3[0])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_graph_after_copies_devs(self):
    self.skip_if_not_multigraph()

    d0, d1, d2, d3 = Device.DEFAULT, f"{Device.DEFAULT}:1", f"{Device.DEFAULT}:2", f"{Device.DEFAULT}:3"
    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(8)]
    b1 = [helper_alloc_rawbuffer(d1, fill=True) for _ in range(1)]
    b2 = [helper_alloc_rawbuffer(d2, fill=True) for _ in range(2)]
    b3 = [helper_alloc_rawbuffer(d3, fill=True) for _ in range(2)]

    graphs = [
      [helper_copy_op(d1, b0[0], b1[0])],
      [helper_copy_op(d2, b0[1], b2[0]), helper_copy_op(d3, b0[2], b3[0])],
      [helper_exec_op(d0, b0[3], [b0[0], b0[2]]), helper_exec_op(d0, b0[4], [b0[3], b0[2]]),
       helper_exec_op(d0, b0[5], [b0[0], b0[2]])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

    graphs = [
      [helper_copy_op(d1, b0[0], b1[0])],
      [helper_exec_op(d0, b0[2], [b0[0], b0[1]])],
      [helper_copy_op(d2, b0[1], b2[0]), helper_copy_op(d3, b0[2], b3[0])],
      [helper_exec_op(d0, b0[3], [b0[0], b0[2]]), helper_exec_op(d0, b0[4], [b0[3], b0[2]]),
       helper_exec_op(d0, b0[5], [b0[0], b0[2]])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

  def test_graph_offset_bufs(self):
    self.skip_if_not_multigraph()

    d0 = Device.DEFAULT
    if not hasattr(Device[d0].allocator, "_offset"): self.skipTest("device does not support _offset")

    b0 = [helper_alloc_rawbuffer(d0, fill=True) for _ in range(1)]
    b0 += [helper_create_offset_rawbuffer(b0[0]), helper_create_offset_rawbuffer(b0[0])]

    graphs = [
      [helper_copy_op(d0, b0[0], b0[2]), helper_exec_op(d0, b0[1], [b0[0], b0[2]])],
    ]

    helper_test_graphs(Device[d0].graph, graphs)

if __name__ == '__main__':
  unittest.main()
